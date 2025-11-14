import argparse
import os
import random
import time
import warnings
import sys
import zipfile
import urllib.request

import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torchvision.datasets.folder import default_loader

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import networks
from tensorboardX import SummaryWriter
from sklearn.metrics import confusion_matrix
from utils import *
from losses import *
from torch.amp import autocast
from torch.cuda.amp import GradScaler

model_names = sorted(name for name in networks.__dict__
    if name.islower() and not name.startswith("__")
    and callable(networks.__dict__[name]))
print(model_names)

parser = argparse.ArgumentParser(description='PyTorch Visual Classification Training')
parser.add_argument('--dataset', default='cifar10', help='dataset setting')
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet34', choices=model_names, help='model architecture: ' + ' | '.join(model_names) + ' (default: resnet32)')
parser.add_argument('--mode', default='norm', choices=['', 'norm', 'fix'], help='the mode of the last linear layer')
parser.add_argument('--loss_type', default="CE", type=str, help='loss type')
parser.add_argument('-s', '--scale', default=5, type=int, help='the scale of logits')
parser.add_argument('-j', '--workers', default=8, type=int, metavar='N', help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=250, type=int, metavar='N', help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N', help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=256, type=int, metavar='N', help='mini-batch size')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float, metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float, metavar='W', help='weight decay (default: 1e-4)', dest='weight_decay')
parser.add_argument('--scheduler', default='Cos', type=str, help='The scheduler')
parser.add_argument('-p', '--print-freq', default=500, type=int, metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH', help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true', help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', action='store_true', help='use pre-trained model')
parser.add_argument('--seed', default=123, type=int, help='seed for initializing training when deterministic mode is enabled.')
parser.add_argument('--deterministic', dest='use_seed', action='store_true',
                    help='Fix random seeds for reproducibility (default behaviour).')
parser.add_argument('--random-init', dest='use_seed', action='store_false',
                    help='Disable fixed seeds for potentially faster but non-deterministic training.')
parser.add_argument('--gpu', default=0, type=int, help='GPU id to use.')
parser.add_argument('--root_log', type=str, default='viclog')
parser.add_argument('--root_model', type=str, default='viccheckpoint')
parser.add_argument('--exp_str', default='0', type=str, help='number to indicate which experiment it is')
parser.add_argument('--reg_type', default='weight', type=str, help='regularization type')
parser.add_argument('--reg', type=float, default=5, help='the weight of regularization term')
parser.add_argument('--warmup-epochs', default=0, type=int, help='number of warm-up epochs (default: 0, disable warm-up)')
parser.add_argument('--amp', dest='amp', action='store_true', help='Enable Automatic Mixed Precision training.')
parser.add_argument('--no-amp', dest='amp', action='store_false', help='Disable Automatic Mixed Precision training.')
parser.add_argument('--save-freq', default=0, type=int,
                    help='epochs between checkpoint saves; <=0 表示仅在达到best时保存')
parser.add_argument('--disable-checkpoint', dest='disable_checkpoint', action='store_true',
                    help='禁用周期性checkpoint保存，仅在达到best时保存（默认启用）')
parser.add_argument('--enable-checkpoint', dest='disable_checkpoint', action='store_false',
                    help='启用周期性checkpoint保存，可配合save-freq控制频率')
parser.add_argument('--save-flush-freq', default=50, type=int,
                    help='累计checkpoint于内存，每N个epoch批量写盘；<=0 表示立即写盘')
parser.set_defaults(use_seed=True, amp=True, disable_checkpoint=True)
best_acc1 = 0

args = parser.parse_args()

if args.amp and not torch.cuda.is_available():
    warnings.warn('AMP requested but CUDA is not available; disabling AMP.')
    args.amp = False

if args.use_seed:
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    cudnn.deterministic = True
    cudnn.benchmark = False
else:
    cudnn.deterministic = False
    cudnn.benchmark = True


def norm_weights(weights):
    weights_norm = F.normalize(weights, dim=1)
    gravity = torch.sum(weights_norm, dim=0)
    return torch.sum(gravity ** 2)

def apply_warmup_lr(optimizer, epoch, batch_idx, num_batches, args):
    if args.warmup_epochs <= 0:
        return
    total_iters = max(1, args.warmup_epochs * num_batches)
    current_iter = epoch * num_batches + batch_idx + 1
    if current_iter > total_iters:
        return
    warmup_lr = args.lr * current_iter / total_iters
    for param_group in optimizer.param_groups:
        param_group['lr'] = warmup_lr

def step_scheduler_with_warmup(scheduler, epoch, args):
    if scheduler is None:
        return
    if args.warmup_epochs > 0:
        if epoch < args.warmup_epochs:
            return
        scheduler.step(epoch - args.warmup_epochs)
    else:
        scheduler.step()

# TINY_IMAGENET_URL = 'http://cs231n.stanford.edu/tiny-imagenet-200.zip'
# TINY_IMAGENET_ARCHIVE = 'tiny-imagenet-200.zip'
#
#
# def download_and_prepare_tinyimagenet(root):
#     dataset_dir = os.path.join(root, 'tiny-imagenet-200')
#     if os.path.isdir(dataset_dir):
#         return dataset_dir
#
#     os.makedirs(root, exist_ok=True)
#     archive_path = os.path.join(root, TINY_IMAGENET_ARCHIVE)
#
#     if not os.path.isfile(archive_path):
#         print(f"Downloading TinyImageNet from {TINY_IMAGENET_URL} ...")
#         try:
#             urllib.request.urlretrieve(TINY_IMAGENET_URL, archive_path)
#         except Exception as exc:
#             raise RuntimeError(f'Failed to download TinyImageNet: {exc}') from exc
#
#     print(f"Extracting {archive_path} ...")
#     with zipfile.ZipFile(archive_path, 'r') as zip_ref:
#         zip_ref.extractall(root)
#
#     return dataset_dir
#
#
# class TinyImageNetValDataset(torch.utils.data.Dataset):
#     def __init__(self, dataset_root, transform, wnid_to_idx):
#         self.transform = transform
#         self.loader = default_loader
#         val_dir = os.path.join(dataset_root, 'val')
#         images_dir = os.path.join(val_dir, 'images')
#         annotation_path = os.path.join(val_dir, 'val_annotations.txt')
#
#         if not os.path.exists(annotation_path):
#             raise FileNotFoundError(f'Cannot find TinyImageNet validation annotations at {annotation_path}')
#
#         self.samples = []
#         with open(annotation_path, 'r') as f:
#             for line in f:
#                 parts = line.strip().split('\t')
#                 if len(parts) < 2:
#                     continue
#                 image_name, wnid = parts[0], parts[1]
#                 image_path = os.path.join(images_dir, image_name)
#                 if not os.path.exists(image_path):
#                     continue
#                 label = wnid_to_idx[wnid]
#                 self.samples.append((image_path, label))
#
#         if not self.samples:
#             raise RuntimeError('TinyImageNet validation set appears to be empty.')
#
#         self.targets = [label for _, label in self.samples]
#
#     def __len__(self):
#         return len(self.samples)
#
#     def __getitem__(self, index):
#         path, target = self.samples[index]
#         img = self.loader(path)
#         if self.transform is not None:
#             img = self.transform(img)
#         return img, target
#
#
# def get_tinyimagenet_datasets(root, transform_train, transform_val):
#     dataset_dir = download_and_prepare_tinyimagenet(root)
#     wnids_path = os.path.join(dataset_dir, 'wnids.txt')
#     if not os.path.exists(wnids_path):
#         raise FileNotFoundError(f'Cannot find TinyImageNet wnids file at {wnids_path}')
#
#     with open(wnids_path, 'r') as f:
#         wnids = [line.strip() for line in f if line.strip()]
#     wnid_to_idx = {wnid: idx for idx, wnid in enumerate(wnids)}
#
#     train_dir = os.path.join(dataset_dir, 'train')
#     train_dataset = datasets.ImageFolder(train_dir, transform=transform_train)
#
#     remapped_samples = []
#     remapped_targets = []
#     for path, target in train_dataset.samples:
#         wnid = train_dataset.classes[target]
#         mapped_target = wnid_to_idx[wnid]
#         remapped_samples.append((path, mapped_target))
#         remapped_targets.append(mapped_target)
#
#     train_dataset.samples = remapped_samples
#     train_dataset.imgs = remapped_samples
#     train_dataset.targets = remapped_targets
#     train_dataset.classes = wnids
#     train_dataset.class_to_idx = wnid_to_idx
#
#     val_dataset = TinyImageNetValDataset(dataset_dir, transform_val, wnid_to_idx)
#     return train_dataset, val_dataset

def main():
    # Format learning rate for naming (replace dot with underscore, e.g., 0.1 -> 0_1, 0.01 -> 0_01)
    lr_str = f"lr{str(args.lr).replace('.', '_')}"
    args.store_name = '_'.join([
        args.dataset,
        args.arch,
        args.mode,
        args.loss_type,
        str(args.scale),
        args.exp_str,
        args.scheduler,
        args.reg_type,
        str(args.reg),
        lr_str,
        f'epochs{args.epochs}'
    ])
    args.dataset = args.dataset.lower()
    print(args)
    prepare_folders(args)
    if args.use_seed:
        warnings.warn('You have chosen to seed training.'
                      'This will turn on the CUDNN deterministic setting.'
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints')
    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')
    ngpus_per_node = torch.cuda.device_count()
    main_worker(args.gpu, ngpus_per_node, args)

def main_worker(gpu, ngpus_per_node, args):
    global best_acc1
    args.gpu = gpu
    if args.gpu is not None:
        print('Use GPU: {} for training'.format(args.arch))
    # create model
    print("=> creating model '{}'".format(args.arch))
    if args.dataset == 'cifar100':
        num_classes = 100
    elif args.dataset == 'tinyimagenet':
        num_classes = 200
    else:
        num_classes = 10
    mode = args.mode
    if args.loss_type in ['CE', 'Focal', 'L_Softmax', 'SphereFace']:
        mode = ''
    if mode == 'fix' and args.dataset == 'cifar10':
        weight = torch.Tensor(np.load('./weight10x512.npy')).cuda(args.gpu)
    elif mode == 'fix' and args.dataset == 'cifar100':
        weight = torch.Tensor(np.load('./weight100x512.npy')).cuda(args.gpu)
    else:
        weight = None
    if args.dataset == 'MNIST' or args.dataset == 'mnist':
        model = networks.cnn(mode=mode)
    else:
        model = networks.__dict__[args.arch](num_classes=num_classes, mode=mode, weight=weight)
    if args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
    else:
        model = torch.nn.DataParallel(model).cuda()
    optimizer = torch.optim.SGD(model.parameters(), args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    scaler = GradScaler(enabled=args.amp and torch.cuda.is_available())
    if args.scheduler == 'Cos':
        t_max = max(1, args.epochs - args.warmup_epochs) if args.warmup_epochs > 0 else args.epochs
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=t_max, eta_min=0.0)
    else:
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoints '{}'".format(args.resume))
            checkpoint = torch.load(args.resume, map_location='cuda:0')
            args.start_epoch = checkpoint['epoch']
            scheduler._step_count = args.start_epoch
            best_acc1 = checkpoint['best_acc1']
            if args.gpu is not None:
                best_acc1 = best_acc1.to(args.gpu)
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            if 'scaler' in checkpoint and checkpoint['scaler'] is not None and scaler is not None and scaler.is_enabled():
                scaler.load_state_dict(checkpoint['scaler'])
            print("=> loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))
    if args.use_seed:
        cudnn.benchmark = False
    else:
        cudnn.benchmark = True

    if args.dataset == 'mnist' or args.dataset == 'MNIST':
        MEAN = [0.1307]
        STD = [0.3081]
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD)])
        train_dataset = datasets.MNIST('../database/MNIST', train=True, download=True, transform=transform)
        val_dataset = datasets.MNIST('../database/MNIST', train=False, download=True, transform=transform)
    elif args.dataset == 'cifar10':
        CIFAR_MEAN = [0.49139968, 0.48215827, 0.44653124]
        CIFAR_STD = [0.24703233, 0.24348505, 0.26158768]

        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(CIFAR_MEAN, CIFAR_STD)])
        transform_val = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(CIFAR_MEAN, CIFAR_STD)])
        train_dataset = datasets.CIFAR10(root='../database/CIFAR10', train=True, download=True, transform=transform_train)
        val_dataset = datasets.CIFAR10(root='../database/CIFAR10', train=False, download=True, transform=transform_val)
    elif args.dataset == 'cifar100':
        CIFAR_MEAN = [0.5071, 0.4865, 0.4409]
        CIFAR_STD = [0.2673, 0.2564, 0.2762]
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(20),
            transforms.ToTensor(),
            transforms.Normalize(CIFAR_MEAN, CIFAR_STD)])

        transform_val = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(CIFAR_MEAN, CIFAR_STD)])
        train_dataset = datasets.CIFAR100(root='../database/CIFAR100', train=True, download=True,
                                         transform=transform_train)
        val_dataset = datasets.CIFAR100(root='../database/CIFAR100', train=False, download=True, transform=transform_val)
    # elif args.dataset == 'tinyimagenet':
    #     TINY_MEAN = [0.4802, 0.4481, 0.3975]
    #     TINY_STD = [0.2302, 0.2265, 0.2262]
    #     transform_train = transforms.Compose([
    #         transforms.RandomCrop(64, padding=8),
    #         transforms.RandomHorizontalFlip(),
    #         transforms.RandomRotation(20),
    #         transforms.ToTensor(),
    #         transforms.Normalize(TINY_MEAN, TINY_STD)])
    #
    #     transform_val = transforms.Compose([
    #         transforms.ToTensor(),
    #         transforms.Normalize(TINY_MEAN, TINY_STD)])
    #     tiny_root = os.path.join('../database', 'TinyImageNet')
    #     train_dataset, val_dataset = get_tinyimagenet_datasets(
    #         tiny_root, transform_train=transform_train, transform_val=transform_val
    #     )
    else:
        warnings.warn('Dataset is not listed!')
        return
    # train_loader = torch.utils.data.DataLoader(
    #     train_dataset, batch_size=args.batch_size, shuffle=True,
    #     num_workers=args.workers, pin_memory=True
    # )
    # val_loader = torch.utils.data.DataLoader(
    #     val_dataset, batch_size=args.batch_size, shuffle=False,
    #     num_workers=args.workers, pin_memory=True
    # )
    worker_has_multiprocessing = args.workers > 0
    loader_common_args = dict(
        num_workers=args.workers,
        pin_memory=True
    )
    if worker_has_multiprocessing:
        loader_common_args.update(
            persistent_workers=True,
            prefetch_factor=4
        )
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False,
        **loader_common_args
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        **loader_common_args
    )
    log_testing = open(os.path.join(args.root_log, args.store_name, 'log_test.csv'), 'w')
    with open(os.path.join(args.root_log, args.store_name, 'args.txt'), 'w') as f:
        f.write(str(args))
    if args.loss_type == 'CE':
        criterion = nn.CrossEntropyLoss()
    elif args.loss_type == 'Focal':
        criterion = FocalLoss(gamma=1)
    elif args.loss_type == 'Norm':
        criterion = NormFaceLoss(scale=args.scale)
    elif args.loss_type == 'LMSoftmax':
        criterion = LMSoftmaxLoss(scale=args.scale)
    elif args.loss_type == 'CosFace':
        criterion = CosFaceLoss(margin=0.1, scale=args.scale)
    elif args.loss_type == 'ArcFace':
        criterion = ArcFaceLoss(margin=0.1, scale=args.scale)
    else:
        warnings.warn('The loss type is not listed!')
        return
    # args.reg = args.reg * 100 / (num_classes ** 2)
    # tf_writer = SummaryWriter(log_dir=os.path.join(args.root_log, args.store_name))
    tf_writer = None
    # Record training start time for total elapsed time tracking
    pending_checkpoints = []
    for epoch in range(args.start_epoch, args.epochs):
        train(train_loader, model, criterion, optimizer, epoch, args, tf_writer, scaler)
        step_scheduler_with_warmup(scheduler, epoch, args)
        acc1 = validate(val_loader, model, criterion, epoch, args, log_testing, tf_writer)
        is_best = acc1 > best_acc1
        best_acc1 = max(acc1, best_acc1)
        # if tf_writer is not None:
        #     tf_writer.add_scalar('acc/test_top1_best', best_acc1, epoch)
        # 原始每个 epoch 都保存的逻辑：
        # save_checkpoint(args, {
        #     'epoch': epoch + 1,
        #     'arch': args.arch,
        #     'state_dict': model.state_dict(),
        #     'best_acc1': best_acc1,
        #     'optimizer': optimizer.state_dict(),
        #     'scaler': scaler.state_dict() if scaler is not None and scaler.is_enabled() else None,
        # }, is_best)

        periodic_enabled = (not args.disable_checkpoint) and (args.save_freq is None or args.save_freq > 0)
        should_save_periodic = periodic_enabled and (args.save_freq is None or (epoch + 1) % args.save_freq == 0)
        if is_best or should_save_periodic:
            checkpoint_payload = {
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'best_acc1': best_acc1,
                'optimizer': optimizer.state_dict(),
                'scaler': scaler.state_dict() if scaler is not None and scaler.is_enabled() else None,
            }
            # save_checkpoint(args, checkpoint_payload, is_best)
            pending_checkpoints.append((epoch + 1, checkpoint_payload, is_best))

        flush_freq = args.save_flush_freq
        if pending_checkpoints:
            if flush_freq is None or flush_freq <= 0:
                for _, payload, mark_best in pending_checkpoints:
                    save_checkpoint(args, payload, mark_best)
                pending_checkpoints.clear()
            elif (epoch + 1) % flush_freq == 0:
                for _, payload, mark_best in pending_checkpoints:
                    save_checkpoint(args, payload, mark_best)
                pending_checkpoints.clear()

    if pending_checkpoints:
        for _, payload, mark_best in pending_checkpoints:
            save_checkpoint(args, payload, mark_best)
        pending_checkpoints.clear()

    # sis = get_margin(train_loader, model)
    # if tf_writer is not None:
    #     tf_writer.add_histogram('similarity/train', sis)
    # sis = get_margin(val_loader, model)
    # if tf_writer is not None:
    #     tf_writer.add_histogram('similarity/test', sis)

def train(train_loader, model, criterion, optimizer, epoch, args, tf_writer, scaler):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    sample_margin = SampleMarginLoss() if args.reg_type == 'margin' else None
    model.train()
    end = time.time()
    num_batches = len(train_loader)
    for i, (input, target) in enumerate(train_loader):
        data_time.update(time.time()-end)
        apply_warmup_lr(optimizer, epoch, i, num_batches, args)
        if args.gpu is not None:
            input = input.cuda(args.gpu, non_blocking=True)
        target = target.cuda(args.gpu, non_blocking=True)
        optimizer.zero_grad()
        with autocast('cuda', enabled=args.amp and torch.cuda.is_available()):
            if args.loss_type in ['L_Softmax', 'SphereFace']:
                embedding = model.get_body(input)
                weight = model.get_weight()
                loss = criterion(weight, embedding, target)
                output = F.linear(embedding, weight)
                sm = None
                if args.reg_type == 'margin' and sample_margin is not None:
                    weight_norm = F.normalize(weight, dim=1)
                    embedding_norm = F.normalize(embedding, dim=1)
                    output_norm = F.linear(embedding_norm, weight_norm)
                    sm = sample_margin(output_norm, target)
            else:
                embedding = model.get_body(input)
                weight = model.get_weight()
                output = model.linear(embedding)
                sm = None
                if args.reg_type == 'margin' and sample_margin is not None:
                    weight_norm = F.normalize(weight, dim=1)
                    embedding_norm = F.normalize(embedding, dim=1)
                    output_norm = F.linear(embedding_norm, weight_norm)
                    sm = sample_margin(output_norm, target)
                if args.reg_type == 'weight':
                    reg = norm_weights(weight)
                elif args.reg_type == 'margin':
                    reg = sm
                else:
                    reg = 0
                loss = criterion(output, target) + args.reg * reg
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), input.size(0))
        top1.update(acc1[0], input.size(0))
        top5.update(acc5[0], input.size(0))

        if scaler is not None and scaler.is_enabled():
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

        batch_time.update(time.time() - end)
        end = time.time()
        # if i % args.print_freq == 0:
        #     current_lr = optimizer.param_groups[-1]['lr']
        #     output = (
        #         f"Epoch: [{epoch}/{args.epochs}][{i}/{len(train_loader)}], lr: {current_lr:.5f}\t"
        #         f"Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t"
        #         f"Data {data_time.val:.3f} ({data_time.avg:.3f})\t"
        #         f"Loss {losses.val:.4f} ({losses.avg:.4f})\t"
        #         f"Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t"
        #         f"Prec@5 {top5.val:.3f} ({top5.avg:.3f})\t"
        #         f"SaMargin {sample_margins.val:.3f} ({sample_margins.avg:.3f})"
        #     )
        #     print(output)
        #     log.write(output + '\n')
        #     log.flush()
    # margin, ratio = model.margin()
    return losses.avg, top1.avg, top5.avg
    # if tf_writer is not None:
    #     tf_writer.add_scalar('loss/train', losses.avg, epoch)
    #     tf_writer.add_scalar('acc/train_top1', top1.avg, epoch)
    #     tf_writer.add_scalar('acc/train_top5', top5.avg, epoch)
    #     tf_writer.add_scalar('lr', optimizer.param_groups[-1]['lr'], epoch)


def validate(val_loader, model, criterion, epoch, args, log=None, tf_writer=None, flag='val'):
    batch_time = AverageMeter('Time', ':6.3f')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    # switch to evaluate mode
    model.eval()
    # all_preds = []
    # all_targets = []
    # sample_margin = SampleMarginLoss()
    with torch.no_grad():
        end = time.time()
        for i, (input, target) in enumerate(val_loader):
            if args.gpu is not None:
                input = input.cuda(args.gpu, non_blocking=True)
            target = target.cuda(args.gpu, non_blocking=True)

            # compute output
            with autocast('cuda', enabled=args.amp and torch.cuda.is_available()):
                embedding = model.get_body(input)
                output = model.linear(embedding)
            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            top1.update(acc1[0], input.size(0))
            top5.update(acc5[0], input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            # _, pred = torch.max(output, 1)
            # all_preds.extend(pred.cpu().numpy())
            # all_targets.extend(target.cpu().numpy())

            # if i % args.print_freq == 0:
            #     output = ('Test: [{0}/{1}]\t'
            #               'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
            #               'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
            #               'Prec@5 {top5.val:.3f} ({top5.avg:.3f})\t'
            #           'SaMargin {sample_margin.val:.3f} ({sample_margin.avg:.3f})'.format(
            #         i, len(val_loader), batch_time=batch_time,
            #         top1=top1, top5=top5, sample_margin=sample_margins))
            #     print(output)
        # cf = confusion_matrix(all_targets, all_preds).astype(float)
        # cls_cnt = cf.sum(axis=1)
        # cls_hit = np.diag(cf)
        # cls_acc = cls_hit / cls_cnt
        summary = (
            '{flag} Results (Epoch {current}/{total}): Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}'
            .format(flag=flag, top1=top1, top5=top5, current=epoch + 1, total=args.epochs)
        )
        print(summary)
        if log is not None:
            log.write(summary + '\n')
            log.flush()

        # if tf_writer is not None:
        #     tf_writer.add_scalar('sample_margin/test_' + flag, -sample_margins.avg, epoch)
        #     tf_writer.add_scalar('acc/test_' + flag + '_top1', top1.avg, epoch)
        #     tf_writer.add_scalar('acc/test_' + flag + '_top5', top5.avg, epoch)
        #     tf_writer.add_scalars('acc/test_' + flag + '_cls_acc', {str(i): x for i, x in enumerate(cls_acc)}, epoch)

    return top1.avg



if __name__ == '__main__':
    main()
