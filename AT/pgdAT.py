import copy
import argparse
import os
import time
import sys

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

# ensure project root is importable like trainer.py
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import networks


def build_dataloaders(dataset: str, batch_size: int, workers: int):
    if dataset.lower() == 'cifar10':
        mean = [0.49139968, 0.48215827, 0.44653124]
        std = [0.24703233, 0.24348505, 0.26158768]
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
        train_dataset = torchvision.datasets.CIFAR10(root='../database/CIFAR10', train=True, download=True, transform=transform_train)
        test_dataset = torchvision.datasets.CIFAR10(root='../database/CIFAR10', train=False, download=True, transform=transform_test)
        num_classes = 10
    elif dataset.lower() == 'cifar100':
        mean = [0.5071, 0.4865, 0.4409]
        std = [0.2673, 0.2564, 0.2762]
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(20),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
        train_dataset = torchvision.datasets.CIFAR100(root='../database/CIFAR100', train=True, download=True, transform=transform_train)
        test_dataset = torchvision.datasets.CIFAR100(root='../database/CIFAR100', train=False, download=True, transform=transform_test)
        num_classes = 100
    else:
        raise ValueError('Unsupported dataset: {}'.format(dataset))

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=workers, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=max(100, batch_size), shuffle=False, num_workers=workers, pin_memory=True)
    return train_loader, test_loader, num_classes

class LinfPGDAttack(object):
    def __init__(self, model):
        self.model = model

    def perturb(self, x_natural, y, epsilon, alpha, k):
        x = x_natural.detach()
        x = x + torch.zeros_like(x).uniform_(-epsilon, epsilon)
        for i in range(k):
            x.requires_grad_()
            with torch.enable_grad():
                logits = self.model(x)
                loss = F.cross_entropy(logits, y)
            grad = torch.autograd.grad(loss, [x])[0]
            x = x.detach() + alpha * torch.sign(grad.detach())
            x = torch.min(torch.max(x, x_natural - epsilon), x_natural + epsilon)
            x = torch.clamp(x, 0, 1)
        return x

def attack(x, y, model, adversary, epsilon, alpha, k):
    model_copied = copy.deepcopy(model)
    model_copied.eval()
    adversary.model = model_copied
    adv = adversary.perturb(x, y, epsilon=epsilon, alpha=alpha, k=k)
    return adv

def main():
    # collect model names from networks like trainer.py
    model_names = sorted(name for name in networks.__dict__ if name.islower() and not name.startswith("__") and callable(networks.__dict__[name]))

    parser = argparse.ArgumentParser(description='PGD Adversarial Training (aligned with trainer.py)')
    parser.add_argument('--dataset', default='cifar10', choices=['cifar10', 'cifar100'])
    parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet34', choices=model_names)
    parser.add_argument('--mode', default='', choices=['', 'norm', 'fix'])
    parser.add_argument('-b', '--batch-size', default=256, type=int)
    parser.add_argument('--epochs', default=200, type=int)
    parser.add_argument('--workers', default=4, type=int)
    parser.add_argument('--lr', '--learning-rate', default=0.1, type=float, dest='lr')
    parser.add_argument('--momentum', default=0.9, type=float)
    parser.add_argument('--weight-decay', default=2e-4, type=float)
    parser.add_argument('--scheduler', default='Original', choices=['Step', 'Cos', 'Original'], help='Original: manual LR decay at epoch 100 and 150')
    parser.add_argument('--gpu', default=None, type=int)
    parser.add_argument('--out', default='viccheckpoint', type=str, help='checkpoint dir (like trainer)')
    parser.add_argument('--save-freq', default=20, type=int, help='save checkpoint every N epochs')
    parser.add_argument('--root_log', type=str, default='viclog', help='log root directory')
    parser.add_argument('--store_name', type=str, default='', help='subdirectory under root_log for this run')
    # PGD params
    parser.add_argument('--epsilon', default=0.0314, type=float)
    parser.add_argument('--pgd-steps', default=7, type=int)
    parser.add_argument('--alpha', default=0.00784, type=float)

    args = parser.parse_args()

    # device
    if args.gpu is not None and torch.cuda.is_available():
        torch.cuda.set_device(args.gpu)
        device = torch.device('cuda:{}'.format(args.gpu))
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # data
    train_loader, test_loader, num_classes = build_dataloaders(args.dataset, args.batch_size, args.workers)

    # model (aligned with trainer)
    weight = None
    model = networks.__dict__[args.arch](num_classes=num_classes, mode=args.mode, weight=weight)
    if args.gpu is not None and torch.cuda.is_available():
        model = model.cuda(args.gpu)
    else:
        model = torch.nn.DataParallel(model).to(device)
    cudnn.benchmark = True

    adversary = LinfPGDAttack(model)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    
    # Setup scheduler (Original uses manual adjustment instead)
    scheduler = None
    if args.scheduler == 'Cos':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=0.0)
    elif args.scheduler == 'Step':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
    # Original: manual LR adjustment at epoch 100 and 150 (implemented in training loop)

    best_acc1 = 0.0
    
    # Manual learning rate adjustment function (matches original code)
    def adjust_learning_rate(optimizer, epoch, initial_lr):
        """Original PGD training schedule: divide by 10 at epoch 100 and 150"""
        lr = initial_lr
        if epoch >= 100:
            lr /= 10
        if epoch >= 150:
            lr /= 10
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

    # logging setup (similar to trainer.py)
    if not args.store_name:
        args.store_name = '{}_{}_{}_pgd'.format(args.dataset, args.arch, args.mode if args.mode else 'std')
    log_dir = os.path.join(args.root_log, args.store_name)
    os.makedirs(log_dir, exist_ok=True)
    with open(os.path.join(log_dir, 'args.txt'), 'w') as f:
        f.write(str(args))
    log_training = open(os.path.join(log_dir, 'log_train.csv'), 'a')
    log_testing = open(os.path.join(log_dir, 'log_test.csv'), 'a')

    def train(epoch):
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            if args.gpu is not None and torch.cuda.is_available():
                inputs = inputs.cuda(args.gpu, non_blocking=True)
                targets = targets.cuda(args.gpu, non_blocking=True)
            else:
                inputs = inputs.to(device)
                targets = targets.to(device)

            optimizer.zero_grad()
            adv = adversary.perturb(inputs, targets, epsilon=args.epsilon, alpha=args.alpha, k=args.pgd_steps)
            outputs = model(adv)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            if batch_idx % 10 == 0:
                print('Train Ep {} [{}/{}] Adv Acc: {:.4f} Loss: {:.4f}'.format(
                    epoch, batch_idx, len(train_loader), predicted.eq(targets).float().mean().item(), loss.item()))
        train_acc = 100. * correct / total
        print('Train Ep {} Adv Acc: {:.2f} Loss: {:.4f}'.format(epoch, train_acc, train_loss))
        return train_acc, train_loss

    def validate(epoch):
        model.eval()
        benign_correct = 0
        adv_correct = 0
        total = 0
        benign_loss = 0.0
        adv_loss = 0.0
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(test_loader):
                if args.gpu is not None and torch.cuda.is_available():
                    inputs = inputs.cuda(args.gpu, non_blocking=True)
                    targets = targets.cuda(args.gpu, non_blocking=True)
                else:
                    inputs = inputs.to(device)
                    targets = targets.to(device)

                outputs = model(inputs)
                loss = criterion(outputs, targets)
                benign_loss += loss.item()
                _, predicted = outputs.max(1)
                benign_correct += predicted.eq(targets).sum().item()

                adv = adversary.perturb(inputs, targets, epsilon=args.epsilon, alpha=args.alpha, k=args.pgd_steps)
                adv_outputs = model(adv)
                loss = criterion(adv_outputs, targets)
                adv_loss += loss.item()
                _, predicted = adv_outputs.max(1)
                adv_correct += predicted.eq(targets).sum().item()

        total = len(test_loader.dataset)
        benign_acc = 100. * benign_correct / total
        adv_acc = 100. * adv_correct / total
        print('Test Ep {} Benign Acc: {:.2f} Adv Acc: {:.2f} Benign Loss: {:.4f} Adv Loss: {:.4f}'.format(
            epoch, benign_acc, adv_acc, benign_loss, adv_loss))
        return adv_acc, benign_acc, benign_loss, adv_loss

    os.makedirs(args.out, exist_ok=True)
    for epoch in range(0, args.epochs):
        epoch_start = time.time()
        # Update learning rate based on scheduler type
        if args.scheduler == 'Original':
            adjust_learning_rate(optimizer, epoch, args.lr)
        elif scheduler is not None:
            scheduler.step()
        
        train_acc, train_loss = train(epoch)
        acc1, benign_acc, benign_loss, adv_loss = validate(epoch)
        best_acc1 = max(best_acc1, acc1)
        elapsed = time.time() - epoch_start
        print('Epoch {} elapsed: {:.3f}s'.format(epoch, elapsed))
        # write logs (epoch, metrics, elapsed)
        log_training.write('{},{:.4f},{:.4f},{:.3f}\n'.format(epoch, train_acc, train_loss, elapsed))
        log_training.flush()
        log_testing.write('{},{:.4f},{:.4f},{:.4f},{:.3f}\n'.format(epoch, benign_acc, acc1, benign_loss, elapsed))
        log_testing.flush()
        # save checkpoint with same keys as trainer.py
        if (epoch + 1) % args.save_freq == 0 or (epoch + 1) == args.epochs:
            state = {
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'best_acc1': best_acc1,
                'optimizer': optimizer.state_dict(),
            }
            ckpt_path = os.path.join(args.out, 'pgd_{}_{}_epoch{}.pth'.format(args.dataset, args.arch, epoch + 1))
            torch.save(state, ckpt_path)
            print('Saved checkpoint to {}'.format(ckpt_path))

    # close logs
    log_training.close()
    log_testing.close()


if __name__ == '__main__':
    main()
