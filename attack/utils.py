import os
import random
import numpy as np
import PIL.Image
import time
import pickle
import matplotlib.pyplot as plt
import torchvision.datasets as datasets
import zipfile
import urllib.request
from collections import defaultdict
from torchvision.datasets.folder import default_loader



'''
return a margin of shape (n)
do not use this.
'''
def margin_loss_n(y, logits):
    a = (logits * y)
    rest = logits - a
    margin = a.max(1) - rest.max(1)
    return margin


'''
return a margin of shape (n,1)
'''
def margin_loss(y, logits):
    y = np.array(y, dtype=bool)
    preds_correct_class = (logits * y).sum(1, keepdims=True)
    diff = preds_correct_class - logits
    diff[y] = np.inf
    margin = diff.min(1, keepdims=True)
    return margin.flatten()


def get_time(): return time.strftime('%Y-%m-%d_%H-%M-%S', time.localtime(time.time()))

TINY_IMAGENET_URL = 'http://cs231n.stanford.edu/tiny-imagenet-200.zip'
TINY_IMAGENET_ARCHIVE = 'tiny-imagenet-200.zip'
TINY_IMAGENET_ROOT = '../database/TinyImageNet'


def download_and_prepare_tinyimagenet(root=TINY_IMAGENET_ROOT):
    dataset_dir = os.path.join(root, 'tiny-imagenet-200')
    if os.path.isdir(dataset_dir):
        return dataset_dir

    os.makedirs(root, exist_ok=True)
    archive_path = os.path.join(root, TINY_IMAGENET_ARCHIVE)

    if not os.path.isfile(archive_path):
        print(f"Downloading TinyImageNet from {TINY_IMAGENET_URL} ...")
        try:
            urllib.request.urlretrieve(TINY_IMAGENET_URL, archive_path)
        except Exception as exc:
            raise RuntimeError(f'下载 TinyImageNet 失败: {exc}') from exc

    print(f"Extracting {archive_path} ...")
    with zipfile.ZipFile(archive_path, 'r') as zip_ref:
        zip_ref.extractall(root)

    return dataset_dir


def _load_tinyimagenet_val_metadata(dataset_dir):
    wnids_path = os.path.join(dataset_dir, 'wnids.txt')
    if not os.path.exists(wnids_path):
        raise FileNotFoundError(f'无法找到 TinyImageNet 类别文件: {wnids_path}')
    with open(wnids_path, 'r') as f:
        wnids = [line.strip() for line in f if line.strip()]
    wnid_to_idx = {wnid: idx for idx, wnid in enumerate(wnids)}

    val_dir = os.path.join(dataset_dir, 'val')
    images_dir = os.path.join(val_dir, 'images')
    annotations_path = os.path.join(val_dir, 'val_annotations.txt')
    if not os.path.exists(annotations_path):
        raise FileNotFoundError(f'无法找到 TinyImageNet 验证标注文件: {annotations_path}')

    samples_by_class = defaultdict(list)
    with open(annotations_path, 'r') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) < 2:
                continue
            image_name, wnid = parts[0], parts[1]
            class_idx = wnid_to_idx.get(wnid)
            if class_idx is None:
                continue
            image_path = os.path.join(images_dir, image_name)
            if os.path.exists(image_path):
                samples_by_class[class_idx].append(image_path)

    return wnids, samples_by_class


class Logger:
    def __init__(self, path):
        self.path = path
        if path != '':
            folder = '/'.join(path.split('/')[:-1])
            if not os.path.exists(folder):
                os.makedirs(folder)

    def print(self, message):
        print(message)
        if self.path != '':
            with open(self.path, 'a') as f:
                f.write(message + '\n')
                f.flush()


def sample_cifar10_every_class(random_seed=0, data_amount=500):
    samples_per_class = int(data_amount / 10)
    data_path = '../database/CIFAR10/cifar-10-batches-py/test_batch'
    with open(data_path, 'rb') as f:
        batch = pickle.load(f, encoding='bytes')
    images = batch[b'data']  # shape: (10000, 3072)
    labels = batch[b'labels']  # list of 10000

    images = images.reshape(-1, 3, 32, 32).astype(np.float32) / 255
    labels = np.array(labels)

    # 按类别分组
    class_indices = {i: np.where(labels == i)[0] for i in range(10)}
    random.seed(random_seed)
    selected_indices = []
    for i in range(10):
        idx = random.sample(list(class_indices[i]), samples_per_class)
        selected_indices.extend(idx)

    x_test = images[selected_indices]
    y_test = np.zeros((samples_per_class * 10, 10), dtype=np.uint8)
    for i, idx in enumerate(selected_indices):
        y_test[i, labels[idx]] = 1

    return x_test, y_test


def sample_cifar100_every_class(random_seed=0, data_amount=500):
    if data_amount % 100 != 0:
        raise ValueError('data_amount 必须能被 100 整除')
    samples_per_class = int(data_amount / 100)
    data_path = '../database/CIFAR100/cifar-100-python/test'
    if os.path.exists(data_path):
        with open(data_path, 'rb') as f:
            batch = pickle.load(f, encoding='bytes')
        images = batch[b'data']  # shape: (10000, 3072)
        labels = batch[b'fine_labels']  # list of 10000
        images = images.reshape(-1, 3, 32, 32).astype(np.float32) / 255
        labels = np.array(labels)
    else:
        cifar_data = datasets.CIFAR100(root='../database/CIFAR100', train=False, download=True)
        images = cifar_data.data.astype(np.float32) / 255
        images = images.transpose(0, 3, 1, 2)
        labels = np.array(cifar_data.targets)

    class_indices = {i: np.where(labels == i)[0] for i in range(100)}
    random.seed(random_seed)
    selected_indices = []
    for i in range(100):
        idx = random.sample(list(class_indices[i]), samples_per_class)
        selected_indices.extend(idx)

    x_test = images[selected_indices]
    y_test = np.zeros((samples_per_class * 100, 100), dtype=np.uint8)
    for i, idx in enumerate(selected_indices):
        y_test[i, labels[idx]] = 1

    return x_test, y_test


def sample_tinyimagenet_every_class(random_seed=0, data_amount=2000):
    if data_amount % 200 != 0:
        raise ValueError('TinyImageNet 的 data_amount 必须能被 200 整除')
    dataset_dir = download_and_prepare_tinyimagenet()
    wnids, samples_by_class = _load_tinyimagenet_val_metadata(dataset_dir)
    num_classes = len(wnids)
    samples_per_class = data_amount // num_classes

    rng = random.Random(random_seed)
    selected_paths = []
    selected_labels = []
    for class_idx in range(num_classes):
        candidates = samples_by_class.get(class_idx, [])
        if len(candidates) < samples_per_class:
            raise ValueError(f'TinyImageNet 类别 {wnids[class_idx]} 可用样本不足 {samples_per_class} 个')
        chosen = rng.sample(candidates, samples_per_class)
        for path in chosen:
            selected_paths.append(path)
            selected_labels.append(class_idx)

    images = []
    labels = []
    for path, label in zip(selected_paths, selected_labels):
        img = default_loader(path)
        img = np.asarray(img, dtype=np.float32) / 255.0
        if img.ndim == 2:
            img = np.stack([img] * 3, axis=-1)
        img = np.transpose(img, (2, 0, 1))
        images.append(img)
        one_hot = np.zeros(num_classes, dtype=np.uint8)
        one_hot[label] = 1
        labels.append(one_hot)

    x_test = np.stack(images, axis=0)
    y_test = np.stack(labels, axis=0)
    return x_test, y_test

