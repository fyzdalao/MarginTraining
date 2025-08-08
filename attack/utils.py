import os
import random
import numpy as np
import PIL.Image
import time
import pickle
import matplotlib.pyplot as plt



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

