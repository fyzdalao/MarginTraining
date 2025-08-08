import argparse

import torch
import time
import victim
from utils import *
import numpy as np
import torch.nn.functional as F



def load_model(args):
   model = victim.Model(args=args)
   return model

def load_data(args):
    x_test, y_test = sample_cifar10_every_class(random_seed=args.data_seed, data_amount=args.data_amount)
    return x_test, y_test

def try_the_model(model, x_test, y_test):
   logits = model(x_test)
   margin = margin_loss_n(y_test, logits)
   fuck = margin>0
   acc = (fuck).sum() / x_test.shape[0]
   print(acc)


def parse_args():
   parser = argparse.ArgumentParser()

   parser.add_argument('--device', default='cuda:0', type=str)
   parser.add_argument('--arch', type=str, default='resnet34')
   parser.add_argument('--dataset', type=str, default='cifar10')
   parser.add_argument('--data_amount', type=int, default=300)
   parser.add_argument('--data_seed', type=int, default=19260817)
   parser.add_argument('--batch_size', type=int, default=300)
   parser.add_argument('--seed', type=int, default=19260817)

   args = parser.parse_args()
   return args


if __name__ == '__main__':
    args = parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    model = load_model(args)
    x_test, y_test = load_data(args)
    try_the_model(model, x_test, y_test)

    logits_clean = model(x_test)
    margin = margin_loss(y_test, logits_clean)
    correct_idx = margin > 0
    correct_idx = correct_idx.reshape((-1,))








