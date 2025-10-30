import argparse

import torch
import time
import victim
from utils import *
import numpy as np
import torch.nn.functional as F

from square import square_attack_linf



def load_model(args):
   model = victim.Model(args=args)
   return model

def load_data(args):
   x_test, y_test = sample_cifar10_every_class(random_seed=args.data_seed, data_amount=args.data_amount)
   return x_test, y_test

def try_the_model(model, x_test, y_test, log=None):
   logits = model(x_test)
   margin = margin_loss_n(y_test, logits)
   fuck = margin>0
   acc = (fuck).sum() / x_test.shape[0]
   #print(acc)
   log.print(str(acc))


def parse_args():
   parser = argparse.ArgumentParser()

   parser.add_argument('--checkpoint', type=str, default='')
   parser.add_argument('--arch', type=str, default='resnet34')

   parser.add_argument('--device', default='cuda:0', type=str)
   parser.add_argument('--dataset', type=str, default='cifar10')
   parser.add_argument('--data_amount', type=int, default=500)
   parser.add_argument('--data_seed', type=int, default=19260817)
   parser.add_argument('--batch_size', type=int, default=300)
   parser.add_argument('--seed', type=int, default=19260817)
   parser.add_argument('--eps', type=float, default=0.031)
   parser.add_argument('--budget', type=int, default=250)

   args = parser.parse_args()
   return args


if __name__ == '__main__':
   args = parse_args()
   log_path = 'attackLog' + '/' + get_time() + '/log.txt'
   log = Logger(log_path)
   log.print(str(args))


   np.random.seed(args.seed)
   torch.manual_seed(args.seed)
   model = load_model(args)
   x_test, y_test = load_data(args)
   try_the_model(model, x_test, y_test, log=log)

   logits_clean = model(x_test)
   margin = margin_loss(y_test, logits_clean)
   correct_idx = margin > 0
   correct_idx = correct_idx.reshape((-1,))

   square_attack_linf(model=model, x=x_test, y=y_test, correct=correct_idx, n_iters=args.budget, eps=args.eps, p_init=0.05, log=log, args=args)








