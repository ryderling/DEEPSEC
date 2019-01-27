#!/usr/bin/env python
# -*- coding: utf-8 -*-
# **************************************
# @Time    : 2018/9/17 15:19
# @Author  : Jiaxu Zou
# @Lab     : nesa.zju.edu.cn
# @File    : NAT_Test.py
# **************************************

import argparse
import os
import random
import sys

import numpy as np
import torch

sys.path.append('%s/../' % os.path.dirname(os.path.realpath(__file__)))

from RawModels.MNISTConv import MNISTConvNet, MNIST_Training_Parameters
from RawModels.ResNet import resnet20_cifar, CIFAR10_Training_Parameters
from RawModels.Utils.dataset import get_mnist_train_validate_loader
from RawModels.Utils.dataset import get_cifar10_train_validate_loader

from Defenses.DefenseMethods.NAT import NATDefense


def main(args):
    # Device configuration
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_index
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Set the random seed manually for reproducibility.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    # Get training parameters, set up model frameworks and then get the train_loader and test_loader
    dataset = args.dataset.upper()
    assert dataset == 'MNIST' or dataset == 'CIFAR10'
    if dataset == 'MNIST':
        training_parameters = MNIST_Training_Parameters
        model_framework = MNISTConvNet().to(device)
        batch_size = training_parameters['batch_size']
        train_loader, valid_loader = get_mnist_train_validate_loader(dir_name='../RawModels/MNIST/', batch_size=batch_size, valid_size=0.1,
                                                                     shuffle=True)
    else:
        training_parameters = CIFAR10_Training_Parameters
        model_framework = resnet20_cifar().to(device)
        batch_size = training_parameters['batch_size']
        train_loader, valid_loader = get_cifar10_train_validate_loader(dir_name='../RawModels/CIFAR10/', batch_size=batch_size, valid_size=0.1,
                                                                       shuffle=True)

    defense_name = 'NAT'
    nat_params = {
        'adv_ratio': args.adv_ratio,
        'eps_min': args.clip_min,
        'eps_max': args.clip_max,
        'mu': args.eps_mu,
        'sigma': args.eps_sigma
    }

    nat = NATDefense(model=model_framework, defense_name=defense_name, dataset=dataset, training_parameters=training_parameters, device=device,
                     **nat_params)
    nat.defense(train_loader=train_loader, validation_loader=valid_loader)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='The NAT Defenses')
    parser.add_argument('--dataset', type=str, default='MNIST', help='the dataset (MNIST or CIFAR10)')
    parser.add_argument('--gpu_index', type=str, default='0', help="gpu index to use")
    parser.add_argument('--seed', type=int, default=100, help='the default random seed for numpy and torch')

    # parameters for the NAT Defense
    parser.add_argument('--adv_ratio', type=float, default=0.3, help='the weight of adversarial example when adversarial training')
    parser.add_argument('--clip_min', type=float, default=0.0, help='the min of epsilon allowed')
    parser.add_argument('--clip_max', type=float, default=0.3, help='the max of epsilon allowed')
    parser.add_argument('--eps_mu', type=int, default=0, help='the \mu value of normal distribution for epsilon')
    parser.add_argument('--eps_sigma', type=int, default=50, help='the \sigma value of normal distribution for epsilon')
    arguments = parser.parse_args()
    main(arguments)
