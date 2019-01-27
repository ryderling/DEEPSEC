#!/usr/bin/env python
# -*- coding: utf-8 -*-
# **************************************
# @Time    : 2018/11/24 15:19
# @Author  : Jiaxu Zou
# @Lab     : nesa.zju.edu.cn
# @File    : EIT_Test.py
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
from RawModels.Utils.dataset import get_mnist_train_validate_loader, get_cifar10_train_validate_loader

from Defenses.DefenseMethods.EIT import EITDefense


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

    dataset = args.dataset.upper()
    assert dataset == 'MNIST' or dataset == 'CIFAR10'
    if dataset == 'MNIST':
        training_parameters = MNIST_Training_Parameters
        model_framework = MNISTConvNet().to(device)
        batch_size = training_parameters['batch_size']
        raw_train_loader, raw_valid_loader = get_mnist_train_validate_loader(dir_name='../RawModels/MNIST/', batch_size=batch_size,
                                                                             valid_size=0.1, shuffle=False)
    else:
        training_parameters = CIFAR10_Training_Parameters
        model_framework = resnet20_cifar().to(device)
        batch_size = training_parameters['batch_size']
        raw_train_loader, raw_valid_loader = get_cifar10_train_validate_loader(dir_name='../RawModels/CIFAR10/', augment=False,
                                                                               batch_size=batch_size, valid_size=0.1, shuffle=False)
        print('cifar 10', len(raw_train_loader.dataset))

    defense_name = 'EIT'
    eit_params = {
        'crop_size': args.crop_size,
        'lambda_tv': args.lambda_tv,
        'JPEG_quality': args.JPEG_quality,
        'bit_depth': args.bit_depth
    }

    EIT = EITDefense(model=model_framework, defense_name=defense_name, dataset=dataset, re_training=True,
                     training_parameters=training_parameters, device=device, **eit_params)
    EIT.defense(train_loader=raw_train_loader, valid_loader=raw_valid_loader)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='EIT Defenses')
    parser.add_argument('--dataset', type=str, default='CIFAR10', help='the dataset (MNIST or CIFAR10)')
    parser.add_argument('--seed', type=int, default=100, help='the default random seed for numpy and torch')
    parser.add_argument('--gpu_index', type=str, help="gpu index to use", default='0')

    # parameters for the EIT Defense
    parser.add_argument('--crop_size', type=int, default=30, help='the cropping size')
    parser.add_argument('--bit_depth', type=int, default=4, help='the quantization level of pixel value')
    parser.add_argument('--JPEG_quality', type=int, default=85, help='the JPEG quality to compress with')
    parser.add_argument('--lambda_tv', type=float, default=0.03, help='the total variance minimization weight')
    arguments = parser.parse_args()
    main(arguments)
