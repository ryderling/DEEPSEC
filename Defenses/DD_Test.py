#!/usr/bin/env python
# -*- coding: utf-8 -*-
# **************************************
# @Time    : 2018/9/17 1:21
# @Author  : Jiaxu Zou
# @Lab     : nesa.zju.edu.cn
# @File    : DD_Test.py
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
from RawModels.Utils.dataset import get_mnist_train_validate_loader, get_mnist_test_loader
from RawModels.Utils.dataset import get_cifar10_train_validate_loader, get_cifar10_test_loader

from Defenses.DefenseMethods.DD import DistillationDefense


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

    # Get training parameters, set up model frames and then get the train_loader and test_loader
    dataset = args.dataset.upper()
    assert dataset == 'MNIST' or dataset == 'CIFAR10'
    if dataset == 'MNIST':
        training_parameters = MNIST_Training_Parameters
        model_framework = MNISTConvNet().to(device)
        batch_size = training_parameters['batch_size']
        # train_loader for training the initial model
        train_loader, valid_loader = get_mnist_train_validate_loader(dir_name='../RawModels/MNIST/', batch_size=batch_size, valid_size=0.1,
                                                                     shuffle=True)
        # raw train_loader (no augmentation) for constructing the SoftLabelDataset and then used to train the distilled model
        raw_train_loader, raw_valid_loader = get_mnist_train_validate_loader(dir_name='../RawModels/MNIST/', batch_size=batch_size,
                                                                             valid_size=0.1, shuffle=False)
        # testing dataset loader
        test_loader = get_mnist_test_loader(dir_name='../RawModels/MNIST/', batch_size=batch_size)
    else:
        training_parameters = CIFAR10_Training_Parameters
        model_framework = resnet20_cifar().to(device)
        batch_size = training_parameters['batch_size']
        # train_loader for training the initial model
        train_loader, valid_loader = get_cifar10_train_validate_loader(dir_name='../RawModels/CIFAR10/', batch_size=batch_size, valid_size=0.1,
                                                                       augment=True, shuffle=True)
        # raw train_loader (no augmentation) for constructing the SoftLabelDataset and then used to train the distilled model
        raw_train_loader, raw_valid_loader = get_cifar10_train_validate_loader(dir_name='../RawModels/CIFAR10/', batch_size=batch_size,
                                                                               valid_size=0.1, augment=False, shuffle=False)
        # testing dataset loader
        test_loader = get_cifar10_test_loader(dir_name='../RawModels/CIFAR10/', batch_size=batch_size)

    defense_name = 'DD'
    dd = DistillationDefense(model=model_framework, defense_name=defense_name, dataset=dataset, temperature=args.temp,
                             training_parameters=training_parameters, device=device)
    dd.defense(initial_flag=args.initial, train_loader=train_loader, validation_loader=valid_loader, raw_train=raw_train_loader,
               raw_valid=raw_valid_loader, test_loader=test_loader)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='The DD Defenses')
    parser.add_argument('--dataset', type=str, default='CIFAR10', help='the dataset (MNIST or CIFAR10)')
    parser.add_argument('--gpu_index', type=str, default='0', help="gpu index to use")
    parser.add_argument('--seed', type=int, default=100, help='the default random seed for numpy and torch')

    # parameters for the DD Defense
    parser.add_argument('--initial', type=lambda x: (str(x).lower() == 'true'), default='False',
                        help='True if there exists a pre-trained initial model')
    parser.add_argument('--temp', type=float, default=50.0, help='distillation temperature')
    arguments = parser.parse_args()
    main(arguments)
