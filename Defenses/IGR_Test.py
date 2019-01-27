#!/usr/bin/env python
# -*- coding: utf-8 -*-
# **************************************
# @Time    : 2018/9/16 14:26
# @Author  : Xiang Ling
# @Lab     : nesa.zju.edu.cn
# @File    : IGR_Test.py
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
from Defenses.DefenseMethods.IGR import IGRDefense


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

    batch_size = 256  # following the settings of the original paper
    # Get training parameters, set up model frameworks and then get the train_loader and test_loader
    dataset = args.dataset.upper()
    assert dataset == 'MNIST' or dataset == 'CIFAR10'
    if args.dataset == 'MNIST':
        training_parameters = MNIST_Training_Parameters
        model_framework = MNISTConvNet().to(device)
        train_loader, valid_loader = get_mnist_train_validate_loader(dir_name='../RawModels/MNIST/', batch_size=batch_size, valid_size=0.1,
                                                                     shuffle=True)
    else:
        training_parameters = CIFAR10_Training_Parameters
        model_framework = resnet20_cifar().to(device)
        train_loader, valid_loader = get_cifar10_train_validate_loader(dir_name='../RawModels/CIFAR10/', batch_size=batch_size, valid_size=0.1,
                                                                       shuffle=True)

    defense_name = 'IGR'
    igr = IGRDefense(model=model_framework, defense_name=defense_name, dataset=dataset, training_parameters=training_parameters,
                     lambda_r=args.lambda_r, device=device)
    igr.defense(train_loader=train_loader, validation_loader=valid_loader)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='The IGR Defenses')
    parser.add_argument('--dataset', type=str, default='MNIST', help='the dataset (MNIST or CIFAR10)')
    parser.add_argument('--gpu_index', type=str, default='0', help="gpu index to use")
    parser.add_argument('--seed', type=int, default=100, help='the default random seed for numpy and torch')

    # parameters for the IGR Defense
    parser.add_argument('--lambda_r', type=int, default=10, help='lambda of regularization term')
    arguments = parser.parse_args()
    main(arguments)
