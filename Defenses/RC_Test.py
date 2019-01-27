#!/usr/bin/env python
# -*- coding: utf-8 -*-
# **************************************
# @Time    : 2018/11/21 21:24
# @Author  : Xiang Ling
# @Lab     : nesa.zju.edu.cn
# @File    : RC_Test.py 
# **************************************
import argparse
import os
import random
import sys

import numpy as np
import torch

sys.path.append('%s/../' % os.path.dirname(os.path.realpath(__file__)))
from RawModels.MNISTConv import MNISTConvNet
from RawModels.ResNet import resnet20_cifar
from RawModels.Utils.dataset import get_mnist_train_validate_loader, get_mnist_test_loader
from RawModels.Utils.dataset import get_cifar10_train_validate_loader, get_cifar10_test_loader

from Defenses.DefenseMethods.RC import RCDefense


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
    batch_size = 1000
    # get the training_parameters, raw_model, and test_loader
    model_location = '{}/{}/model/{}_raw.pt'.format('../RawModels', dataset, dataset)
    if dataset == 'MNIST':
        raw_model = MNISTConvNet().to(device)
        test_loader = get_mnist_test_loader(dir_name='../RawModels/MNIST/', batch_size=batch_size)
    else:
        raw_model = resnet20_cifar().to(device)
        test_loader = get_cifar10_test_loader(dir_name='../RawModels/CIFAR10/', batch_size=batch_size)

    raw_model.load(path=model_location, device=device)

    defense_name = 'RC'
    rc = RCDefense(model=raw_model, defense_name=defense_name, dataset=dataset, device=device, num_points=args.num_points)

    # search the radius r
    if args.search:
        # get the validation dataset (10% with the training dataset)
        print('start to search the radius r using validation dataset ...')
        if dataset == 'MNIST':
            _, valid_loader = get_mnist_train_validate_loader(dir_name='../RawModels/MNIST/', batch_size=batch_size, valid_size=0.02,
                                                              shuffle=True)
        else:
            _, valid_loader = get_cifar10_train_validate_loader(dir_name='../RawModels/CIFAR10/', batch_size=batch_size, valid_size=0.02,
                                                                shuffle=True)
        radius = rc.search_best_radius(validation_loader=valid_loader, radius_min=args.radius_min, radius_max=args.radius_max,
                                       radius_step=args.radius_step)
    else:
        radius = round(args.radius, 2)
    print('######\nthe radius for RC is set or searched as: {}\n######'.format(radius))

    # calculate the accuracy of region-based classification defense on testing dataset
    print('\nStart to calculate the accuracy of region-based classification defense on testing dataset')
    raw_model.eval()
    total = 0.0
    correct = 0.0
    with torch.no_grad():
        for images, labels in test_loader:
            # apply the region-based classification on images using radius
            rc_labels = rc.region_based_classification(samples=images, radius=radius)
            rc_labels = torch.from_numpy(rc_labels)

            total += labels.size(0)
            correct += (rc_labels == labels).sum().item()
        ratio = correct / total
        print('\nTest accuracy of the {} model on the testing dataset: {:.1f}/{:.1f} = {:.2f}%\n'.format(raw_model.model_name, correct, total,
                                                                                                         ratio * 100))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='The RC Defenses')
    parser.add_argument('--dataset', type=str, default='CIFAR10', help='the dataset (MNIST or CIFAR10)')
    parser.add_argument('--seed', type=int, default=100, help='the default random seed for numpy and torch')
    parser.add_argument('--gpu_index', type=str, default='0', help="gpu index to use")

    # parameters for the RC Defense
    parser.add_argument('--search', default=True, type=lambda x: (str(x).lower() == 'true'), help='indicate whether search the radius r')
    parser.add_argument('--radius', type=float, default=0.02, help='in the case of not search radius r, we set the radius of the hypercube')
    parser.add_argument('--num_points', type=int, default=1000, help='number of points chosen in the adjacent region for each image')

    # parameters used for finding the best radius for the defense
    parser.add_argument('--radius_min', type=float, default=0.0, help='lower bound of radius')
    parser.add_argument('--radius_max', type=float, default=0.1, help='upper bound of radius')
    parser.add_argument('--radius_step', type=float, default=0.01, help='step size of radius in searching')
    arguments = parser.parse_args()
    main(arguments)
