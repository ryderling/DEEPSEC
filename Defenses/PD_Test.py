#!/usr/bin/env python
# -*- coding: utf-8 -*-
# **************************************
# @Time    : 2018/11/23 21:41
# @Author  : Saizhuo Wang & Xiang Ling
# @Lab     : nesa.zju.edu.cn
# @File    : PD_Test.py 
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
from RawModels.Utils.dataset import get_mnist_test_loader, get_cifar10_test_loader
from Defenses.DefenseMethods.PD import PixelDefend


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

    batch_size = args.batch_size
    model_location = '{}/{}/model/{}_raw.pt'.format('../RawModels', args.dataset, args.dataset)
    # Get training parameters, set up model frameworks and then get the train_loader and test_loader
    dataset = args.dataset.upper()
    assert dataset == 'MNIST' or dataset == 'CIFAR10'
    if dataset == 'MNIST':
        raw_model = MNISTConvNet().to(device)
        test_loader = get_mnist_test_loader(dir_name='../RawModels/MNIST/', batch_size=batch_size)
    else:
        raw_model = resnet20_cifar().to(device)
        test_loader = get_cifar10_test_loader(dir_name='../RawModels/CIFAR10/', batch_size=batch_size)
    raw_model.load(path=model_location, device=device)

    defense_name = 'PD'

    pd = PixelDefend(model=raw_model, defense_name=defense_name, dataset=dataset, pixel_cnn_dir='../Defenses/', device=device)
    eps = args.epsilon

    # predicting the testing dataset using the randomization transformation defense
    raw_model.eval()
    total = 0.0
    correct = 0.0
    with torch.no_grad():
        for index, (images, labels) in enumerate(test_loader):
            print('index = {}'.format(index))
            images = images.cpu().numpy()
            # input images should be first purified and then the resulting images are fed into the original model
            de_noised_images = pd.de_noising_samples(samples=images, batch_size=images.shape[0], eps=eps)
            outputs = raw_model(torch.from_numpy(de_noised_images).to(device))

            labels = labels.to(device)
            _, predicted = torch.max(outputs.data, 1)
            total = total + labels.size(0)
            correct = correct + (predicted == labels).sum().item()
        ratio = correct / total
        print('\nTest accuracy of the {} model on the testing dataset: {:.1f}/{:.1f} = {:.2f}%\n'.format(raw_model.model_name, correct, total,
                                                                                                         ratio * 100))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='The PixelDefend Defense')
    parser.add_argument('--dataset', type=str, default='CIFAR10', help='the dataset (MNIST or CIFAR10)')
    parser.add_argument('--seed', type=int, default=100, help='the default random seed for numpy and torch')
    parser.add_argument('--gpu_index', type=str, default='0,1', help="gpu index to use")

    # parameters for the PixelDefend defense
    parser.add_argument('--epsilon', type=float, default=0.0627, help="radius of e_ball")
    parser.add_argument('--batch_size', type=int, default=20, help="batch size")
    arguments = parser.parse_args()
    main(arguments)
