#!/usr/bin/env python
# -*- coding: utf-8 -*-
# **************************************
# @Time    : 2018/10/15 14:22
# @Author  : Xiang Ling
# @Lab     : nesa.zju.edu.cn
# @File    : train_cifar10.py
# **************************************

import argparse
import copy
import os
import random
import sys

import numpy as np
import torch
import torch.optim as optim

sys.path.append('%s/../' % os.path.dirname(os.path.realpath(__file__)))
from RawModels.ResNet import CIFAR10_Training_Parameters, adjust_learning_rate, resnet20_cifar
from RawModels.Utils.TrainTest import train_one_epoch, validation_evaluation, testing_evaluation
from RawModels.Utils.dataset import get_cifar10_train_validate_loader, get_cifar10_test_loader


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

    # get the training and testing dataset loaders
    train_loader, valid_loader = get_cifar10_train_validate_loader(dir_name='./CIFAR10/', batch_size=CIFAR10_Training_Parameters['batch_size'],
                                                                   valid_size=0.1, shuffle=True)
    test_loader = get_cifar10_test_loader(dir_name='./CIFAR10/', batch_size=CIFAR10_Training_Parameters['batch_size'])

    # set up the model and optimizer
    resnet_model = resnet20_cifar().to(device)
    optimizer = optim.Adam(resnet_model.parameters(), lr=CIFAR10_Training_Parameters['lr'])

    # Training
    best_val_acc = None
    model_saver = './CIFAR10/model/CIFAR10_' + 'raw' + '.pt'
    for epoch in range(CIFAR10_Training_Parameters['num_epochs']):

        # training the model within one epoch
        train_one_epoch(model=resnet_model, train_loader=train_loader, optimizer=optimizer, epoch=epoch, device=device)
        # validation
        val_acc = validation_evaluation(model=resnet_model, validation_loader=valid_loader, device=device)

        # adjust the learning rate
        adjust_learning_rate(optimizer=optimizer, epoch=epoch)

        if not best_val_acc or round(val_acc, 4) >= round(best_val_acc, 4):
            if best_val_acc is not None:
                os.remove(model_saver)
            best_val_acc = val_acc
            resnet_model.save(name=model_saver)
        else:
            print('Train Epoch{:>3}: validation dataset accuracy did not improve from {:.4f}\n'.format(epoch, best_val_acc))

    # Testing
    final_model = copy.deepcopy(resnet_model)
    final_model.load(path=model_saver, device=device)
    accuracy = testing_evaluation(model=final_model, test_loader=test_loader, device=device)
    print('Finally, the ACCURACY of saved model [{}] on testing dataset is {:.2f}%\n'.format(final_model.model_name, accuracy * 100.0))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Training and Testing the model for CIFAR10')
    parser.add_argument('--gpu_index', type=str, default='0', help="gpu index to use")
    parser.add_argument('--seed', type=int, default=100, help='the default random seed for numpy and torch')
    arguments = parser.parse_args()
    main(arguments)
