#!/usr/bin/env python
# -*- coding: utf-8 -*-
# **************************************
# @Time    : 2018/9/7 23:04
# @Author  : Xiang Ling
# @Lab     : nesa.zju.edu.cn
# @File    : train_mnist.py
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

from RawModels.MNISTConv import MNISTConvNet, MNIST_Training_Parameters
from RawModels.Utils.TrainTest import train_one_epoch, testing_evaluation, validation_evaluation
from RawModels.Utils.dataset import get_mnist_train_validate_loader, get_mnist_test_loader


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
    train_loader, valid_loader = get_mnist_train_validate_loader(dir_name='./MNIST/', batch_size=MNIST_Training_Parameters['batch_size'],
                                                                 valid_size=0.1, shuffle=True)
    test_loader = get_mnist_test_loader(dir_name='./MNIST/', batch_size=MNIST_Training_Parameters['batch_size'])

    # set up the model and optimizer
    mnist_model = MNISTConvNet().to(device)
    optimizer = optim.SGD(mnist_model.parameters(), lr=MNIST_Training_Parameters['learning_rate'],
                          momentum=MNIST_Training_Parameters['momentum'], weight_decay=MNIST_Training_Parameters['decay'], nesterov=True)

    # Training
    best_val_acc = None
    model_saver = './MNIST/model/MNIST_' + 'raw' + '.pt'
    for epoch in range(MNIST_Training_Parameters['num_epochs']):

        # training the model within one epoch
        train_one_epoch(model=mnist_model, train_loader=train_loader, optimizer=optimizer, epoch=epoch, device=device)
        # validation
        val_acc = validation_evaluation(model=mnist_model, validation_loader=valid_loader, device=device)

        if not best_val_acc or round(val_acc, 4) >= round(best_val_acc, 4):
            if best_val_acc is not None:
                os.remove(model_saver)
            best_val_acc = val_acc
            mnist_model.save(name=model_saver)
        else:
            print('Train Epoch{:>3}: validation dataset accuracy did not improve from {:.4f}\n'.format(epoch, best_val_acc))

    # Testing
    final_model = copy.deepcopy(mnist_model)
    final_model.load(path=model_saver, device=device)
    accuracy = testing_evaluation(model=final_model, test_loader=test_loader, device=device)
    print('Finally, the ACCURACY of saved model [{}] on testing dataset is {:.2f}%\n'.format(final_model.model_name, accuracy * 100.0))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Training and Testing the model for MNIST')
    parser.add_argument('--gpu_index', type=str, default='0', help="gpu index to use")
    parser.add_argument('--seed', type=int, default=100, help='the default random seed for numpy and torch')
    arguments = parser.parse_args()
    main(arguments)
