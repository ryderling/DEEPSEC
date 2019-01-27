#!/usr/bin/env python
# -*- coding: utf-8 -*-
# **************************************
# @Time    : 2018/10/15 15:49
# @Author  : Xiang Ling
# @Lab     : nesa.zju.edu.cn
# @File    : Generation.py
# **************************************

import os
import shutil
from abc import ABCMeta

import numpy as np
import torch

from RawModels.MNISTConv import MNISTConvNet
from RawModels.ResNet import resnet20_cifar


class Generation(object):
    __metaclass__ = ABCMeta

    def __init__(self, dataset='MNIST', attack_name='FGSM', targeted=False, raw_model_location='../RawModels/',
                 clean_data_location='../CleanDatasets/', adv_examples_dir='../AdversarialExampleDatasets/',
                 device=torch.device('cpu')):
        # check and set the support data set
        self.dataset = dataset.upper()
        if self.dataset not in {'MNIST', 'CIFAR10'}:
            raise ValueError("The data set must be MNIST or CIFAR10")

        # check and set the supported attack
        self.attack_name = attack_name.upper()
        supported = {'FGSM', 'RFGSM', 'BIM', 'PGD', 'UMIFGSM', 'UAP', 'DEEPFOOL', 'OM', 'LLC', "RLLC", 'ILLC', 'TMIFGSM', 'JSMA', 'BLB', 'CW2',
                     'EAD'}
        if self.attack_name not in supported:
            raise ValueError(self.attack_name + 'is unknown!\nCurrently, our implementation support the attacks: ' + ', '.join(supported))

        # load the raw model
        raw_model_location = '{}{}/model/{}_raw.pt'.format(raw_model_location, self.dataset, self.dataset)
        if self.dataset == 'MNIST':
            self.raw_model = MNISTConvNet().to(device)
            self.raw_model.load(path=raw_model_location, device=device)
        else:
            self.raw_model = resnet20_cifar().to(device)
            self.raw_model.load(path=raw_model_location, device=device)

        # get the clean data sets / true_labels / targets (if the attack is one of the targeted attacks)
        print('Loading the prepared clean samples (nature inputs and corresponding labels) that will be attacked ...... ')
        self.nature_samples = np.load('{}{}/{}_inputs.npy'.format(clean_data_location, self.dataset, self.dataset))
        self.labels_samples = np.load('{}{}/{}_labels.npy'.format(clean_data_location, self.dataset, self.dataset))

        if targeted:
            print('For Targeted Attacks, loading the randomly selected targeted labels that will be attacked ......')
            if self.attack_name.upper() in ['LLC', 'RLLC', 'ILLC']:
                print('#### Especially, for LLC, RLLC, ILLC, loading the least likely class that will be attacked')
                self.targets_samples = np.load('{}{}/{}_llc.npy'.format(clean_data_location, self.dataset, self.dataset))
            else:
                self.targets_samples = np.load('{}{}/{}_targets.npy'.format(clean_data_location, self.dataset, self.dataset))

        # prepare the directory for the attacker to save their generated adversarial examples
        self.adv_examples_dir = adv_examples_dir + self.attack_name + '/' + self.dataset + '/'
        if self.attack_name not in os.listdir(adv_examples_dir):
            os.mkdir(adv_examples_dir + self.attack_name + '/')

        if self.dataset not in os.listdir(adv_examples_dir + self.attack_name + '/'):
            os.mkdir(self.adv_examples_dir)
        else:
            shutil.rmtree('{}'.format(self.adv_examples_dir))
            os.mkdir(self.adv_examples_dir)

        # set up device
        self.device = device

    def generate(self):
        print("abstract method of Generation is not implemented")
        raise NotImplementedError
