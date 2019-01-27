#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Reference: A. S. Ross et al., "Improving the adversarial robustness and interpretability of deep neural networks by regularizing their input gradients," in AAAI, 2018.
# Reference Implementation from Authors (TensorFlow): https://github.com/dtak/adversarial-robustness-public
# **************************************
# @Time    : 2018/9/17 11:22
# @Author  : Jiaxu Zou & Xiang Ling
# @Lab     : nesa.zju.edu.cn
# @File    : IGR.py
# **************************************

import os

import torch
import torch.nn.functional as F
import torch.optim as optim

from Defenses.DefenseMethods.defenses import Defense
from RawModels.Utils.TrainTest import validation_evaluation


class IGRDefense(Defense):

    def __init__(self, model=None, defense_name=None, dataset=None, lambda_r=None, training_parameters=None, device=None):
        """

        :param model:
        :param defense_name:
        :param dataset:
        :param lambda_r:
        :param training_parameters:
        :param device:
        """
        super(IGRDefense, self).__init__(model=model, defense_name=defense_name)
        self.model = model
        self.defense_name = defense_name
        self.device = device

        self.Dataset = dataset.upper()
        assert self.Dataset in ['MNIST', 'CIFAR10'], "The data set must be MNIST or CIFAR10"

        self.lam_r = lambda_r
        self.num_epochs = training_parameters['num_epochs']

        # keeping use the Adam optimizer for both datasets within the original paper
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.0002, eps=1e-4)

    def train_one_epoch_with_lambda_regularization(self, train_loader, epoch):
        """
        train the model using input gradient regularization
        ensure that if any input changes slightly, the KL divergence between predictions and the labels will not change significantly
        :param train_loader:
        :param epoch:
        :return: None
        """
        # set the model in the training mode
        self.model.train()
        for index, (images, labels) in enumerate(train_loader):
            images.requires_grad = True
            images = images.to(self.device)
            labels = labels.to(self.device)

            logits = self.model(images)
            # calculate the loss1
            l1 = F.cross_entropy(logits, labels)
            # calculate the loss2
            grads = torch.autograd.grad(l1, images, create_graph=True)[0]
            l2 = torch.Tensor.norm(grads, p=2) ** 2
            # add the two losses
            loss = l1 + self.lam_r * l2

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            print('\rTrain Epoch{:>3}: [batch:{:>4}/{:>4}]  \tloss1:{:.4f} + lambda:{} * loss2:{:.4f} = {:.4f} ===> '. \
                  format(epoch, index, len(train_loader), l1, self.lam_r, l2, loss), end=' ')

    def defense(self, train_loader=None, validation_loader=None):

        best_val_acc = None
        for epoch in range(self.num_epochs):
            # training the model using input gradient regularization
            self.train_one_epoch_with_lambda_regularization(train_loader=train_loader, epoch=epoch)
            val_acc = validation_evaluation(model=self.model, validation_loader=validation_loader, device=self.device)

            # save the retained defense-enhanced model
            assert os.path.exists('../DefenseEnhancedModels/{}'.format(self.defense_name))
            defense_enhanced_saver = '../DefenseEnhancedModels/{}/{}_{}_enhanced.pt'.format(self.defense_name, self.Dataset, self.defense_name)
            if not best_val_acc or round(val_acc, 4) >= round(best_val_acc, 4):
                if best_val_acc is not None:
                    os.remove(defense_enhanced_saver)
                best_val_acc = val_acc
                self.model.save(name=defense_enhanced_saver)
            else:
                print('Train Epoch{:>3}: validation dataset accuracy of did not improve from {:.4f}\n'.format(epoch, best_val_acc))
