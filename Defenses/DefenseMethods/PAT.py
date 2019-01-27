#!/usr/bin/env python
# -*- coding: utf-8 -*-
# References:   A. Madry, et al., "Towards deep learning models resistant to adversarial attacks," in ICLR, 2018.
# Reference Implementation from Authors (TensorFlow):
#   [1] https://github.com/MadryLab/mnist_challenge
#   [2] https://github.com/MadryLab/cifar10_challenge
# **************************************
# @Time    : 2018/9/17 1:20
# @Author  : Jiaxu Zou & Xiang Ling
# @Lab     : nesa.zju.edu.cn
# @File    : PAT.py
# **************************************

import os

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

from Attacks.AttackMethods.AttackUtils import tensor2variable
from Defenses.DefenseMethods.defenses import Defense
from RawModels.ResNet import adjust_learning_rate
from RawModels.Utils.TrainTest import validation_evaluation


class PATDefense(Defense):

    def __init__(self, model=None, defense_name=None, dataset=None, training_parameters=None, device=None, **kwargs):
        """

        :param model:
        :param defense_name:
        :param dataset:
        :param training_parameters:
        :param device:
        :param kwargs:
        """
        super(PATDefense, self).__init__(model=model, defense_name=defense_name)
        self.model = model
        self.defense_name = defense_name
        self.device = device

        self.Dataset = dataset.upper()
        assert self.Dataset in ['MNIST', 'CIFAR10'], "The data set must be MNIST or CIFAR10"

        # make sure to parse the parameters for the defense
        assert self._parsing_parameters(**kwargs)

        # get the training_parameters, the same as the settings of RawModels
        self.num_epochs = training_parameters['num_epochs']
        self.batch_size = training_parameters['batch_size']

        if self.Dataset == "MNIST":
            self.optimizer = optim.SGD(self.model.parameters(), lr=training_parameters['learning_rate'],
                                       momentum=training_parameters['momentum'], weight_decay=training_parameters['decay'], nesterov=True)
        else:
            self.optimizer = optim.Adam(self.model.parameters(), lr=training_parameters['lr'])

    def _parsing_parameters(self, **kwargs):

        assert kwargs is not None, "the parameters should be specified"

        print("\nparsing the user configuration for the {} defense".format(self.defense_name))
        for key in kwargs:
            print('\t{} = {}'.format(key, kwargs.get(key)))

        self.attack_step_num = kwargs.get('attack_step_num')
        self.step_size = kwargs.get('step_size')
        self.epsilon = kwargs.get('epsilon')

        return True

    def pgd_generation(self, var_natural_images=None, var_natural_labels=None):
        """

        :param var_natural_images:
        :param var_natural_labels:
        :return:
        """
        self.model.eval()
        natural_images = var_natural_images.cpu().numpy()

        copy_images = natural_images.copy()
        copy_images = copy_images + np.random.uniform(-self.epsilon, self.epsilon, copy_images.shape).astype('float32')

        for i in range(self.attack_step_num):
            var_copy_images = torch.from_numpy(copy_images).to(self.device)
            var_copy_images.requires_grad = True

            preds = self.model(var_copy_images)
            loss = F.cross_entropy(preds, var_natural_labels)
            gradient = torch.autograd.grad(loss, var_copy_images)[0]
            gradient_sign = torch.sign(gradient).cpu().numpy()

            copy_images = copy_images + self.step_size * gradient_sign

            copy_images = np.clip(copy_images, natural_images - self.epsilon, natural_images + self.epsilon)
            copy_images = np.clip(copy_images, 0.0, 1.0)

        return torch.from_numpy(copy_images).to(self.device)

    def train_one_epoch_with_pgd_and_nat(self, train_loader, epoch):
        """

        :param train_loader:
        :param epoch:
        :return:
        """
        for index, (images, labels) in enumerate(train_loader):
            nat_images = images.to(self.device)
            nat_labels = labels.to(self.device)

            # prepare for adversarial examples using the pgd attack
            self.model.eval()
            adv_images = self.pgd_generation(var_natural_images=nat_images, var_natural_labels=nat_labels)

            # set the model in the training mode
            self.model.train()

            logits_nat = self.model(nat_images)
            loss_nat = F.cross_entropy(logits_nat, nat_labels)
            logits_adv = self.model(adv_images)
            loss_adv = F.cross_entropy(logits_adv, nat_labels)
            loss = 0.5 * (loss_nat + loss_adv)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            print('\rTrain Epoch {:>3}: [{:>5}/{:>5}]  \tloss_nat={:.4f}, loss_adv={:.4f}, total_loss={:.4f} ===> '. \
                  format(epoch, (index + 1) * len(images), len(train_loader) * len(images), loss_nat, loss_adv, loss), end=' ')

    def defense(self, train_loader=None, validation_loader=None):

        best_val_acc = None
        for epoch in range(self.num_epochs):

            # training the model with natural examples and corresponding adversarial examples
            self.train_one_epoch_with_pgd_and_nat(train_loader=train_loader, epoch=epoch)
            val_acc = validation_evaluation(model=self.model, validation_loader=validation_loader, device=self.device)

            # adjust the learning rate for cifar10
            if self.Dataset == 'CIFAR10':
                adjust_learning_rate(epoch=epoch, optimizer=self.optimizer)

            # save the retained defense-enhanced model
            assert os.path.exists('../DefenseEnhancedModels/{}'.format(self.defense_name))
            defense_enhanced_saver = '../DefenseEnhancedModels/{}/{}_{}_enhanced.pt'.format(self.defense_name, self.Dataset, self.defense_name)
            if not best_val_acc or round(val_acc, 4) >= round(best_val_acc, 4):
                if best_val_acc is not None:
                    os.remove(defense_enhanced_saver)
                best_val_acc = val_acc
                self.model.save(name=defense_enhanced_saver)
            else:
                print('Train Epoch{:>3}: validation dataset accuracy did not improve from {:.4f}\n'.format(epoch, best_val_acc))
