#!/usr/bin/env python
# -*- coding: utf-8 -*-
# References: A. Kurakin, I. Goodfellow, and S. Bengio, "Adversarial machine learning at scale," in ICLR, 2017.
# **************************************
# @Time    : 2018/9/17 13:19
# @Author  : Jiaxu Zou & Xiang Ling
# @Lab     : nesa.zju.edu.cn
# @File    : NAT.py
# **************************************


import os

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

from Defenses.DefenseMethods.defenses import Defense
from RawModels.ResNet import adjust_learning_rate
from RawModels.Utils.TrainTest import validation_evaluation


class NATDefense(Defense):

    def __init__(self, model=None, defense_name=None, dataset=None, training_parameters=None, device=None, **kwargs):
        """

        :param model:
        :param defense_name:
        :param dataset:
        :param training_parameters:
        :param device:
        :param kwargs:
        """
        super(NATDefense, self).__init__(model=model, defense_name=defense_name)
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

        # prepare the optimizers
        if self.Dataset == 'MNIST':
            self.optimizer = optim.SGD(self.model.parameters(), lr=training_parameters['learning_rate'],
                                       momentum=training_parameters['momentum'], weight_decay=training_parameters['decay'], nesterov=True)
        else:
            self.optimizer = optim.Adam(self.model.parameters(), lr=training_parameters['lr'])

    def _parsing_parameters(self, **kwargs):

        assert kwargs is not None, "the parameters should be specified"

        print("\nUser configurations for the {} defense".format(self.defense_name))
        for key in kwargs:
            print('\t{} = {}'.format(key, kwargs[key]))

        self.adv_ratio = kwargs['adv_ratio']
        self.clip_eps_min = kwargs['eps_min']
        self.clip_eps_max = kwargs['eps_max']
        self.eps_mu = kwargs['mu']
        self.eps_sigma = kwargs['sigma']

        return True

    def random_llc_generation(self, var_natural_images=None):
        """

        :param var_natural_images:
        :return:
        """
        self.model.eval()
        clone_var_natural_images = var_natural_images.clone()

        # get the random epsilon for the Random LLC generation
        random_eps = np.random.normal(loc=self.eps_mu, scale=self.eps_sigma, size=[var_natural_images.size(0)]) / 255.0
        random_eps = np.clip(np.abs(random_eps), self.clip_eps_min, self.clip_eps_max)

        clone_var_natural_images.requires_grad = True

        # prepare the least likely class labels (avoid label leaking effect)
        logits = self.model(clone_var_natural_images)
        llc_labels = torch.min(logits, dim=1)[1]
        # get the loss and gradients
        loss_llc = F.cross_entropy(logits, llc_labels)
        gradients_llc = torch.autograd.grad(loss_llc, clone_var_natural_images)[0]

        clone_var_natural_images.requires_grad = False

        gradients_sign = torch.sign(gradients_llc)
        var_random_eps = torch.from_numpy(random_eps).float().to(self.device)

        # generation of adversarial examples
        with torch.no_grad():
            list_var_adv_images = []
            for i in range(var_natural_images.size(0)):
                var_adv_image = var_natural_images[i] - var_random_eps[i] * gradients_sign[i]
                var_adv_image = torch.clamp(var_adv_image, min=0.0, max=1.0)
                list_var_adv_images.append(var_adv_image)
            ret_adv_images = torch.stack(list_var_adv_images)
        ret_adv_images = torch.clamp(ret_adv_images, min=0.0, max=1.0)

        return ret_adv_images

    def train_one_epoch_with_adv_and_nat(self, train_loader, epoch):
        """

        :param train_loader:
        :param epoch:
        :return:
        """

        for index, (images, labels) in enumerate(train_loader):
            nat_images = images.to(self.device)
            nat_labels = labels.to(self.device)

            # set the model in the eval mode and generate the adversarial examples using the LLC (Least Likely Class) attack
            self.model.eval()
            adv_images = self.random_llc_generation(var_natural_images=nat_images)

            # set the model in the train mode
            self.model.train()

            logits_nat = self.model(nat_images)
            loss_nat = F.cross_entropy(logits_nat, nat_labels)  # loss on natural images

            logits_adv = self.model(adv_images)
            loss_adv = F.cross_entropy(logits_adv, nat_labels)  # loss on the generated adversarial images

            # add two parts of loss
            loss = (loss_nat + self.adv_ratio * loss_adv) / (1.0 + self.adv_ratio)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            print('\rTrain Epoch {:>3}: [batch:{:>4}/{:>4}]  \tloss_nat={:.4f}, loss_adv={:.4f}, total_loss={:.4f} ===> '. \
                  format(epoch, index, len(train_loader), loss_nat, loss_adv, loss), end=' ')

    def defense(self, train_loader=None, validation_loader=None):
        """

        :param train_loader:
        :param validation_loader:
        :return:
        """
        best_val_acc = None
        for epoch in range(self.num_epochs):
            # training the model with natural examples and corresponding adversarial examples
            self.train_one_epoch_with_adv_and_nat(train_loader=train_loader, epoch=epoch)
            val_acc = validation_evaluation(model=self.model, validation_loader=validation_loader, device=self.device)

            # adjust the learning rate for cifar10
            if self.Dataset == 'CIFAR10':
                adjust_learning_rate(epoch=epoch, optimizer=self.optimizer)

            # save the retrained defense-enhanced model
            assert os.path.exists('../DefenseEnhancedModels/{}'.format(self.defense_name))
            defense_enhanced_saver = '../DefenseEnhancedModels/{}/{}_{}_enhanced.pt'.format(self.defense_name, self.Dataset, self.defense_name)
            if not best_val_acc or round(val_acc, 4) >= round(best_val_acc, 4):
                if best_val_acc is not None:
                    os.remove(defense_enhanced_saver)
                best_val_acc = val_acc
                self.model.save(name=defense_enhanced_saver)
            else:
                print('Train Epoch{:>3}: validation dataset accuracy did not improve from {:.4f}\n'.format(epoch, best_val_acc))
