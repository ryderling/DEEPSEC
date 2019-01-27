#!/usr/bin/env python
# -*- coding: utf-8 -*-
# References:   F. Tram`er, et al.,"Ensemble adversarial training: Attacks and defenses," in ICLR, 2018.
# Reference Implementation from Authors (TensorFlow):   https://github.com/ftramer/ensemble-adv-training
# **************************************
# @Time    : 2018/9/18 19:16
# @Author  : Jiaxu Zou & Xiang Ling
# @Lab     : nesa.zju.edu.cn
# @File    : EAT.py
# **************************************


import os

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

# import external model architectures
from Defenses.DefenseMethods.EAT_External_Models import CIFAR10_A, CIFAR10_B, CIFAR10_C, CIFAR10_D
from Defenses.DefenseMethods.EAT_External_Models import MNIST_A, MNIST_B, MNIST_C, MNIST_D
from Defenses.DefenseMethods.defenses import Defense
from RawModels.ResNet import adjust_learning_rate
from RawModels.Utils.TrainTest import testing_evaluation, train_one_epoch, validation_evaluation


class EATDefense(Defense):

    def __init__(self, model=None, defense_name=None, dataset=None, training_parameters=None, device=None, **kwargs):
        """

        :param model:
        :param defense_name:
        :param dataset:
        :param training_parameters:
        :param device:
        :param kwargs:
        """
        super(EATDefense, self).__init__(model=model, defense_name=defense_name)
        self.model = model
        self.defense_name = defense_name
        self.device = device
        self.training_parameters = training_parameters

        self.Dataset = dataset.upper()
        assert self.Dataset in ['MNIST', 'CIFAR10'], "The data set must be MNIST or CIFAR10"

        # make sure to parse the parameters for the defense
        assert self._parsing_parameters(**kwargs)

        # get the training_parameters, the same as the settings of RawModels
        self.num_epochs = training_parameters['num_epochs']
        self.batch_size = training_parameters['batch_size']

        # prepare the optimizers
        if self.Dataset == "MNIST":
            self.optimizer_adv = optim.SGD(self.model.parameters(), lr=training_parameters['learning_rate'],
                                           momentum=training_parameters['momentum'], weight_decay=training_parameters['decay'], nesterov=True)
        else:
            self.optimizer_adv = optim.Adam(self.model.parameters(), lr=training_parameters['lr'])

    def _parsing_parameters(self, **kwargs):
        """

        :param kwargs:
        :return:
        """
        assert kwargs is not None, "the parameters should be specified"

        print("\nUser configurations for the {} defense".format(self.defense_name))
        for key in kwargs:
            print('\t{} = {}'.format(key, kwargs[key]))

        self.epsilon = kwargs['eps']
        self.alpha = kwargs['alpha']

        return True

    def train_external_model_group(self, train_loader=None, validation_loader=None):
        """

        :param train_loader:
        :param validation_loader:
        :return:
        """
        # Set up the model group with 4 static external models
        if self.Dataset == 'MNIST':
            model_group = [MNIST_A(), MNIST_B(), MNIST_C(), MNIST_D()]
        else:
            model_group = [CIFAR10_A(), CIFAR10_B(), CIFAR10_C(), CIFAR10_D()]
        model_group = [model.to(self.device) for model in model_group]

        # training the models in model_group one by one
        for i in range(len(model_group)):

            # prepare the optimizer for MNIST
            if self.Dataset == "MNIST":
                optimizer_external = optim.SGD(model_group[i].parameters(), lr=self.training_parameters['learning_rate'],
                                               momentum=self.training_parameters['momentum'], weight_decay=self.training_parameters['decay'],
                                               nesterov=True)
            # prepare the optimizer for CIFAR10
            else:
                if i == 3:
                    optimizer_external = optim.SGD(model_group[i].parameters(), lr=0.001, momentum=0.9, weight_decay=1e-6)
                else:
                    optimizer_external = optim.Adam(model_group[i].parameters(), lr=self.training_parameters['lr'])

            print('\nwe are training the {}-th static external model ......'.format(i))
            best_val_acc = None
            for index_epoch in range(self.num_epochs):

                train_one_epoch(model=model_group[i], train_loader=train_loader, optimizer=optimizer_external, epoch=index_epoch,
                                device=self.device)
                val_acc = validation_evaluation(model=model_group[i], validation_loader=validation_loader, device=self.device)

                if self.Dataset == 'CIFAR10':
                    adjust_learning_rate(epoch=index_epoch, optimizer=optimizer_external)

                assert os.path.exists('../DefenseEnhancedModels/{}'.format(self.defense_name))
                defense_external_saver = '../DefenseEnhancedModels/{}/{}_EAT_{}.pt'.format(self.defense_name, self.Dataset, str(i))
                if not best_val_acc or round(val_acc, 4) >= round(best_val_acc, 4):
                    if best_val_acc is not None:
                        os.remove(defense_external_saver)
                    best_val_acc = val_acc
                    model_group[i].save(name=defense_external_saver)
                else:
                    print('Train Epoch {:>3}: validation dataset accuracy did not improve from {:.4f}\n'.format(index_epoch, best_val_acc))

    def load_external_model_group(self, model_dir='../DefenseEnhancedModels/EAT/', test_loader=None):
        """

        :param model_dir:
        :param test_loader:
        :return:
        """
        print("\n!!! Loading static external models ...")
        # Set up 4 static external models
        if self.Dataset == 'MNIST':
            model_group = [MNIST_A(), MNIST_B(), MNIST_C(), MNIST_D()]
        else:
            model_group = [CIFAR10_A(), CIFAR10_B(), CIFAR10_C(), CIFAR10_D()]
        model_group = [model.to(self.device) for model in model_group]

        for i in range(len(model_group)):
            print('loading the {}-th static external model'.format(i))
            model_path = '{}{}_EAT_{}.pt'.format(model_dir, self.Dataset, str(i))
            assert os.path.exists(model_path), "please train the external model first!!!"
            model_group[i].load(path=model_path, device=self.device)
            testing_evaluation(model=model_group[i], test_loader=test_loader, device=self.device)

        return model_group

    def random_fgsm_generation(self, model=None, natural_images=None):
        """
        A new randomized single step attack (RFGSM)
        :param model:
        :param natural_images:
        :return:
        """
        attack_model = model.to(self.device)
        attack_model.eval()

        with torch.no_grad():
            random_sign = torch.sign(torch.randn(*natural_images.size())).to(self.device)
            new_images = torch.clamp(natural_images + self.alpha * random_sign, min=0.0, max=1.0)

        new_images.requires_grad = True

        logits_attack = attack_model(new_images)
        # To avoid label leaking, we use the model's output instead of the true labels
        labels_attack = torch.max(logits_attack, dim=1)[1]
        loss_attack = F.cross_entropy(logits_attack, labels_attack)
        gradient = torch.autograd.grad(loss_attack, new_images)[0]

        new_images.requires_grad = False

        # generation of adversarial examples
        with torch.no_grad():
            xs_adv = new_images + (self.epsilon - self.alpha) * torch.sign(gradient)
            xs_adv = torch.clamp(xs_adv, min=0.0, max=1.0)
        return xs_adv

    def train_one_epoch_with_adv_from_external_models(self, pre_trained_models=None, train_loader=None, epoch=None):
        """

        :param pre_trained_models:
        :param train_loader:
        :param epoch:
        :return:
        """
        for index, (images, labels) in enumerate(train_loader):
            nat_images = images.to(self.device)
            nat_labels = labels.to(self.device)

            # in each mini_batch, we randomly choose the attack model which adversarial examples are generated on
            idx = np.random.randint(5)
            if idx == 0:
                attacking_model = self.model
            else:
                attacking_model = pre_trained_models[idx - 1]

            # get corresponding adversarial examples via RFGSM attack on the attack model
            adv_images = self.random_fgsm_generation(model=attacking_model, natural_images=nat_images)

            # set the model in the training mode
            self.model.train()

            logits_nat = self.model(nat_images)
            loss_nat = F.cross_entropy(logits_nat, nat_labels)
            logits_adv = self.model(adv_images)
            loss_adv = F.cross_entropy(logits_adv, nat_labels)
            loss = 0.5 * (loss_nat + loss_adv)

            self.optimizer_adv.zero_grad()
            loss.backward()
            self.optimizer_adv.step()

            print('\rTrain Epoch {:>3}: [{:>5}/{:>5}]  \tloss_nat={:.4f}, loss_adv={:.4f}, total_loss={:.4f} ===> '. \
                  format(epoch, (index + 1) * len(images), len(train_loader) * len(images), loss_nat, loss_adv, loss), end=' ')

    def defense(self, pre_trained_models=None, train_loader=None, validation_loader=None):

        best_val_acc = None
        for epoch in range(self.num_epochs):
            # training the model with natural examples and corresponding adversarial examples from external models
            self.train_one_epoch_with_adv_from_external_models(pre_trained_models=pre_trained_models, train_loader=train_loader, epoch=epoch)
            val_acc = validation_evaluation(model=self.model, validation_loader=validation_loader, device=self.device)

            # adjust the learning rate for CIFAR10
            if self.Dataset == 'CIFAR10':
                adjust_learning_rate(epoch=epoch, optimizer=self.optimizer_adv)

            # save the re-trained defense-enhanced model
            assert os.path.exists('../DefenseEnhancedModels/{}'.format(self.defense_name))
            defense_enhanced_saver = '../DefenseEnhancedModels/{}/{}_{}_enhanced.pt'.format(self.defense_name, self.Dataset, self.defense_name)
            if not best_val_acc or round(val_acc, 4) >= round(best_val_acc, 4):
                if best_val_acc is not None:
                    os.remove(defense_enhanced_saver)
                best_val_acc = val_acc
                self.model.save(name=defense_enhanced_saver)
            else:
                print('Train Epoch {:>3}: validation dataset accuracy did not improve from {:.4f}\n'.format(epoch, best_val_acc))
