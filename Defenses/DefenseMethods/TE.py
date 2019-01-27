#!/usr/bin/env python
# -*- coding: utf-8 -*-
# References: J. Buckman, et al., "Thermometer Encoding: One hot way to resist adversarial examples," in ICLR, 2018.
# Reference Implementation (TensorFlow): https://github.com/anishathalye/obfuscated-gradients/tree/master/thermometer
# **************************************
# @Time    : 2018/11/22 18:30
# @Author  : Jiaxu Zou
# @Lab     : nesa.zju.edu.cn
# @File    : TE.py
# **************************************

import os

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

from Defenses.DefenseMethods.defenses import Defense
from RawModels.ResNet import adjust_learning_rate


def one_hot_encoding(samples=None, level=None, device=None):
    """
    the help function to encode the samples using the one-hot encoding schema
    :param samples:
    :param level:
    :param device:
    :return:
    """
    assert level is not None and isinstance(level, int), 'level should specified as an integer'
    assert torch.is_tensor(samples), "input samples must be a PyTorch tensor"
    if len(samples.shape) >= 4 and (samples.shape[1] == 1 or samples.shape[1] == 3):
        samples = samples.permute(0, 2, 3, 1)

    # inserting the last position for samples (handle the upper bound by multiplying 0.9999)
    discretized_samples = torch.unsqueeze(input=(0.99999 * samples * level).long().to(device), dim=4)
    # make the last dim be the level number
    shape = discretized_samples.shape
    # convert to one_hot encoding
    one_hot_samples = torch.zeros([shape[0], shape[1], shape[2], shape[3], level]).to(device).scatter_(-1, discretized_samples, 1)
    one_hot_samples = one_hot_samples.float()

    return one_hot_samples


def thermometer_encoding(samples=None, level=None, device=None):
    """
    the help function to encode the samples using the thermometer encoding schema
    :param samples:
    :param level:
    :param device:
    :return:
    """
    assert level is not None and isinstance(level, int), 'level should specified as an integer'
    assert torch.is_tensor(samples), "input samples must be a PyTorch tensor"
    if len(samples.shape) >= 4 and (samples.shape[1] == 1 or samples.shape[1] == 3):
        samples = samples.permute(0, 2, 3, 1)

    # convert one hot encoding to thermometer encoding
    one_hot_samples = one_hot_encoding(samples=samples, level=level, device=device)
    therm_samples = torch.cumsum(one_hot_samples, dim=-1)

    # the returned samples is a type of numpy data with shape [BatchSize * (Channel * Level) * Weight* Height]
    shape = samples.shape
    therm_samples_numpy = torch.reshape(therm_samples, (shape[0], shape[1], shape[2], shape[3] * level))
    therm_samples_numpy = therm_samples_numpy.permute(0, 3, 1, 2).cpu().numpy()

    return therm_samples_numpy


class TEDefense(Defense):

    def __init__(self, model=None, defense_name=None, dataset=None, training_parameters=None, device=None, **kwargs):
        """

        :param model:
        :param defense_name:
        :param dataset:
        :param training_parameters:
        :param device:
        :param kwargs:
        """
        super(TEDefense, self).__init__(model=model, defense_name=defense_name)
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

        self.level = kwargs['level']
        self.steps = kwargs['steps']
        self.attack_eps = kwargs['attack_eps']
        self.attack_step_size = kwargs['attack_step_size']

        return True

    def lspga_generation(self, samples=None, ys=None, noise_init=True):
        """
        one type of white-box attacks on discretized inputs (thermometer encoding) -- Logit-Space Projected Gradient Ascent (LS-PGA)
        the detailed pseudo-code for LS-PGA attack is described in Algorithm 3 of the referenced paper
        :param samples:
        :param ys:
        :param noise_init:
        :return:
        """
        # STEP 1: sub-routine for getting an \epsilon-discretized masked of an image
        lowest = torch.clamp(samples - self.attack_eps, 0.0, 1.0)
        highest = torch.clamp(samples + self.attack_eps, 0.0, 1.0)

        # get the masking of intervals between lowest and highest
        masked_intervals = 0.0
        for alpha in np.linspace(0., 1., self.level):
            single_one_hot = one_hot_encoding(samples=alpha * lowest + (1. - alpha) * highest, level=self.level, device=self.device)
            masked_intervals += single_one_hot
        masked = (masked_intervals > 0.0).float()

        shape = masked.shape

        # STEP 2: main function for generating adversarial examples using LS-PGA
        # init each of logits randomly with values sampled from a standard normal distribution.
        if noise_init is True:
            us_numpy = torch.randn(shape).cpu().numpy()
        else:
            us_numpy = torch.zeros_like(masked).cpu().numpy()

        # generating
        inv_temp = 1.0
        sigma_rate = 1.2
        self.model.eval()
        for i in range(self.steps):
            us_logits = torch.from_numpy(us_numpy).to(self.device).float()
            us_logits.requires_grad = True
            # if not masked ( equal 0), turn it to be -inf (-99999)
            # then embedding the logits using softmax function with temperature to
            us_probs = F.softmax(inv_temp * (us_logits * masked - 999999.0 * (1. - masked)), dim=-1)

            # apply the cumulative sum function and reshape to get the distribution embedding
            thermometer_probs = torch.cumsum(us_probs, dim=-1)
            thermometer_probs = torch.reshape(thermometer_probs, (shape[0], shape[1], shape[2], shape[3] * self.level))
            # convert the channel back to the second position
            thermometer_probs = thermometer_probs.permute(0, 3, 1, 2)

            logits = self.model(thermometer_probs)

            if ys is None and i == 0:
                ys = torch.argmax(logits, dim=1)

            loss = F.cross_entropy(logits, ys)
            gradients = torch.autograd.grad(loss, us_logits)[0]
            signed_gradient = torch.sign(gradients).cpu().numpy()

            us_numpy += self.attack_step_size * signed_gradient
            inv_temp *= sigma_rate  # anneal the temperature via exponential decay with rate sigma

        us_logits = torch.from_numpy(us_numpy).to(self.device).float()
        logits_results = us_logits * masked - 999999.0 * (1. - masked)
        logits_final = torch.argmax(logits_results, dim=-1, keepdim=True)

        one_hot_adv_samples = torch.zeros([shape[0], shape[1], shape[2], shape[3], self.level]).to(self.device).scatter_(-1, logits_final, 1)
        one_hot_adv_samples = one_hot_adv_samples.float()

        # the returned samples is a type of numpy dataset
        therm_adv_samples = torch.cumsum(one_hot_adv_samples, dim=-1)
        final_adv_samples = torch.reshape(therm_adv_samples, (shape[0], shape[1], shape[2], shape[3] * self.level))
        final_adv_samples_numpy = final_adv_samples.permute(0, 3, 1, 2).cpu().numpy()

        return final_adv_samples_numpy

    def train_one_epoch_with_adv_lspga(self, train_loader=None, epoch=None, weight_regular=None):
        """

        :param train_loader:
        :param epoch:
        :return:
        """
        for index, (images, labels) in enumerate(train_loader):
            nat_images_numpy = thermometer_encoding(samples=images.to(self.device), level=self.level, device=self.device)
            nat_labels = labels.to(self.device)

            # prepare for LSPGA perturbation
            self.model.eval()
            adv_images_numpy = self.lspga_generation(samples=images.to(self.device))

            # concatenate the nature samples and adversarial examples
            batch_images_numpy = np.concatenate((nat_images_numpy, adv_images_numpy), axis=0)
            batch_images = torch.from_numpy(batch_images_numpy).to(self.device)
            # concatenate the true labels
            batch_labels = torch.cat((nat_labels, nat_labels), dim=0)

            # set the model in the training mode
            self.model.train()
            # forward the nn
            logits = self.model(batch_images)
            loss = F.cross_entropy(logits, batch_labels)

            # backward
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            print('\rTrain Epoch{:>3}: [batch:{:>4}/{:>4}]  \tLoss={:.4f} ===> '.format(epoch, index, len(train_loader), loss), end=' ')

    def thermometer_validation_evaluation(self, validation_loader, device):
        """
        validation evaluation with slight modification for thermometer encoded input samples
        :param validation_loader:
        :param device:
        :return:
        """
        self.model.eval()

        total = 0.0
        correct = 0.0
        with torch.no_grad():
            for index, (inputs, labels) in enumerate(validation_loader):
                therm_inputs = thermometer_encoding(samples=inputs.to(self.device), level=self.level, device=device)
                therm_inputs = torch.from_numpy(therm_inputs).to(self.device)
                labels = labels.to(device)

                outputs = self.model(therm_inputs)
                _, predicted = torch.max(outputs.data, 1)
                total = total + labels.size(0)
                correct = correct + (predicted == labels).sum().item()
            ratio = correct / total
        print('validation set accuracy is ', ratio)
        return ratio

    def defense(self, train_loader=None, validation_loader=None):

        best_val_acc = None
        for epoch in range(self.num_epochs):
            # training the model with nature examples and corresponding adversarial examples
            self.train_one_epoch_with_adv_lspga(train_loader=train_loader, epoch=epoch, weight_regular=1e-4)
            val_acc = self.thermometer_validation_evaluation(validation_loader=validation_loader, device=self.device)

            # adjust the learning rate for cifar10 training
            if self.Dataset == 'CIFAR10':
                adjust_learning_rate(optimizer=self.optimizer, epoch=epoch)

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
