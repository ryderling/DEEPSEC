#!/usr/bin/env python
# -*- coding: utf-8 -*-
# References:   A. Kurakin, I. Goodfellow, and S. Bengio, "Adversarial examples in the physical world," in ICLR, 2017.
# **************************************
# @Time    : 2018/9/9 20:13
# @Author  : Xiang Ling
# @Lab     : nesa.zju.edu.cn
# @File    : ILLC.py
# **************************************

import math

import numpy as np
import torch

from Attacks.AttackMethods.AttackUtils import tensor2variable
from Attacks.AttackMethods.attacks import Attack


class ILLCAttack(Attack):
    def __init__(self, model=None, epsilon=None, eps_iter=None, num_steps=5):
        """

        :param model:
        :param epsilon:
        :param eps_iter:
        :param num_steps:
        """
        super(ILLCAttack, self).__init__(model)
        self.model = model

        self.epsilon = epsilon
        self.epsilon_iter = eps_iter
        self.num_steps = num_steps

    def perturbation(self, samples, ys_target, device):
        """

        :param samples:
        :param ys_target:
        :param device:
        :return:
        """
        copy_samples = np.copy(samples)
        var_ys_target = tensor2variable(torch.from_numpy(ys_target), device)

        for index in range(self.num_steps):
            var_samples = tensor2variable(torch.from_numpy(copy_samples), device=device, requires_grad=True)
            self.model.eval()
            preds = self.model(var_samples)

            loss_fun = torch.nn.CrossEntropyLoss()
            loss = loss_fun(preds, var_ys_target)
            loss.backward()
            gradient_sign = var_samples.grad.data.cpu().sign().numpy()

            copy_samples = copy_samples - self.epsilon_iter * gradient_sign
            copy_samples = np.clip(copy_samples, samples - self.epsilon, samples + self.epsilon)
            copy_samples = np.clip(copy_samples, 0.0, 1.0)

        return copy_samples

    def batch_perturbation(self, xs, ys_target, batch_size, device):
        """

        :param xs:
        :param ys_target:
        :param batch_size:
        :param device:
        :return:
        """
        assert len(xs) == len(ys_target), "The lengths of samples and its ys should be equal"

        adv_sample = []
        number_batch = int(math.ceil(len(xs) / batch_size))
        for index in range(number_batch):
            start = index * batch_size
            end = min((index + 1) * batch_size, len(xs))
            print('\r===> in batch {:>2}, {:>4} ({:>4} in total) nature examples are perturbed ... '.format(index, end - start, end), end=' ')

            batch_adv_images = self.perturbation(xs[start:end], ys_target[start:end], device)

            adv_sample.extend(batch_adv_images)
        return np.array(adv_sample)
