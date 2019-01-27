#!/usr/bin/env python
# -*- coding: utf-8 -*-
# References:   A. Kurakin, I. Goodfellow, and S. Bengio, "Adversarial examples in the physical world," in ICLR, 2017.
# **************************************
# @Time    : 2018/9/9 16:45
# @Author  : Xiang Ling
# @Lab     : nesa.zju.edu.cn
# @File    : LLC.py
# **************************************

import math

import numpy as np
import torch

from Attacks.AttackMethods.AttackUtils import tensor2variable
from Attacks.AttackMethods.attacks import Attack


class LLCAttack(Attack):

    def __init__(self, model=None, epsilon=None):
        """

        :param model:
        :param epsilon:
        """
        super(LLCAttack, self).__init__(model)
        self.model = model

        self.epsilon = epsilon

    def perturbation(self, samples, ys_target, device):
        """

        :param samples:
        :param ys_target:
        :param device:
        :return:
        """
        copy_samples = np.copy(samples)

        var_samples = tensor2variable(torch.from_numpy(copy_samples), device=device, requires_grad=True)
        var_ys_target = tensor2variable(torch.from_numpy(ys_target), device)

        self.model.eval()
        preds = self.model(var_samples)
        loss_fun = torch.nn.CrossEntropyLoss()
        loss = loss_fun(preds, var_ys_target)
        loss.backward()
        gradient_sign = var_samples.grad.data.cpu().sign().numpy()

        adv_samples = copy_samples - self.epsilon * gradient_sign
        adv_samples = np.clip(adv_samples, 0.0, 1.0)

        return adv_samples

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
