#!/usr/bin/env python
# -*- coding: utf-8 -*-
# References:   F. Tram`er, et al.,"Ensemble adversarial training: Attacks and defenses," in ICLR, 2018.
# **************************************
# @Time    : 2018/9/9 18:18
# @Author  : Xiang Ling
# @Lab     : nesa.zju.edu.cn
# @File    : RLLC.py
# **************************************

import math

import numpy as np
import torch

from Attacks.AttackMethods.AttackUtils import tensor2variable
from Attacks.AttackMethods.attacks import Attack


class RLLCAttack(Attack):

    def __init__(self, model=None, epsilon=None, alpha_ratio=None):
        """

        :param model:
        :param epsilon:
        :param alpha:
        """
        super(RLLCAttack, self).__init__(model)
        self.model = model

        self.epsilon = epsilon
        self.alpha_ratio = alpha_ratio

    def perturbation(self, samples, ys_target, device):
        """

        :param samples:
        :param ys_target:
        :return:
        """
        copy_samples = np.copy(samples)

        # randomization
        copy_samples = np.clip(copy_samples + self.alpha_ratio * self.epsilon * np.sign(np.random.randn(*copy_samples.shape)), 0.0, 1.0).astype(
            np.float32)

        var_samples = tensor2variable(torch.from_numpy(copy_samples), device=device, requires_grad=True)
        var_ys_target = tensor2variable(torch.from_numpy(ys_target), device)

        eps = (1 - self.alpha_ratio) * self.epsilon

        self.model.eval()
        preds = self.model(var_samples)
        loss_fun = torch.nn.CrossEntropyLoss()
        loss = loss_fun(preds, var_ys_target)
        loss.backward()
        gradient_sign = var_samples.grad.data.cpu().sign().numpy()

        adv_samples = copy_samples - eps * gradient_sign
        adv_samples = np.clip(adv_samples, 0, 1)
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
