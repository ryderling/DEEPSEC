#!/usr/bin/env python
# -*- coding: utf-8 -*-
# References:   I. J. Goodfellow, J. Shlens, and C. Szegedy, "Explaining and harnessing adversarial examples," in ICLR, 2015.
# **************************************
# @Time    : 2018/9/7 23:05
# @Author  : Xiang Ling
# @Lab     : nesa.zju.edu.cn
# @File    : FGSM.py
# **************************************

import math

import numpy as np
import torch

from Attacks.AttackMethods.AttackUtils import tensor2variable
from Attacks.AttackMethods.attacks import Attack


class FGSMAttack(Attack):
    def __init__(self, model=None, epsilon=None):
        """
        :param model:
        :param epsilon:
        """
        super(FGSMAttack, self).__init__(model)
        self.model = model

        self.epsilon = epsilon

    def perturbation(self, samples, ys, device):
        """

        :param samples:
        :param ys:
        :param device:
        :return:
        """
        copy_samples = np.copy(samples)

        var_samples = tensor2variable(torch.from_numpy(copy_samples), device=device, requires_grad=True)
        var_ys = tensor2variable(torch.LongTensor(ys), device=device)

        self.model.eval()
        preds = self.model(var_samples)
        loss_fun = torch.nn.CrossEntropyLoss()
        loss = loss_fun(preds, torch.max(var_ys, 1)[1])
        loss.backward()
        gradient_sign = var_samples.grad.data.cpu().sign().numpy()

        adv_samples = copy_samples + self.epsilon * gradient_sign
        adv_samples = np.clip(adv_samples, 0.0, 1.0)

        return adv_samples

    def batch_perturbation(self, xs, ys, batch_size, device):
        """

        :param xs:
        :param ys:
        :param batch_size:
        :param device:
        :return:
        """
        assert len(xs) == len(ys), "The lengths of samples and its ys should be equal"

        adv_sample = []
        number_batch = int(math.ceil(len(xs) / batch_size))
        for index in range(number_batch):
            start = index * batch_size
            end = min((index + 1) * batch_size, len(xs))
            print('\r===> in batch {:>2}, {:>4} ({:>4} in total) nature examples are perturbed ... '.format(index, end - start, end), end=' ')

            batch_adv_images = self.perturbation(xs[start:end], ys[start:end], device)
            adv_sample.extend(batch_adv_images)
        return np.array(adv_sample)
