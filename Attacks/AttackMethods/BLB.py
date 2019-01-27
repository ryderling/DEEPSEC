#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Reference: C. Szegedy, et al., "Intriguing properties of neural networks," in ICLR, 2014.
# **************************************
# @Time    : 2018/9/20 14:04
# @Author  : Jiannan Wang & Saizhuo Wang
# @Lab     : nesa.zju.edu.cn
# @File    : BLB.py
# **************************************

import math

import numpy as np
import torch

from Attacks.AttackMethods.AttackUtils import tensor2variable
from Attacks.AttackMethods.attacks import Attack


class BLBAttack(Attack):

    def __init__(self, model=None, init_const=1e-2, binary_search_steps=5, max_iterations=1000):
        """

        :param model:
        :param init_const:
        :param binary_search_steps:
        :param max_iterations:
        """
        super(BLBAttack, self).__init__(model)
        self.model = model

        self.init_const = init_const
        self.binary_search_steps = binary_search_steps
        self.max_iterations = max_iterations

    def perturbation(self, samples, ys_targets, batch_size, device):
        """

        :param samples:
        :param ys_targets:
        :param batch_size:
        :param device:
        :return:
        """
        assert len(samples) == batch_size, "the length of sample is not equal to the batch_size"

        copy_samples = np.copy(samples)
        var_samples = tensor2variable(torch.from_numpy(copy_samples), device=device)
        var_targets = tensor2variable(torch.LongTensor(ys_targets), device=device)

        # set the lower and upper bound for searching 'c' const
        const_origin = np.ones(shape=batch_size, dtype=float) * self.init_const
        c_upper_bound = [1e10] * batch_size
        c_lower_bound = np.zeros(batch_size)

        # L2 norm attack
        best_l2 = [1e10] * batch_size
        best_perturbation = np.zeros(var_samples.size())
        current_prediction_class = [-1] * batch_size

        def attack_achieved(pre_softmax, target_class):
            return np.argmax(pre_softmax) == target_class

        # note that we should turn off the reduction when applying the loss function to a batch of samples
        loss_fun = torch.nn.CrossEntropyLoss(reduction='none')

        self.model.eval()
        # Outer loop for linearly searching for c
        for search_for_c in range(self.binary_search_steps):
            # the perturbation
            r = torch.zeros_like(var_samples).float()
            r = tensor2variable(r, device=device, requires_grad=True)

            # use LBFGS to optimize the perturbation r, with default learning rate parameter and other parameters
            optimizer = torch.optim.LBFGS([r], max_iter=self.max_iterations)
            var_const = tensor2variable(torch.FloatTensor(const_origin), device=device)
            print("\tbinary search step {}:".format(search_for_c))

            # The steps to be done when doing optimization iteratively.
            def closure():
                perturbed_images = torch.clamp(var_samples + r, min=0.0, max=1.0)
                prediction = self.model(perturbed_images)
                l2dist = torch.sum((perturbed_images - var_samples) ** 2, [1, 2, 3])
                constraint_loss = loss_fun(prediction, var_targets)
                loss_f = var_const * constraint_loss
                loss = l2dist.sum() + loss_f.sum()  # minimize c|r| + loss_f(x+r,l), l is the target label, r is the perturbation
                optimizer.zero_grad()
                loss.backward(retain_graph=True)
                return loss

            optimizer.step(closure)

            perturbed_images = torch.clamp(var_samples + r, min=0.0, max=1.0)
            prediction = self.model(perturbed_images)
            l2dist = torch.sum((perturbed_images - var_samples) ** 2, [1, 2, 3])

            # the following is analogy to CW2 attack
            for i, (dist, score, perturbation) in enumerate(
                    zip(l2dist.data.cpu().numpy(), prediction.data.cpu().numpy(), perturbed_images.data.cpu().numpy())):
                if dist < best_l2[i] and attack_achieved(score, ys_targets[i]):
                    best_l2[i] = dist
                    current_prediction_class[i] = np.argmax(score)
                    best_perturbation[i] = perturbation

            # update the best constant c for each sample in the batch
            for i in range(batch_size):
                if current_prediction_class[i] == ys_targets[i] and current_prediction_class[i] != -1:
                    c_upper_bound[i] = min(c_upper_bound[i], const_origin[i])
                    if c_upper_bound[i] < 1e10:
                        const_origin[i] = (c_lower_bound[i] + c_upper_bound[i]) / 2.
                else:
                    c_lower_bound[i] = max(c_lower_bound[i], const_origin[i])
                    if c_upper_bound[i] < 1e10:
                        const_origin = (c_lower_bound[i] + c_upper_bound[i]) / 2
                    else:
                        const_origin[i] *= 10
        return np.array(best_perturbation)

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

            batch_adv_images = self.perturbation(xs[start:end], ys_target[start:end], batch_size, device)
            adv_sample.extend(batch_adv_images)
        return np.array(adv_sample)
