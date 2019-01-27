#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Reference: P. Chen, et al., "EAD: elastic-net attacks to deep neural networks via adversarial examples," in AAAI, 2018.
# Reference Implementation from Authors (TensorFlow): https://github.com/ysharma1126/EAD_Attack
# **************************************
# @Time    : 2018/11/1 19:01
# @Author  : Saizuo Wang & Xiang Ling
# @Lab     : nesa.zju.edu.cn
# @File    : EAD.py
# **************************************


import math

import numpy as np
import torch

from Attacks.AttackMethods.AttackUtils import tensor2variable
from Attacks.AttackMethods.attacks import Attack


class EADAttack(Attack):
    def __init__(self, model=None, kappa=0, init_const=0.001, lr=0.02, binary_search_steps=5, max_iters=10000, lower_bound=0.0, upper_bound=1.0,
                 beta=1e-3, EN=True):
        super(EADAttack, self).__init__(model)
        self.model = model

        self.kappa = kappa * 1.0
        self.learning_rate = lr
        self.init_const = init_const
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.max_iterations = max_iters
        self.binary_search_steps = binary_search_steps

        self.beta = beta
        self.EN = EN
        if self.EN:
            print("\nEN Decision Rule")
        else:
            print('\nL1 Decision Rule')

    def perturbation(self, samples, ys_targets, batch_size, device):

        assert len(samples) == batch_size, "the length of sample is not equal to the batch_size"
        copy_sample = np.copy(samples)

        # help function
        def attack_achieved(pre_softmax, target_class):
            pre_softmax[target_class] -= self.kappa
            return np.argmax(pre_softmax) == target_class

        # help function: Iterative Shrinkage-Threshold-ing Algorithm
        def ISTA(new, old):
            with torch.no_grad():
                diff = new - old
                var_beta = torch.FloatTensor(np.ones(shape=diff.shape, dtype=float) * self.beta).to(device)
                # test if the perturbation is out of bound. If it is, then reduce the perturbation by beta
                cropped_diff = torch.max(torch.abs(diff) - var_beta, torch.zeros(diff.shape, device=device)) * diff.sign().to(device)
                fist_new = old + cropped_diff
                return torch.clamp(input=fist_new, min=0.0, max=1.0)

        # transform the samples [lower_bound, upper_bound] to [0, 1]
        var_samples = tensor2variable(torch.from_numpy(samples), device=device, requires_grad=True)

        # set the lower and upper bound for searching 'c' const in EAD attack
        const_origin = np.ones(shape=batch_size, dtype=float) * self.init_const
        c_upper_bound = [1e10] * batch_size
        c_lower_bound = np.zeros(batch_size)

        # convert targets to one hot encoder
        temp_one_hot_matrix = np.eye(10)
        targets_one_hot = []
        for i in range(batch_size):
            current_target = temp_one_hot_matrix[ys_targets[i]]
            targets_one_hot.append(current_target)
        targets_one_hot = torch.FloatTensor(np.array(targets_one_hot)).to(device)

        # initialize
        best_elastic = [1e10] * batch_size
        best_perturbation = np.zeros(var_samples.size())
        current_prediction_class = [-1] * batch_size

        flag = [False] * batch_size

        self.model.eval()
        # Outer loop for linearly searching for c
        for search_for_c in range(self.binary_search_steps):

            slack = tensor2variable(torch.from_numpy(copy_sample), device=device, requires_grad=True)  # The slack variable (y) of x
            optimizer_y = torch.optim.SGD([slack], lr=self.learning_rate)
            old_image = slack.clone()  # Save the previous version of new_img in the iteration
            var_const = tensor2variable(torch.FloatTensor(const_origin), device=device)
            print("\tbinary search step {}:".format(search_for_c))

            for iteration_times in range(self.max_iterations):
                # optimize the slack variable
                output_y = self.model(slack).to(device)
                l2dist_y = torch.sum((slack - var_samples) ** 2, [1, 2, 3])
                kappa_t = torch.FloatTensor([self.kappa] * batch_size).to(device)
                target_loss_y = torch.max((output_y - 1e10 * targets_one_hot).max(1)[0] - (output_y * targets_one_hot).sum(1), -1 * kappa_t)
                c_loss_y = var_const * target_loss_y
                loss_y = l2dist_y.sum() + c_loss_y.sum()

                optimizer_y.zero_grad()
                loss_y.backward()
                optimizer_y.step()

                # convert to new image and save the previous version
                new_image = ISTA(slack, var_samples)
                slack.data = new_image.data + ((iteration_times / (iteration_times + 3.0)) * (new_image - old_image)).data
                old_image = new_image.clone()

                # calculate the loss for decision
                output = self.model(new_image)
                l1dist = torch.sum(torch.abs(new_image - var_samples), [1, 2, 3])
                l2dist = torch.sum((new_image - var_samples) ** 2, [1, 2, 3])
                target_loss = torch.max((output - 1e10 * targets_one_hot).max(1)[0] - (output * targets_one_hot).sum(1), -1 * kappa_t)

                if self.EN:
                    decision_loss = self.beta * l1dist + l2dist + var_const * target_loss
                else:
                    decision_loss = self.beta * l1dist + var_const * target_loss

                # Update best results
                for i, (dist, score, img) in enumerate(
                        zip(decision_loss.data.cpu().numpy(), output.data.cpu().numpy(), new_image.data.cpu().numpy())):
                    if dist < best_elastic[i] and attack_achieved(score, ys_targets[i]):
                        best_elastic[i] = dist
                        current_prediction_class[i] = np.argmax(score)
                        best_perturbation[i] = img
                        flag[i] = True

            # update the best constant c for each sample in the batch
            for i in range(batch_size):
                if current_prediction_class[i] == ys_targets[i] and current_prediction_class[i] != -1:
                    c_upper_bound[i] = min(c_upper_bound[i], const_origin[i])
                    if c_upper_bound[i] < 1e10:
                        const_origin[i] = (c_lower_bound[i] + c_upper_bound[i]) / 2.0
                else:
                    c_lower_bound[i] = max(c_lower_bound[i], const_origin[i])
                    if c_upper_bound[i] < 1e10:
                        const_origin = (c_lower_bound[i] + c_upper_bound[i]) / 2.0
                    else:
                        const_origin[i] *= 10

        cnt = 0
        for i in range(batch_size):
            if flag[i]:
                cnt += 1
        print("Success: {}".format(cnt))

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
