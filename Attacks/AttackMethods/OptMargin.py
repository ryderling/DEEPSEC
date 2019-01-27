#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Reference: W. He, B. Li, and D. Song, "Decision boundary analysis of adversarial examples," in ICLR, 2018.
# Reference Implementation from Authors (TensorFlow): https://github.com/sunblaze-ucb/decision-boundaries
# **************************************
# @Time    : 2018/10/17 23:03
# @Author  : Saizuo Wang
# @Lab     : nesa.zju.edu.cn
# @File    : OptMargin.py
# **************************************


import math

import numpy as np
import torch

from Attacks.AttackMethods.AttackUtils import tensor2variable
from Attacks.AttackMethods.attacks import Attack


class OMAttack(Attack):
    def __init__(self, model=None, kappa=0, init_const=0.001, lr=0.02, binary_search_steps=5, max_iters=10000, lower_bound=0.0, upper_bound=1.0,
                 noise_count=20, noise_magnitude=0.3):
        """

        :param model:
        :param kappa:
        :param init_const:
        :param lr:
        :param binary_search_steps:
        :param max_iters:
        :param lower_bound:
        :param upper_bound:
        :param noise_count:
        :param noise_magnitude:
        """
        super(OMAttack, self).__init__(model)
        self.model = model

        self.kappa = kappa * 1.0
        self.learning_rate = lr
        self.init_const = init_const
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.max_iterations = max_iters
        self.binary_search_steps = binary_search_steps

        self.noise_count = noise_count
        self.noise_magnitude = noise_magnitude

    def perturbation(self, samples, true_label, batch_size, device):
        """

        :param samples:
        :param true_label:
        :param batch_size:
        :param device:
        :return:
        """
        assert len(samples) == batch_size, "the length of sample is not equal to the batch_size"

        # initialize noise
        channel, width, length = samples.shape[1:]
        noise_raw = np.random.normal(scale=self.noise_magnitude, size=(channel * length * width, self.noise_count)).astype(np.float32)
        noise_unit_vector, _ = np.linalg.qr(noise_raw)  # turn the noises to orthogonal unit vectors using QR

        assert noise_unit_vector.shape[1] == self.noise_count
        # noise_vector = noise_unit_vector * np.sqrt(channel * width * length) * self.noise_magnitude
        noise_vector = noise_unit_vector * (1.0 / np.max(np.abs(noise_unit_vector))) * self.noise_magnitude
        noise_vector = noise_vector.transpose((1, 0)).reshape((self.noise_count, channel, width, length))
        noise_vector[self.noise_count - 1] = 0  # set the last noise to 0
        noise_vector = tensor2variable(torch.from_numpy(noise_vector), device, requires_grad=False)

        # transform the samples [lower, upper] to [-1, 1] and then to the arctanh space
        mid_point = (self.upper_bound + self.lower_bound) * 0.5
        half_range = (self.upper_bound - self.lower_bound) * 0.5
        arctanh_samples = np.arctanh((samples - mid_point) / half_range * 0.9999)
        var_samples = tensor2variable(torch.from_numpy(arctanh_samples), device=device, requires_grad=True)

        # set the lower and upper bound for searching 'c' in the OM attack
        const_origin = np.ones(shape=batch_size, dtype=float) * self.init_const
        c_upper_bound = [1e10] * batch_size
        c_lower_bound = np.zeros(batch_size)

        # convert targets to one-hot encoder
        temp_one_hot_matrix = np.eye(10)
        labels_in_one_hot = []
        for i in range(batch_size):
            current_label = temp_one_hot_matrix[true_label[i]]
            labels_in_one_hot.append(current_label)
        labels_in_one_hot = tensor2variable(torch.FloatTensor(np.array(labels_in_one_hot)), device=device)

        best_l2 = [1e10] * batch_size
        best_perturbation = np.zeros(var_samples.size())
        current_prediction_class = [-1] * batch_size

        def un_targeted_attack_achieved(pre_softmax, true_class):
            pre_softmax[true_class] += self.kappa
            return np.argmax(pre_softmax) != true_class

        self.model.eval()
        # Outer loop for linearly searching for c
        for search_for_c in range(self.binary_search_steps):

            modifier = torch.zeros(var_samples.size()).float()
            modifier = tensor2variable(modifier, device=device, requires_grad=True)
            optimizer = torch.optim.Adam([modifier], lr=self.learning_rate)
            var_const = tensor2variable(torch.FloatTensor(const_origin), device=device)

            print("\tbinary search step {}:".format(search_for_c))
            for iteration_times in range(self.max_iterations):
                # inverse the transform tanh -> [0, 1]
                perturbed_img = torch.tanh(var_samples + modifier) * half_range + mid_point
                perturbed_img = torch.clamp(perturbed_img, min=0.0, max=1.0)

                perturbed_img_plus_noises = perturbed_img[None, :, :, :, :] + noise_vector[:, None, :, :, :]
                perturbed_img_plus_noises = torch.clamp(perturbed_img_plus_noises, min=0.0, max=1.0)
                # size = noise_count * batch_size * channel * width * height #

                # minimize |r| + c * loss_f(x+r,l)
                l2dist = torch.sum((perturbed_img - (torch.tanh(var_samples) * half_range + mid_point)) ** 2, [1, 2, 3])

                loss = l2dist.clone()

                # add the 20 loss terms one by one
                for i in range(self.noise_count):
                    prediction = self.model(perturbed_img_plus_noises[i])
                    c_loss = torch.max((prediction * labels_in_one_hot).sum(1) - (prediction - 1e10 * labels_in_one_hot).max(1)[0],
                                       torch.ones(batch_size, device=device) * self.kappa * -1)
                    loss += var_const * c_loss

                loss = loss.sum()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                for i, (dist, score, img) in enumerate(
                        zip(l2dist.data.cpu().numpy(), prediction.data.cpu().numpy(), perturbed_img.data.cpu().numpy())):
                    if dist < best_l2[i] and un_targeted_attack_achieved(score, true_label[i]):
                        best_l2[i] = dist
                        current_prediction_class[i] = np.argmax(score)
                        best_perturbation[i] = img

            for i in range(batch_size):
                if current_prediction_class[i] != true_label[i] and current_prediction_class[i] != -1:
                    c_upper_bound[i] = min(c_upper_bound[i], const_origin[i])
                    if c_upper_bound[i] < 1e10:
                        const_origin[i] = (c_lower_bound[i] + c_upper_bound[i]) / 2.0
                else:
                    c_lower_bound[i] = max(c_lower_bound[i], const_origin[i])
                    if c_upper_bound[i] < 1e10:
                        const_origin = (c_lower_bound[i] + c_upper_bound[i]) / 2.0
                    else:
                        const_origin[i] *= 10

        return np.array(best_perturbation)

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

            batch_adv_images = self.perturbation(xs[start:end], ys[start:end], batch_size, device)
            adv_sample.extend(batch_adv_images)
        return np.array(adv_sample)
