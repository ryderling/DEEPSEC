#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Reference: X. Cao and N. Z. Gong, "Mitigating evasion attacks to deep neural networks via region-based classification," in ACSAC, 2017.
# **************************************
# @Time    : 2018/9/19 20:09
# @Author  : Jiaxu Zou & Xiang Ling
# @Lab     : nesa.zju.edu.cn
# @File    : RC.py
# **************************************

import numpy as np
import torch

from Defenses.DefenseMethods.defenses import Defense
from RawModels.Utils.TrainTest import validation_evaluation


class RCDefense(Defense):
    def __init__(self, model=None, defense_name='RC', dataset=None, device=None, num_points=100):
        """

        :param model:
        :param defense_name:
        :param dataset:
        :param device:
        :param num_points:
        """
        super(RCDefense, self).__init__(model=model, defense_name=defense_name)
        self.model = model
        self.defense_name = defense_name
        self.device = device

        self.Dataset = dataset.upper()
        assert self.Dataset in ['MNIST', 'CIFAR10'], "The data set must be MNIST or CIFAR10"

        # parameters for the region-based classification defense
        self.num_points = num_points

    def search_best_radius(self, validation_loader=None, radius_min=0.0, radius_max=1.0, radius_step=0.01):
        """

        :param validation_loader:
        :param radius_min:
        :param radius_max:
        :param radius_step:
        :return:
        """
        self.model.eval()
        with torch.no_grad():
            # compute the original classification accuracy on validation dataset
            val_acc = validation_evaluation(model=self.model, validation_loader=validation_loader, device=self.device)
            print('<--- original classification accuracy on validation dataset is {:.4f} --->'.format(val_acc))

            # learn the radius through a search process
            total_step = int((radius_max - radius_min) / radius_step)
            for index in range(total_step):

                # update the radius
                tmp_radius = radius_min + radius_step * (index + 1)

                # calculate the accuracy of region-based classification on validation dataset
                total = 0.0
                correct = 0.0
                for images, labels in validation_loader:
                    rc_preds = self.region_based_classification(samples=images, radius=tmp_radius)
                    rc_labels = torch.from_numpy(rc_preds)

                    correct += (rc_labels == labels).sum().item()
                    total += labels.size(0)
                rc_acc = correct / total

                print('\tcurrent radius is {:.2f}, validation accuracy is {:.1f}/{:.1f}={:.5f}'.format(tmp_radius, correct, total, rc_acc))

                if (val_acc - rc_acc) > 1e-2:
                    return round(tmp_radius - radius_step, 2)

            return radius_max

    def region_based_classification_single(self, sample, radius):
        """

        :param sample: one sample (1*channel*H*W)
        :param radius:
        :return:
        """
        self.model.eval()

        assert sample.shape[0] == 1, "the sample parameter should be one example in numpy format"
        copy_sample = np.copy(sample)

        with torch.no_grad():
            copy_sample = torch.from_numpy(copy_sample).to(self.device)

            # prepare the hypercube samples (size=num_points) for the sample (size=1)
            hypercube_samples = copy_sample.repeat(self.num_points, 1, 1, 1).to(self.device).float()
            random_space = torch.Tensor(*hypercube_samples.size()).to(self.device).float()
            random_space.uniform_(-radius, radius)
            hypercube_samples = torch.clamp(hypercube_samples + random_space, min=0.0, max=1.0)

            # predicting for hypercube samples
            hypercube_preds = self.model(hypercube_samples)
            hypercube_labels = torch.max(hypercube_preds, dim=1)[1]

            # voting for predicted labels
            bin_count = torch.bincount(hypercube_labels)
            rc_label = torch.max(bin_count, dim=0)[1]

            return rc_label.cpu().numpy()

    def region_based_classification(self, samples, radius):
        """

        :param samples: batch samples (batch_size*channel*H*W)
        :param radius:
        :return:
        """
        self.model.eval()
        rc_labels = []
        for i in range(samples.shape[0]):
            x = samples[i: i + 1]
            label = self.region_based_classification_single(sample=x, radius=radius)
            rc_labels.append(label)
        return np.array(rc_labels)

    def defense(self):
        print('As the defense of RT does not retrain the model, we do not implement this method')
