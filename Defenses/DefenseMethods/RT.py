#!/usr/bin/env python
# -*- coding: utf-8 -*-
# References: C. Xie, et al., "Mitigating adversarial effects through randomization," in ICLR, 2018.
# Reference Implementation from Authors (TensorFlow): https://github.com/cihangxie/NIPS2017_adv_challenge_defense
# **************************************
# @Time    : 2018/11/12 17:03
# @Author  : Xiang Ling
# @Lab     : nesa.zju.edu.cn
# @File    : RT.py 
# **************************************

import numpy as np
import torch
from skimage.transform import rescale

from Defenses.DefenseMethods.defenses import Defense


class RTDefense(Defense):
    def __init__(self, model=None, defense_name=None, dataset=None, device=None):
        """

        :param model:
        :param defense_name:
        :param dataset:
        :param device:
        """
        super(RTDefense, self).__init__(model=model, defense_name=defense_name)
        self.model = model
        self.defense_name = defense_name
        self.device = device

        self.Dataset = dataset.upper()
        assert self.Dataset in ['MNIST', 'CIFAR10'], "The data set must be MNIST or CIFAR10"

    def randomization_transformation(self, samples=None, original_size=None, final_size=None):
        """

        :param samples:
        :param original_size:
        :param final_size:
        :return:
        """
        # Convert torch Tensor to numpy array
        if torch.is_tensor(samples) is True:
            samples = samples.cpu().numpy()
        # convert the channel of images
        samples = np.transpose(samples, (0, 2, 3, 1))
        assert samples.shape[-1] == 1 or samples.shape[-1] == 3, 'in the randomization transform function, channel must be placed in the last'

        transformed_samples = []
        # print ('transforming the images (size: {}) ...'.format(samples.shape))
        for image in samples:
            # Step 1: Random Resizing Layer
            # specify the random size which the image will be rescaled to
            rnd = np.random.randint(original_size, final_size)
            scale = (rnd * 1.0) / original_size
            rescaled_image = rescale(image=image, scale=scale, multichannel=True, preserve_range=True, mode='constant', anti_aliasing=False)

            # Step 2: Random Padding Layer
            h_rem = final_size - rnd
            w_rem = final_size - rnd
            pad_left = np.random.randint(0, w_rem)
            pad_right = w_rem - pad_left
            pad_top = np.random.randint(0, h_rem)
            pad_bottom = h_rem - pad_top
            # padding the image to the new size using gray pixels
            padded_image = np.pad(rescaled_image, ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)), 'constant', constant_values=0.5)
            transformed_samples.append(padded_image)

        # reset the channel location back and convert numpy back as the Tensor
        transformed_samples = np.array(transformed_samples)
        transformed_samples = torch.from_numpy(np.transpose(transformed_samples, (0, 3, 1, 2))).float().to(self.device)
        return transformed_samples

    def defense(self):
        print('As the defense of RT does not retrain the model, we do not implement this method')
