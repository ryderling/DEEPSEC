#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Reference: Y. Song, et al. "Pixeldefend: Leveraging generative models to understand and defend against adversarial examples," in ICLR, 2018.
# Reference Implementation from Authors (TensorFlow): https://github.com/yang-song/pixeldefend
# **************************************
# @Time    : 2018/11/23 21:41
# @Author  : Saizhuo Wang & Xiang Ling
# @Lab     : nesa.zju.edu.cn
# @File    : PD.py 
# **************************************

import math
import os

import numpy as np
import torch
import torch.nn as nn

from Defenses.DefenseMethods.defenses import Defense

try:

    from Defenses.DefenseMethods.External.pixel_cnn_pp.model import PixelCNN
    from Defenses.DefenseMethods.External.pixel_cnn_pp.utils import decode, load_part_of_model
except:
    print('please git clone the repo [] and train the generative PixelCNN model first')
    raise ImportError

rescaling = lambda x: (x - 0.5) * 2
inv_rescaling = lambda x: x * 0.5 + 0.5
res_1_to_255 = lambda x: x * 127.5 + 127.5
res_255_to_1 = lambda x: (x - 127.5) / 127.5


class PixelDefend(Defense):
    def __init__(self, model=None, defense_name=None, dataset=None, pixel_cnn_dir=None, device=None):

        super(PixelDefend, self).__init__(model=model, defense_name=defense_name)
        self.model = model
        self.defense_name = defense_name
        self.device = device

        self.Dataset = dataset.upper()
        assert self.Dataset in ['MNIST', 'CIFAR10'], "The data set must be MNIST or CIFAR10"

        # load the trained PixelCNN model
        # The structure of PixelCNN is fixed as follows in this implementation, the same as https://github.com/SaizhuoWang/pixel-cnn-pp
        self.pixel_cnn_model = PixelCNN(nr_resnet=5, nr_filters=160, nr_logistic_mix=10, resnet_nonlinearity='concat_elu',
                                        input_channels=3 if self.Dataset == 'CIFAR10' else 1).to(self.device)
        self.pixel_cnn_model = nn.DataParallel(self.pixel_cnn_model)
        self.load_pixel_cnn_model(dir=pixel_cnn_dir)

    def load_pixel_cnn_model(self, dir=None):

        pixel_cnn_model_location = '{}DefenseMethods/External/pixel_cnn_pp/models/{}_pixel_cnn.pth'.format(dir, self.Dataset)
        print('\nstarting to load the pixel cnn model from ', pixel_cnn_model_location)
        assert os.path.exists(pixel_cnn_model_location), "the pixel cnn model in {} does not exist, please try the model first !".format(
            pixel_cnn_model_location)
        load_part_of_model(model=self.pixel_cnn_model, path=pixel_cnn_model_location)

    def de_noising_samples(self, samples=None, batch_size=20, eps=None):
        """

        :param samples:
        :param eps:
        :return:
        """
        # samples.shape = (B, C, W, H)
        assert len(samples.shape) == 4 and isinstance(samples, (np.ndarray, np.generic)), \
            "input samples should be type of numpy with 4 dimensions"
        assert samples.shape[0] == batch_size, 'make sure the batch_size in the first dimension'
        channel = samples.shape[1]
        assert channel == 1 or channel == 3, "the second dimension should be the channel"

        copy_samples = np.copy(samples)
        copy_samples = torch.from_numpy(copy_samples).to(self.device).float()
        copy_samples = rescaling(copy_samples)  # [0, 1] ==> [-1, 1]

        assert eps < 1.0 and eps > 0.0
        int_epsilon = int(round(eps * 255.0, 0))

        width, height = samples.shape[2], samples.shape[3]
        for i in range(width):
            for j in range(height):
                output = self.pixel_cnn_model(copy_samples, sample=True)
                out = decode(copy_samples, output, self.Dataset, self.device)

                copy_sample_de_norm = res_1_to_255(copy_samples)  # [-1, 1] ==> [0, 255]
                copy_sample_int = copy_sample_de_norm.clone().int()
                lb = torch.clamp(copy_sample_int - int_epsilon, min=0)
                ub = torch.clamp(copy_sample_int + int_epsilon, max=255)
                template = (torch.range(0, 255, step=1, dtype=torch.int).to(self.device) + torch.zeros_like(copy_sample_int, dtype=torch.int)[
                    ..., None]).to(self.device)
                lb = lb[..., None] + torch.zeros_like(template, dtype=torch.int)
                ub = ub[..., None] + torch.zeros_like(template, dtype=torch.int)

                template = torch.clamp((torch.lt(template, lb) + torch.gt(template, ub)), max=1, min=0).float()
                template = template.permute(0, 2, 3, 1, 4)
                out = out - template * 1e10  # out.shape = (B, W, H, C, 256)
                out = res_255_to_1(torch.argmax(out, dim=4).permute(0, 3, 1, 2).float())  # [0, 255] -> [-1, 1]
                # out.shape = (B, C, W, H)
                copy_samples[:, :, i, j] = out.data[:, :, i, j]
        copy_sample = inv_rescaling(copy_samples)

        return copy_sample.data.cpu().numpy()

    def de_noising_samples_batch(self, samples=None, batch_size=20, eps=None):

        purified_images = []
        number_batch = int(math.ceil(len(samples) / batch_size))
        for index in range(number_batch):
            start = index * batch_size
            end = min((index + 1) * batch_size, len(samples))
            print('\r===> in batch {:>2}, {:>4} ({:>4} in total) samples are purified ... '.format(index, end - start, end), end=' ')
            rtn = self.de_noising_samples(samples=samples[start:end], batch_size=batch_size, eps=eps)
            purified_images.extend(rtn)
        return np.array(purified_images)

    def defense(self):
        print('As the defense of PixelDefend does not retrain the model, we do not implement this method')
