#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Reference: C. Guo et al., "Countering Adversarial Images Using Input Transformations" in ICLR, 2018.
# Reference Implementation from Authors (TensorFlow): https://github.com/facebookarchive/adversarial_image_defenses
# **************************************
# @Time    : 2018/11/23 20:30
# @Author  : Jiaxu Zou
# @Lab     : nesa.zju.edu.cn
# @File    : EIT.py
# **************************************

import os
from tqdm import tqdm
import numpy as np
import torch
import torch.optim as optim
from PIL import Image
from torchvision.transforms import Compose, RandomAffine, RandomCrop, RandomHorizontalFlip, Resize, ToPILImage, ToTensor

from Defenses.DefenseMethods.defenses import Defense
from RawModels.ResNet import adjust_learning_rate
from RawModels.Utils.TrainTest import train_one_epoch, validation_evaluation


def image_crop_rescale(sample, crop_size, color_mode):
    image = ToPILImage(mode=color_mode)(sample)
    cropped_image = RandomCrop(crop_size)(image)
    rescaled_image = Resize((sample.shape[1], sample.shape[2]), interpolation=0)(cropped_image)
    cropped_rescaled_sample = ToTensor()(rescaled_image)
    return cropped_rescaled_sample


def bit_depth_reduction(samples, depth):
    level = 2 ** depth
    reduced_images = np.rint(samples * (level - 1)) / (level - 1)
    return reduced_images


def total_variance_minimization(image, lambda_tv):
    from Defenses.DefenseMethods.External.InputTransformations import defend_tv
    image_numpy_channel = image.permute(1, 2, 0).numpy()
    tv_min_image = defend_tv(input_array=image_numpy_channel, lambda_tv=lambda_tv)
    return torch.from_numpy(tv_min_image).float().permute(2, 0, 1)


def jpeg_compress(image, quality, color_mode):
    from Defenses.DefenseMethods.External.InputTransformations import defend_jpeg
    return defend_jpeg(input_tensor=image, image_mode=color_mode, quality=quality)


# For help when requiring self-defined dataset provision with batches during training
class TransformedDataset(torch.utils.data.Dataset):
    def __init__(self, images, labels, dataset, transform):
        super(TransformedDataset, self).__init__()
        self.images = images
        self.labels = labels
        self.transform = transform
        self.color_mode = 'RGB' if dataset == 'CIFAR10' else 'L'

    def __getitem__(self, index):
        single_image, single_label = self.images[index], self.labels[index]
        if self.transform:
            img = ToPILImage(mode=self.color_mode)(single_image)
            single_image = self.transform(img)
        return single_image, single_label

    def __len__(self):
        return len(self.images)


class EITDefense(Defense):

    def __init__(self, model=None, defense_name=None, dataset=None, re_training=True, training_parameters=None, device=None, **kwargs):

        super(EITDefense, self).__init__(model=model, defense_name=defense_name)
        self.model = model
        self.defense_name = defense_name
        self.device = device

        self.Dataset = dataset.upper()
        assert self.Dataset in ['MNIST', 'CIFAR10'], "The data set must be MNIST or CIFAR10"

        # make sure to parse the parameters for the defense
        assert self._parsing_parameters(**kwargs)

        if re_training:
            # get the training_parameters, the same as the settings of RawModels
            self.num_epochs = training_parameters['num_epochs']
            self.batch_size = training_parameters['batch_size']

            # prepare the optimizers
            if self.Dataset == 'MNIST':
                self.optimizer = optim.SGD(self.model.parameters(), lr=training_parameters['learning_rate'],
                                           momentum=training_parameters['momentum'], weight_decay=training_parameters['decay'], nesterov=True)
            else:
                self.optimizer = optim.Adam(self.model.parameters(), lr=training_parameters['lr'])

        self.color_mode = 'RGB' if self.Dataset == 'CIFAR10' else 'L'
        if self.Dataset == 'CIFAR10':
            self.transform = Compose([
                RandomAffine(degrees=0, translate=(0.1, 0.1)),
                RandomHorizontalFlip(),
                ToTensor()
            ])
        else:
            self.transform = None

    def _parsing_parameters(self, **kwargs):

        assert kwargs is not None, "the parameters should be specified"
        print("\nUser configurations for the {} defense".format(self.defense_name))
        for key in kwargs:
            print('\t{} = {}'.format(key, kwargs[key]))

        self.bit_depth = kwargs['bit_depth']
        self.crop_size = kwargs['crop_size']
        self.JPEG_quality = kwargs['JPEG_quality']
        self.lambda_tv = kwargs['lambda_tv']
        return True

    def ensemble_input_transformations(self, images):
        transformed_batch_images_list = []
        for index in range(images.shape[0]):
            image = images[index]
            image = torch.from_numpy(image).to('cpu')
            # Image Crop and Rescaling
            cropped_rescaled_image = image_crop_rescale(sample=image, crop_size=self.crop_size, color_mode=self.color_mode)
            # Total Variance Minimization
            tv_mim_image = total_variance_minimization(image=cropped_rescaled_image, lambda_tv=self.lambda_tv)
            # JPEG Compression
            compressed_image = jpeg_compress(image=tv_mim_image, quality=self.JPEG_quality, color_mode=self.color_mode)
            transformed_batch_images_list.append(compressed_image.numpy())
        transformed_batch_images_numpy = np.array(transformed_batch_images_list)
        # bit depth for batch images
        transformed_batch_images_numpy = bit_depth_reduction(transformed_batch_images_numpy, depth=self.bit_depth)

        return transformed_batch_images_numpy

    def transforming_dataset(self, data_loader=None):
        transformed_data = []
        transformed_label = []

        print('\ntransforming dataset ....\n')

        for index, (images, labels) in enumerate(tqdm(data_loader)):
            np_images = images.cpu().numpy()
            np_labels = labels.cpu().numpy()

            transformed_image_numpy = self.ensemble_input_transformations(images=np_images)
            transformed_data.extend(transformed_image_numpy)
            transformed_label.extend(np_labels)

        return np.array(transformed_data), np.array(transformed_label)

    def defense(self, train_loader=None, valid_loader=None):

        transformed_train_data_numpy, transformed_train_label_numpy = self.transforming_dataset(train_loader)
        transformed_val_data_numpy, transformed_val_label_numpy = self.transforming_dataset(valid_loader)

        transformed_train_dataset = TransformedDataset(images=torch.from_numpy(transformed_train_data_numpy),
                                                       labels=torch.from_numpy(transformed_train_label_numpy), dataset=self.Dataset,
                                                       transform=self.transform)
        transformed_train_loader = torch.utils.data.DataLoader(transformed_train_dataset, batch_size=self.batch_size, shuffle=True)

        transformed_val_dataset = TransformedDataset(images=torch.from_numpy(transformed_val_data_numpy),
                                                     labels=torch.from_numpy(transformed_val_label_numpy), dataset=self.Dataset,
                                                     transform=None)
        transformed_val_loader = torch.utils.data.DataLoader(transformed_val_dataset, batch_size=self.batch_size, shuffle=False)

        best_val_acc = None
        for epoch in range(self.num_epochs):

            train_one_epoch(model=self.model, train_loader=transformed_train_loader, optimizer=self.optimizer, epoch=epoch, device=self.device)
            val_acc = validation_evaluation(model=self.model, validation_loader=transformed_val_loader, device=self.device)

            if self.Dataset == 'CIFAR10':
                adjust_learning_rate(epoch=epoch, optimizer=self.optimizer)

            # save the retrained defense-enhanced model
            assert os.path.exists('../DefenseEnhancedModels/{}'.format(self.defense_name))
            defense_enhanced_saver = '../DefenseEnhancedModels/{}/{}_{}_enhanced.pt'.format(self.defense_name, self.Dataset, self.defense_name)
            if not best_val_acc or round(val_acc, 4) >= round(best_val_acc, 4):
                if best_val_acc is not None:
                    os.remove(defense_enhanced_saver)
                best_val_acc = val_acc
                self.model.save(name=defense_enhanced_saver)
            else:
                print('Train Epoch{:>3}: validation dataset accuracy did not improve from {:.4f}\n'.format(epoch, best_val_acc))
