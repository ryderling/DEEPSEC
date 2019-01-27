#!/usr/bin/env python
# -*- coding: utf-8 -*-
# **************************************
# @Time    : 2018/11/7 22:45
# @Author  : Xiang Ling
# @Lab     : nesa.zju.edu.cn
# @File    : dataset.py
# **************************************
import numpy as np

import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import Dataset
from torchvision.transforms import  ToPILImage

# help functions to get the training/validation/testing data loader

def get_mnist_train_validate_loader(dir_name, batch_size, valid_size=0.1, shuffle=True, random_seed=100, num_workers=1):
    """

    :param dir_name:
    :param batch_size:
    :param valid_size:
    :param shuffle:
    :param random_seed:
    :param num_workers:
    :return:
    """
    assert 0.0 <= valid_size <= 1.0, 'the size of validation set should be in the range of [0, 1]'

    train_mnist_dataset = torchvision.datasets.MNIST(root=dir_name, train=True, transform=transforms.ToTensor(), download=True)
    valid_mnist_dataset = torchvision.datasets.MNIST(root=dir_name, train=True, transform=transforms.ToTensor(), download=True)

    num_train = len(train_mnist_dataset)
    indices = list(range(num_train))

    split = int(np.floor(valid_size * num_train))

    if shuffle is True:
        np.random.seed(random_seed)
        np.random.shuffle(indices)

    train_idx, valid_idx = indices[split:], indices[:split]

    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)

    train_loader = torch.utils.data.DataLoader(train_mnist_dataset, batch_size=batch_size, sampler=train_sampler, num_workers=num_workers)
    valid_loader = torch.utils.data.DataLoader(valid_mnist_dataset, batch_size=batch_size, sampler=valid_sampler, num_workers=num_workers)

    return train_loader, valid_loader


def get_mnist_test_loader(dir_name, batch_size, shuffle=False, num_worker=1):
    """

    :param dir_name:
    :param batch_size:
    :param shuffle:
    :param num_worker:
    :return:
    """
    test_mnist_dataset = torchvision.datasets.MNIST(root=dir_name, train=False, download=True, transform=transforms.ToTensor())
    test_loader = torch.utils.data.DataLoader(dataset=test_mnist_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_worker)
    return test_loader


def get_cifar10_train_validate_loader(dir_name, batch_size, valid_size=0.1, augment=True, shuffle=True, random_seed=100, num_workers=1):
    """

    :param dir_name:
    :param batch_size:
    :param valid_size:
    :param augment:
    :param shuffle:
    :param random_seed:
    :param num_workers:
    :return:
    """
    # training dataset's transform
    if augment is True:
        train_transform = transforms.Compose([
            # transforms.RandomCrop(32),
            # transforms.RandomCrop(32, padding=4),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()
        ])
    else:
        train_transform = transforms.Compose([transforms.ToTensor()])

    # validation dataset's transform
    valid_transform = transforms.Compose([transforms.ToTensor()])

    # load the dataset
    train_cifar10_dataset = torchvision.datasets.CIFAR10(root=dir_name, train=True, download=True, transform=train_transform)
    valid_cifar10_dataset = torchvision.datasets.CIFAR10(root=dir_name, train=True, download=True, transform=valid_transform)

    num_train = len(train_cifar10_dataset)
    indices = list(range(num_train))

    split = int(np.floor(valid_size * num_train))

    if shuffle is True:
        np.random.seed(random_seed)
        np.random.shuffle(indices)

    train_idx, valid_idx = indices[split:], indices[:split]

    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)

    train_loader = torch.utils.data.DataLoader(train_cifar10_dataset, batch_size=batch_size, sampler=train_sampler, num_workers=num_workers)
    valid_loader = torch.utils.data.DataLoader(valid_cifar10_dataset, batch_size=batch_size, sampler=valid_sampler, num_workers=num_workers)

    return train_loader, valid_loader


def get_cifar10_test_loader(dir_name, batch_size, shuffle=False, num_worker=1):
    """

    :param dir_name:
    :param batch_size:
    :param shuffle:
    :param num_worker:
    :return:
    """
    test_cifar10_dataset = torchvision.datasets.CIFAR10(root=dir_name, train=False, download=True, transform=transforms.ToTensor())
    test_loader = torch.utils.data.DataLoader(dataset=test_cifar10_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_worker)
    return test_loader

# self defined dataset
class MyDataset(Dataset):

    def __init__(self, images, labels, dataset, transform):
        super(MyDataset, self).__init__()
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
