#!/usr/bin/env python
# -*- coding: utf-8 -*-
# **************************************
# @Time    : 2018/9/15 16:18
# @Author  : Xiang Ling
# @Lab     : nesa.zju.edu.cn
# @File    : CandidatesSelection.py 
# **************************************

import argparse
import os
import random
import shutil
import sys

import numpy as np
import torch

sys.path.append('%s/../' % os.path.dirname(os.path.realpath(__file__)))
from RawModels.MNISTConv import MNISTConvNet
from RawModels.ResNet import resnet20_cifar

from RawModels.Utils.dataset import get_cifar10_test_loader, get_mnist_test_loader


def main(args):
    # Device configuration
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_index
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Set the random seed manually for reproducibility.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    # prepare the dataset name, candidate num, dataset location and raw model location
    dataset = args.dataset.upper()
    num = args.number
    dataset_location = '../RawModels/{}/'.format(dataset)
    raw_model_location = '../RawModels/{}/model/{}_raw.pt'.format(dataset, dataset)
    print("\nStarting to select {} {} Candidates Example, which are correctly classified by the Raw Model from {}\n".format(num, dataset,
                                                                                                                            raw_model_location))

    # load the raw model and testing dataset
    assert args.dataset == 'MNIST' or args.dataset == 'CIFAR10'
    if dataset == 'MNIST':
        raw_model = MNISTConvNet().to(device)
        raw_model.load(path=raw_model_location, device=device)
        test_loader = get_mnist_test_loader(dir_name=dataset_location, batch_size=1, shuffle=False)
    else:
        raw_model = resnet20_cifar().to(device)
        raw_model.load(path=raw_model_location, device=device)
        test_loader = get_cifar10_test_loader(dir_name=dataset_location, batch_size=1, shuffle=False)

    # get the successfully classified examples

    successful = []
    raw_model.eval()

    with torch.no_grad():
        for image, label in test_loader:
            image = image.to(device)
            label = label.to(device)
            output = raw_model(image)
            _, predicted = torch.max(output.data, 1)
            if predicted == label:
                _, least_likely_class = torch.min(output.data, 1)
                successful.append([image, label, least_likely_class])

    print(len(successful))
    candidates = random.sample(successful, num)

    candidate_images = []
    candidate_labels = []
    candidates_llc = []
    candidate_targets = []

    for index in range(len(candidates)):
        image = candidates[index][0].cpu().numpy()
        image = np.squeeze(image, axis=0)
        candidate_images.append(image)

        label = candidates[index][1].cpu().numpy()[0]
        llc = candidates[index][2].cpu().numpy()[0]

        # selection for the targeted label
        classes = [i for i in range(10)]
        classes.remove(label)
        target = random.sample(classes, 1)[0]

        one_hot_label = [0 for i in range(10)]
        one_hot_label[label] = 1

        one_hot_llc = [0 for i in range(10)]
        one_hot_llc[llc] = 1

        one_hot_target = [0 for i in range(10)]
        one_hot_target[target] = 1

        candidate_labels.append(one_hot_label)
        candidates_llc.append(one_hot_llc)
        candidate_targets.append(one_hot_target)

    candidate_images = np.array(candidate_images)
    candidate_labels = np.array(candidate_labels)
    candidates_llc = np.array(candidates_llc)
    candidate_targets = np.array(candidate_targets)

    if dataset not in os.listdir('./'):
        os.mkdir('./{}/'.format(dataset))
    else:
        shutil.rmtree('{}'.format(dataset))
        os.mkdir('./{}/'.format(dataset))

    np.save('./{}/{}_inputs.npy'.format(dataset, dataset), candidate_images)
    np.save('./{}/{}_labels.npy'.format(dataset, dataset), candidate_labels)
    np.save('./{}/{}_llc.npy'.format(dataset, dataset), candidates_llc)
    np.save('./{}/{}_targets.npy'.format(dataset, dataset), candidate_targets)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Candidate Selection for Clean Data set')

    parser.add_argument('--dataset', type=str, default='CIFAR10', help='the dataset (MNIST or CIFAR10)')
    parser.add_argument('--seed', type=int, default=100, help='the default random seed for numpy and torch')
    parser.add_argument('--gpu_index', type=str, default='0', help="gpu index to use")

    parser.add_argument('--number', type=int, default=1000, help='the total number of candidate samples that will be randomly selected')

    arguments = parser.parse_args()
    main(arguments)
