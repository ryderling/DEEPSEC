#!/usr/bin/env python
# -*- coding: utf-8 -*-
# **************************************
# @Time    : 2018/10/21 21:00
# @Author  : Xiang Ling
# @Lab     : nesa.zju.edu.cn
# @File    : UAP_Generation.py
# **************************************


import argparse
import os
import random
import sys

import numpy as np
import torch

sys.path.append('%s/../' % os.path.dirname(os.path.realpath(__file__)))
from Attacks.AttackMethods.AttackUtils import predict
from Attacks.AttackMethods.UAP import UniversalAttack
from Attacks.Generation import Generation
from RawModels.Utils.dataset import get_cifar10_train_validate_loader, get_mnist_train_validate_loader


class UAPGeneration(Generation):
    def __init__(self, dataset, attack_name, targeted, raw_model_location, clean_data_location, adv_examples_dir, device, max_iter_uni, frate,
                 epsilon, overshoot, max_iter_df):
        super(UAPGeneration, self).__init__(dataset, attack_name, targeted, raw_model_location, clean_data_location, adv_examples_dir, device)

        self.max_iter_uni = max_iter_uni
        self.fooling_rate = frate
        self.epsilon = epsilon

        self.overshoot = overshoot
        self.max_iter_df = max_iter_df

    def generate(self):
        attacker = UniversalAttack(model=self.raw_model, fooling_rate=self.fooling_rate, max_iter_universal=self.max_iter_uni,
                                   epsilon=self.epsilon, overshoot=self.overshoot, max_iter_deepfool=self.max_iter_df)

        assert self.dataset.upper() == 'MNIST' or self.dataset.upper() == 'CIFAR10', "dataset should be MNIST or CIFAR10!"
        if self.dataset.upper() == 'MNIST':
            samples_loader, valid_loader = get_mnist_train_validate_loader(dir_name='../RawModels/MNIST/', batch_size=1, valid_size=0.9,
                                                                           shuffle=True)
        else:  # 'CIFAR10':
            samples_loader, valid_loader = get_cifar10_train_validate_loader(dir_name='../RawModels/CIFAR10/', batch_size=1, valid_size=0.9,
                                                                             augment=False, shuffle=True)

        universal_perturbation = attacker.universal_perturbation(dataset=samples_loader, validation=valid_loader, device=self.device)
        universal_perturbation = universal_perturbation.cpu().numpy()
        np.save('{}{}_{}_universal_perturbation'.format(self.adv_examples_dir, self.attack_name, self.dataset), universal_perturbation)

        adv_samples = attacker.perturbation(xs=self.nature_samples, uni_pert=universal_perturbation, device=self.device)

        adv_labels = predict(model=self.raw_model, samples=adv_samples, device=self.device)
        adv_labels = torch.max(adv_labels, 1)[1]
        adv_labels = adv_labels.cpu().numpy()

        np.save('{}{}_AdvExamples.npy'.format(self.adv_examples_dir, self.attack_name), adv_samples)
        np.save('{}{}_AdvLabels.npy'.format(self.adv_examples_dir, self.attack_name), adv_labels)
        np.save('{}{}_TrueLabels.npy'.format(self.adv_examples_dir, self.attack_name), self.labels_samples)

        mis = 0
        for i in range(len(adv_samples)):
            if self.labels_samples[i].argmax(axis=0) != adv_labels[i]:
                mis = mis + 1
        print('\nFor **{}** on **{}**: misclassification ratio is {}/{}={:.1f}%\n'.format(self.attack_name, self.dataset, mis, len(adv_samples),
                                                                                          mis / len(adv_labels) * 100))


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

    name = 'UAP'
    targeted = False

    df = UAPGeneration(dataset=args.dataset, attack_name=name, targeted=targeted, raw_model_location=args.modelDir,
                       clean_data_location=args.cleanDir, adv_examples_dir=args.adv_saver, device=device, max_iter_uni=args.max_iter_universal,
                       frate=args.fool_rate, epsilon=args.epsilon, overshoot=args.overshoot, max_iter_df=args.max_iter_deepfool)
    df.generate()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='The UAP Attack Generation')

    # common arguments
    parser.add_argument('--dataset', type=str, default='CIFAR10', help='the dataset should be MNIST or CIFAR10')
    parser.add_argument('--modelDir', type=str, default='../RawModels/', help='the directory for the raw model')
    parser.add_argument('--cleanDir', type=str, default='../CleanDatasets/', help='the directory for the clean dataset that will be attacked')
    parser.add_argument('--adv_saver', type=str, default='../AdversarialExampleDatasets/',
                        help='the directory used to save the generated adversarial examples')
    parser.add_argument('--seed', type=int, default=100, help='the default random seed for numpy and torch')
    parser.add_argument('--gpu_index', type=str, default='0', help="gpu index to use")

    # arguments for the particular attack
    parser.add_argument('--fool_rate', type=float, default=1.0, help="the fooling rate")
    parser.add_argument('--epsilon', type=float, default=0.1, help='controls the magnitude of the perturbation')
    parser.add_argument('--max_iter_universal', type=int, default=20, help="the maximum iterations for UAP")

    parser.add_argument('--overshoot', type=float, default=0.02, help='the overshoot parameter for DeepFool')
    parser.add_argument('--max_iter_deepfool', type=int, default=10, help='the maximum iterations for DeepFool')

    arguments = parser.parse_args()
    main(arguments)
