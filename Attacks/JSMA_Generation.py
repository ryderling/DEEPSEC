#!/usr/bin/env python
# -*- coding: utf-8 -*-
# **************************************
# @Time    : 2018/10/23 14:22
# @Author  : Jiannan Wang & Xiang Ling
# @Lab     : nesa.zju.edu.cn
# @File    : JSMA_Generation.py 
# **************************************

import argparse
import os
import random
import sys

import numpy as np
import torch

sys.path.append('%s/../' % os.path.dirname(os.path.realpath(__file__)))
from Attacks.AttackMethods.AttackUtils import predict
from Attacks.AttackMethods.JSMA import JSMAAttack
from Attacks.Generation import Generation


class JSMAGeneration(Generation):

    def __init__(self, dataset, attack_name, targeted, raw_model_location, clean_data_location, adv_examples_dir, device, theta, gamma):
        super(JSMAGeneration, self).__init__(dataset, attack_name, targeted, raw_model_location, clean_data_location, adv_examples_dir, device)

        self.theta = theta
        self.gamma = gamma

    def generate(self):
        attacker = JSMAAttack(model=self.raw_model, theta=self.theta, gamma=self.gamma)

        # get the targeted labels
        targets = np.argmax(self.targets_samples, axis=1)
        # generating
        adv_samples = attacker.perturbation(xs=self.nature_samples, ys_target=targets, device=self.device)

        adv_labels = predict(model=self.raw_model, samples=adv_samples, device=self.device)
        adv_labels = torch.max(adv_labels, 1)[1]
        adv_labels = adv_labels.cpu().numpy()

        np.save('{}{}_AdvExamples.npy'.format(self.adv_examples_dir, self.attack_name), adv_samples)
        np.save('{}{}_AdvLabels.npy'.format(self.adv_examples_dir, self.attack_name), adv_labels)
        np.save('{}{}_TrueLabels.npy'.format(self.adv_examples_dir, self.attack_name), self.labels_samples)

        mis_target = 0
        for i in range(len(adv_samples)):
            if targets[i] == adv_labels[i]:
                mis_target += 1
        print('\nFor **{}**(targeted attack) on **{}**, {}/{}={:.1f}% samples are misclassified as the specified targeted label\n'.format(
            self.attack_name, self.dataset, mis_target, len(adv_samples), mis_target / len(adv_samples) * 100.0))


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

    name = 'JSMA'
    targeted = True
    jsma = JSMAGeneration(dataset=args.dataset, attack_name=name, targeted=targeted, raw_model_location=args.modelDir,
                          clean_data_location=args.cleanDir, adv_examples_dir=args.adv_saver, device=device, theta=args.theta, gamma=args.gamma)
    jsma.generate()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='The JSMA Attack Generation')

    # common arguments
    parser.add_argument('--dataset', type=str, default='MNIST', help='the dataset should be MNIST or CIFAR10')
    parser.add_argument('--modelDir', type=str, default='../RawModels/', help='the directory for the raw model')
    parser.add_argument('--cleanDir', type=str, default='../CleanDatasets/', help='the directory for the clean dataset that will be attacked')
    parser.add_argument('--adv_saver', type=str, default='../AdversarialExampleDatasets/',
                        help='the directory used to save the generated adversarial examples')
    parser.add_argument('--seed', type=int, default=100, help='the default random seed for numpy and torch')
    parser.add_argument('--gpu_index', type=str, default='0', help="gpu index to use")

    # arguments for the particular attack
    parser.add_argument('--theta', type=float, default=1.0, help='theta')
    parser.add_argument('--gamma', type=float, default=0.1, help="gamma")

    arguments = parser.parse_args()
    main(arguments)
