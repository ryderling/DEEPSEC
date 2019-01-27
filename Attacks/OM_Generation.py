#!/usr/bin/env python
# -*- coding: utf-8 -*-
# **************************************
# @Time    : 2018/10/28 19:32
# @Author  : Saizhuo Wang
# @Lab     : nesa.zju.edu.cn
# @File    : OM_Generation.py
# **************************************

import argparse
import os
import random
import sys

import numpy as np
import torch

sys.path.append('%s/../' % os.path.dirname(os.path.realpath(__file__)))
from Attacks.AttackMethods.AttackUtils import predict
from Attacks.AttackMethods.OptMargin import OMAttack
from Attacks.Generation import Generation


class OMGeneration(Generation):
    def __init__(self, dataset, attack_name, targeted, raw_model_location, clean_data_location, adv_examples_dir, device, attack_batch_size,
                 kappa, init_const, lr, binary_search_steps, max_iterations, lower_bound, upper_bound, noise_count, noise_mag):
        super(OMGeneration, self).__init__(dataset, attack_name, targeted, raw_model_location, clean_data_location, adv_examples_dir, device)
        self.attack_batch_size = attack_batch_size

        self.kappa = kappa
        self.init_const = init_const
        self.lr = lr
        self.binary_search_steps = binary_search_steps
        self.max_iter = max_iterations
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound

        self.noise_count = noise_count
        self.noise_mag = noise_mag

    def generate(self):
        attacker = OMAttack(model=self.raw_model, kappa=self.kappa, init_const=self.init_const, lr=self.lr,
                            binary_search_steps=self.binary_search_steps, max_iters=self.max_iter, lower_bound=self.lower_bound,
                            upper_bound=self.upper_bound, noise_count=self.noise_count, noise_magnitude=self.noise_mag)
        # get the true labels
        true_labels = np.argmax(self.labels_samples, axis=1)

        adv_samples = attacker.batch_perturbation(xs=self.nature_samples, ys=true_labels, batch_size=self.attack_batch_size, device=self.device)

        adv_labels = predict(model=self.raw_model, samples=adv_samples, device=self.device)
        adv_labels = torch.max(adv_labels, 1)[1]
        adv_labels = adv_labels.cpu().numpy()

        np.save('{}{}_AdvExamples.npy'.format(self.adv_examples_dir, self.attack_name), adv_samples)
        np.save('{}{}_AdvLabels.npy'.format(self.adv_examples_dir, self.attack_name), adv_labels)
        np.save('{}{}_TrueLabels.npy'.format(self.adv_examples_dir, self.attack_name), self.labels_samples)

        mis = 0
        for i in range(len(adv_samples)):
            if true_labels[i] != adv_labels[i]:
                mis += 1
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

    name = 'OM'
    targeted = False
    om = OMGeneration(dataset=args.dataset, attack_name=name, targeted=targeted, raw_model_location=args.modelDir,
                      clean_data_location=args.cleanDir, adv_examples_dir=args.adv_saver, device=device,
                      attack_batch_size=args.attack_batch_size, kappa=args.confidence, init_const=args.initial_const,
                      binary_search_steps=args.search_steps, lr=args.learning_rate, lower_bound=args.lower_bound,
                      upper_bound=args.upper_bound, max_iterations=args.iteration, noise_count=args.noise_count,
                      noise_mag=args.noise_mag)
    om.generate()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='The OM Attack Generation')

    # common arguments
    parser.add_argument('--dataset', type=str, default='MNIST', help='the dataset should be MNIST or CIFAR10')
    parser.add_argument('--modelDir', type=str, default='../RawModels/', help='the directory for the raw model')
    parser.add_argument('--cleanDir', type=str, default='../CleanDatasets/', help='the directory for the clean dataset that will be attacked')
    parser.add_argument('--adv_saver', type=str, default='../AdversarialExampleDatasets/',
                        help='the directory used to save the generated adversarial examples')

    parser.add_argument('--seed', type=int, default=100, help='the default random seed for numpy and torch')
    parser.add_argument('--gpu_index', type=str, default='0', help="gpu index to use")

    # arguments for the particular attack
    parser.add_argument('--confidence', type=float, default=0, help='the confidence of adversarial examples')
    parser.add_argument('--initial_const', type=float, default=0.02, help="the initial value of c in the binary search.")
    parser.add_argument('--learning_rate', type=float, default=0.2, help="the learning rate of gradient descent.")
    parser.add_argument('--iteration', type=int, default=10000, help='maximum iteration')
    parser.add_argument('--lower_bound', type=float, default=0.0, help='the minimum pixel value for examples (default=0.0).')
    parser.add_argument('--upper_bound', type=float, default=1.0, help='the maximum pixel value for examples (default=1.0).')
    parser.add_argument('--search_steps', type=int, default=4, help="The binary search steps to find the optimal const.")

    parser.add_argument('--noise_count', type=int, default=20, help="The number of directions in noises.")
    parser.add_argument('--noise_mag', type=float, default=0.3, help="The magnitude of the noise.")

    parser.add_argument('--attack_batch_size', type=int, default=100, help='the default batch size for adversarial example generation')

    arguments = parser.parse_args()
    main(arguments)
