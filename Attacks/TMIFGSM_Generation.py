#!/usr/bin/env python
# -*- coding: utf-8 -*-
# **************************************
# @Time    : 2018/10/16 20:15
# @Author  : Xiang Ling
# @Lab     : nesa.zju.edu.cn
# @File    : TMIFGSM_Generation.py 
# **************************************

import argparse
import os
import random
import sys

import numpy as np
import torch

sys.path.append('%s/../' % os.path.dirname(os.path.realpath(__file__)))
from Attacks.Generation import Generation
from Attacks.AttackMethods.TMIFGSM import TMIFGSMAttack
from Attacks.AttackMethods.AttackUtils import predict


class TMIFGSMGeneration(Generation):

    def __init__(self, dataset, attack_name, targeted, raw_model_location, clean_data_location, adv_examples_dir, device, eps, eps_iter,
                 decay_factor, num_steps, attack_batch_size):
        super(TMIFGSMGeneration, self).__init__(dataset, attack_name, targeted, raw_model_location, clean_data_location, adv_examples_dir,
                                                device)
        self.attack_batch_size = attack_batch_size

        self.epsilon = eps
        self.epsilon_iter = eps_iter
        self.num_steps = num_steps
        self.decay_factor = decay_factor

    def generate(self):
        attacker = TMIFGSMAttack(model=self.raw_model, epsilon=self.epsilon, eps_iter=self.epsilon_iter, num_steps=self.num_steps,
                                 decay_factor=self.decay_factor)
        # get the targeted labels
        targets = np.argmax(self.targets_samples, axis=1)
        # generating
        adv_samples = attacker.batch_perturbation(xs=self.nature_samples, ys_target=targets, batch_size=self.attack_batch_size,
                                                  device=self.device)

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

    name = 'TMIFGSM'
    targeted = True
    tmifgsm = TMIFGSMGeneration(dataset=args.dataset, attack_name=name, targeted=targeted, raw_model_location=args.modelDir,
                                clean_data_location=args.cleanDir, adv_examples_dir=args.adv_saver, device=device,
                                eps=args.epsilon, attack_batch_size=args.attack_batch_size, eps_iter=args.epsilon_iter,
                                num_steps=args.num_steps, decay_factor=args.decay_factor)
    tmifgsm.generate()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='The TMIFGSM Attack Generation')

    # common arguments
    parser.add_argument('--dataset', type=str, default='CIFAR10', help='the dataset should be MNIST or CIFAR10')
    parser.add_argument('--modelDir', type=str, default='../RawModels/', help='the directory for the raw model')
    parser.add_argument('--cleanDir', type=str, default='../CleanDatasets/', help='the directory for the clean dataset that will be attacked')
    parser.add_argument('--adv_saver', type=str, default='../AdversarialExampleDatasets/',
                        help='the directory used to save the generated adversarial examples')
    parser.add_argument('--seed', type=int, default=100, help='the default random seed for numpy and torch')
    parser.add_argument('--gpu_index', type=str, default='0', help="gpu index to use")

    # arguments for the particular attack
    parser.add_argument('--epsilon', type=float, default=0.1, help='the max epsilon value that is allowed to be perturbed')
    parser.add_argument('--epsilon_iter', type=float, default=0.01, help='the one iterative eps of TMIFGSM')
    parser.add_argument('--num_steps', type=int, default=15, help='the number of perturbation steps')
    parser.add_argument('--decay_factor', type=float, default=1.0, help='the speed at which the momentum will decrease')
    parser.add_argument('--attack_batch_size', type=int, default=100, help='the default batch size for adversarial example generation')

    arguments = parser.parse_args()
    main(arguments)
