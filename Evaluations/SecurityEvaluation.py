#!/usr/bin/env python
# -*- coding: utf-8 -*-
# **************************************
# @Time    : 2018/11/7 22:27
# @Author  : Xiang Ling
# @Lab     : nesa.zju.edu.cn
# @File    : SecurityEvaluation.py 
# **************************************

import argparse
import os
import random
import sys
import warnings

import numpy as np

warnings.filterwarnings("ignore")

import torch

sys.path.append('%s/../' % os.path.dirname(os.path.realpath(__file__)))
from RawModels.MNISTConv import MNISTConvNet
from RawModels.ResNet import resnet20_cifar
from Attacks.AttackMethods.AttackUtils import predict


# help functions


class SecurityEvaluate:

    def __init__(self, DataSet='MNIST', AttackName='LLC', AdvExamplesDir='../AdversarialExampleDatasets/', device=torch.device('cpu')):
        """

        :param DataSet:
        :param AttackName:
        :param AdvExamplesDir:
        :param device:
        """
        self.device = device

        # check and set the support data set
        assert DataSet.upper() in ['MNIST', 'CIFAR10'], "The data set must be MNIST or CIFAR10"
        self.dataset = DataSet.upper()

        # load the raw model
        raw_model_location = '{}{}/model/{}_raw.pt'.format('../RawModels/', self.dataset, self.dataset)
        if self.dataset == 'MNIST':
            self.raw_model = MNISTConvNet().to(device)
            self.raw_model.load(path=raw_model_location, device=device)
        else:
            self.raw_model = resnet20_cifar().to(device)
            self.raw_model.load(path=raw_model_location, device=device)
        self.raw_model.eval()

        # check and set the supported attack name
        self.attack_name = AttackName.upper()
        supported_un_targeted = ['FGSM', 'RFGSM', 'BIM', 'PGD', 'UMIFGSM', 'DEEPFOOL', 'UAP', 'OM']
        supported_targeted = ['LLC', "RLLC", 'ILLC', 'JSMA', 'TMIFGSM', 'BLB', 'CW2', 'EAD']
        assert self.attack_name in supported_un_targeted or self.attack_name in supported_targeted, \
            "\nCurrently, our implementation support attacks of FGSM, RFGSM, BIM, UMIFGSM, DeepFool, LLC, RLLC, ILLC, TMIFGSM, JSMA, CW2,....\n"

        # set the Target (UA or TA) according to the AttackName
        if self.attack_name.upper() in supported_un_targeted:
            self.Targeted = False
            print('the # {} # attack is a kind of Un-targeted attacks'.format(self.attack_name))
        else:
            self.Targeted = True
            print('the # {} # attack is a kind of Targeted attacks'.format(self.attack_name))

        # load the adversarial examples / corresponding adversarial labels / corresponding true labels /
        self.adv_samples = np.load('{}{}/{}/{}_AdvExamples.npy'.format(AdvExamplesDir, self.attack_name, self.dataset, self.attack_name)).astype(
            np.float32)
        self.adv_labels = np.load('{}{}/{}/{}_AdvLabels.npy'.format(AdvExamplesDir, self.attack_name, self.dataset, self.attack_name))
        self.true_labels = np.load('{}{}/{}/{}_TrueLabels.npy'.format(AdvExamplesDir, self.attack_name, self.dataset, self.attack_name))

        # get the targets labels
        # prepare the targeted label (least likely label) for LLC RLLC and ILLC
        if self.attack_name.upper() in ['LLC', 'RLLC', 'ILLC']:
            self.targets_samples = np.load('{}{}/{}_llc.npy'.format('../CleanDatasets/', self.dataset, self.dataset))
        else:
            self.targets_samples = np.load('{}{}/{}_targets.npy'.format('../CleanDatasets/', self.dataset, self.dataset))

    def defense_predication(self, DefenseModelDirs, defense_name, **kwargs):
        """

        :param DefenseModelDirs:
        :param defense_name:
        :param kwargs:
        :return:
        """
        re_train_defenses = {'NAT', 'EAT', 'PAT', 'DD', 'IGR'}
        input_transformation_defenses = {'EIT', 'RT', 'PD', 'TE'}
        other_defenses = {'RC'}

        defense_name = defense_name.upper().strip()
        assert defense_name in re_train_defenses or input_transformation_defenses or other_defenses

        if defense_name in re_train_defenses:
            print('\n##{}## defense is a kind of complete defenses that retrain the model'.format(defense_name))
            # load the defense-enhanced model
            defended_model_location = '{}/{}/{}_{}_enhanced.pt'.format(DefenseModelDirs, defense_name, self.dataset, defense_name)
            defended_model = MNISTConvNet().to(self.device) if self.dataset == 'MNIST' else resnet20_cifar().to(self.device)
            defended_model.load(path=defended_model_location, device=self.device)
            defended_model.eval()

            predication = predict(model=defended_model, samples=self.adv_samples, device=self.device)
            labels = torch.argmax(predication, 1).cpu().numpy()
            return labels

        elif defense_name in input_transformation_defenses:
            print('\n##{}## defense is a kind of complete defense that need to transform the images ... '.format(defense_name))
            if defense_name == 'EIT':

                from Defenses.DefenseMethods.EIT import EITDefense
                eit_params = {
                    'crop_size': kwargs['crop_size'],
                    'lambda_tv': kwargs['lambda_tv'],
                    'JPEG_quality': kwargs['JPEG_quality'],
                    'bit_depth': kwargs['bit_depth']
                }
                defended_model = MNISTConvNet().to(self.device) if self.dataset == 'MNIST' else resnet20_cifar().to(self.device)
                defended_model_location = '{}/{}/{}_{}_enhanced.pt'.format('../DefenseEnhancedModels', defense_name, self.dataset, defense_name)
                defended_model = defended_model.to(self.device)
                defended_model.load(path=defended_model_location, device=self.device)
                defended_model.eval()

                EIT = EITDefense(model=defended_model, defense_name=defense_name, dataset=self.dataset, re_training=False,
                                 training_parameters=None, device=self.device, **eit_params)

                transformed_images = EIT.ensemble_input_transformations(images=self.adv_samples)
                predication = predict(model=defended_model, samples=transformed_images, device=self.device)
                labels = torch.argmax(predication, 1).cpu().numpy()
                return labels

            elif defense_name == 'RT':
                assert 'rt_resize' in kwargs
                final_size = kwargs['rt_resize']
                assert isinstance(final_size, int)
                warnings.warn(message='For the RT defense, the #resize# parameter is specified as {}, please check ...'.format(final_size))

                from Defenses.DefenseMethods.RT import RTDefense
                rt = RTDefense(model=self.raw_model, defense_name='RT', dataset=self.dataset, device=self.device)
                transformed_images = rt.randomization_transformation(samples=self.adv_samples, original_size=self.adv_samples.shape[-1],
                                                                     final_size=final_size)
                predication = predict(model=self.raw_model, samples=transformed_images, device=self.device)
                labels = torch.argmax(predication, 1).cpu().numpy()
                return labels

            elif defense_name == 'PD':
                assert 'pd_eps' in kwargs
                epsilon = kwargs['pd_eps']
                warnings.warn(
                    message='For the PixelDefend defense, the #epsilon# parameter is specified as {}, please check ...'.format(epsilon))
                from Defenses.DefenseMethods.PD import PixelDefend

                pd = PixelDefend(model=self.raw_model, defense_name='PD', dataset=self.dataset, pixel_cnn_dir='../Defenses/', device=self.device)
                purified_images = pd.de_noising_samples_batch(samples=self.adv_samples, batch_size=20, eps=epsilon)

                predication = predict(model=self.raw_model, samples=purified_images, device=self.device)
                labels = torch.argmax(predication, 1).cpu().numpy()
                return labels

            else:
                assert 'te_level' in kwargs
                level = kwargs['te_level']
                assert defense_name == 'TE' and isinstance(level, int)
                warnings.warn(message='For the TE defense, the #level# parameter is specified as {}, please check ...'.format(level))

                # load the defense-enhanced model (for TE)
                defended_model_location = '{}/{}/{}_{}_enhanced.pt'.format('../DefenseEnhancedModels', defense_name, self.dataset, defense_name)
                te_defended_model = MNISTConvNet(thermometer=True, level=level).to(self.device) if self.dataset == 'MNIST' \
                    else resnet20_cifar(thermometer=True, level=level).to(self.device)

                te_defended_model.load(path=defended_model_location, device=self.device)
                te_defended_model.eval()

                from Defenses.DefenseMethods.TE import thermometer_encoding
                therm_inputs = thermometer_encoding(samples=torch.from_numpy(self.adv_samples).to(self.device), level=level, device=self.device)

                predication = predict(model=te_defended_model, samples=therm_inputs, device=self.device)
                labels = torch.argmax(predication, 1).cpu().numpy()
                return labels
        else:
            if defense_name == 'RC':
                print('\n##{}## defense is a kind of region-based classification defenses ... '.format(defense_name))
                from Defenses.DefenseMethods.RC import RCDefense
                num_points = 1000

                assert 'rc_radius' in kwargs
                radius = kwargs['rc_radius']
                rc = RCDefense(model=self.raw_model, defense_name='RC', dataset=self.dataset, device=self.device, num_points=num_points)

                labels = rc.region_based_classification(samples=self.adv_samples, radius=radius)
                return labels
            else:
                raise ValueError('{} is not supported!!!'.format(defense_name))

    def success_rate(self, defense_predication):
        """

        :param defense_predication:
        :return:
        """
        true_labels = np.argmax(self.true_labels, 1)
        targets = np.argmax(self.targets_samples, 1)
        assert defense_predication.shape == true_labels.shape and true_labels.shape == self.adv_labels.shape and self.adv_labels.shape == targets.shape

        original_misclassification = 0.0
        defense_success = 0.0
        for i in range(len(defense_predication)):
            if self.Targeted:
                # successfully attack
                if self.adv_labels[i] == targets[i]:
                    original_misclassification += 1
                    # successfully defense
                    if defense_predication[i] == true_labels[i]:
                        defense_success += 1
            else:
                # successfully attack
                if self.adv_labels[i] != true_labels[i]:
                    original_misclassification += 1
                    # successfully defense
                    if defense_predication[i] == true_labels[i]:
                        defense_success += 1
        return original_misclassification, defense_success


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

    dataset = args.dataset.upper()
    assert dataset == 'MNIST' or dataset == 'CIFAR10'

    security_eval = SecurityEvaluate(DataSet=dataset, AttackName=args.attack, AdvExamplesDir='../AdversarialExampleDatasets/',
                                     device=device)

    defense_names = args.defenses.upper().split(',')

    params = {
        'crop_size': args.crop_size,
        'lambda_tv': args.lambda_tv,
        'JPEG_quality': args.JPEG_quality,
        'bit_depth': args.bit_depth,

        'rt_resize': args.resize,
        'pd_eps': args.epsilon,
        'te_level': args.level,
        'rc_radius': args.radius
    }

    print("\n****************************")
    print("The classification accuracy of adversarial examples ({}) w.r.t following defenses:".format(args.attack))
    for defense in defense_names:
        defense = defense.strip()
        preds = security_eval.defense_predication(DefenseModelDirs='../DefenseEnhancedModels', defense_name=defense, **params)
        original_misclassification, defense_success = security_eval.success_rate(preds)

        print('\tFor {} defense, accuracy={:.0f}/{:.0f}={:.1f}%\n'.format(defense, defense_success, original_misclassification,
                                                                          defense_success / original_misclassification * 100))
    print("****************************")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='CIFAR10', help='the dataset should be MNIST or CIFAR10')
    parser.add_argument('--gpu_index', type=str, default='0', help="gpu index to use")
    parser.add_argument('--seed', type=int, default=100, help='the default random seed for numpy and torch')

    parser.add_argument('--attack', type=str, default='FGSM', help='the attack name (only one)')
    parser.add_argument('--defenses', type=str, default='NAT, DD, RT, TE',
                        help="the defense methods to be evaluated (multiply defenses should be split by comma)")

    # default argument for the EIT defense (only crop_size is different)
    parser.add_argument('--crop_size', type=int, default=30, help='the cropping size (26 for mnist and 30 for cifar10)')
    parser.add_argument('--bit_depth', type=int, default=4, help='the quantization level of pixel value')
    parser.add_argument('--JPEG_quality', type=int, default=85, help='the JPEG quality to compress with')
    parser.add_argument('--lambda_tv', type=float, default=0.03, help='the total variance minimization weight')

    # default argument for the RT defense in MNIST (final_size=31) or in CIFAR10 (final_size=36)
    parser.add_argument('--resize', type=int, default=36, help='the final size parameter only for the RT defense')
    # default argument for the PD defense in MNIST (epsilon=76.5) or in CIFAR10 (epsilon=16)
    parser.add_argument('--epsilon', type=float, default=16, help="radius of e_ball")
    # default argument for the TE defense (level=16)
    parser.add_argument('--level', type=int, default=16, help='the discretization level of pixel value')
    # default argument for the RC defense in MNIST (radius=0.3) or in CIFAR10 (radius=0.02)
    parser.add_argument('--radius', type=float, default=0.02, help='in the case of not search radius r, we set the radius of the hypercube')

    arguments = parser.parse_args()
    main(arguments)
