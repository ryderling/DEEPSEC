#!/usr/bin/env python
# -*- coding: utf-8 -*-
# **************************************
# @Time    : 2018/10/24 18:52
# @Author  : Xiang Ling & Jiannan Wang
# @Lab     : nesa.zju.edu.cn
# @File    : AttackEval.py
# **************************************

import argparse
import os
import random
import shutil
import sys
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import torch
from PIL import Image, ImageFilter
from skimage.measure import compare_ssim as SSIM

sys.path.append('%s/../' % os.path.dirname(os.path.realpath(__file__)))

from Attacks.AttackMethods.AttackUtils import predict
from RawModels.MNISTConv import MNISTConvNet
from RawModels.ResNet import resnet20_cifar


# help function for the Gaussian Blur transformation of images
def gaussian_blur_transform(AdvSample, radius, oriDataset):
    if oriDataset.upper() == 'CIFAR10':
        assert AdvSample.shape == (3, 32, 32)
        sample = np.transpose(np.round(AdvSample * 255), (1, 2, 0))

        image = Image.fromarray(np.uint8(sample))
        gb_image = image.filter(ImageFilter.GaussianBlur(radius=radius))
        gb_image = np.transpose(np.array(gb_image), (2, 0, 1)).astype('float32') / 255.0
        return gb_image

    if oriDataset.upper() == 'MNIST':
        assert AdvSample.shape == (1, 28, 28)
        sample = np.transpose(np.round(AdvSample * 255), (1, 2, 0))
        # for MNIST, there is no RGB
        sample = np.squeeze(sample, axis=2)

        image = Image.fromarray(np.uint8(sample))
        gb_image = image.filter(ImageFilter.GaussianBlur(radius=radius))

        gb_image = np.expand_dims(np.array(gb_image).astype('float32'), axis=0) / 255.0
        return gb_image


# help function for the image compression transformation of images
def image_compress_transform(IndexAdv, AdvSample, dir_name, quality, oriDataset):
    if oriDataset.upper() == 'CIFAR10':
        assert AdvSample.shape == (3, 32, 32)
        sample = np.transpose(np.round(AdvSample * 255), (1, 2, 0))
        image = Image.fromarray(np.uint8(sample))

        saved_adv_image_path = os.path.join(dir_name, '{}th-adv-cifar.png'.format(IndexAdv))
        image.save(saved_adv_image_path)
        output_IC_path = os.path.join(dir_name, '{}th-IC-adv-cifar.jpg'.format(IndexAdv))

        cmd = 'guetzli --quality {} {} {}'.format(quality, saved_adv_image_path, output_IC_path)
        assert os.system(cmd) == 0, 'guetzli tool should be install before, https://github.com/google/guetzli'

        IC_image = Image.open(output_IC_path).convert('RGB')
        IC_image = np.transpose(np.array(IC_image), (2, 0, 1)).astype('float32') / 255.0
        return IC_image

    if oriDataset.upper() == 'MNIST':
        assert AdvSample.shape == (1, 28, 28)
        sample = np.transpose(np.round(AdvSample * 255), (1, 2, 0))
        sample = np.squeeze(sample, axis=2)  # for MNIST, there is no RGB
        image = Image.fromarray(np.uint8(sample), mode='L')

        saved_adv_image_path = os.path.join(dir_name, '{}th-adv-mnist.png'.format(IndexAdv))
        image.save(saved_adv_image_path)
        output_IC_path = os.path.join(dir_name, '{}th-IC-adv-mnist.jpg'.format(IndexAdv))

        cmd = 'guetzli --quality {} {} {}'.format(quality, saved_adv_image_path, output_IC_path)
        assert os.system(cmd) == 0, 'guetzli tool should be install before, https://github.com/google/guetzli'

        IC_image = Image.open(output_IC_path).convert('L')
        IC_image = np.expand_dims(np.array(IC_image).astype('float32'), axis=0) / 255.0
        return IC_image


class AttackEvaluate:

    def __init__(self, DataSet='MNIST', AttackName='FGSM', RawModelLocation='../RawModels/', CleanDataLocation='../CleanDatasets/',
                 AdvExamplesDir='../AdversarialExampleDatasets/', device=torch.device('cpu')):

        self.device = device

        # check and set the support data set
        assert DataSet.upper() in ['MNIST', 'CIFAR10'], "The data set must be MNIST or CIFAR10"
        self.dataset = DataSet.upper()
        self.color_mode = 'RGB' if self.dataset == 'CIFAR10' else 'L'

        # check and set the supported attack name
        self.attack_name = AttackName.upper()
        supported_un_targeted = ['FGSM', 'RFGSM', 'BIM', 'PGD', 'UMIFGSM', 'DEEPFOOL', 'UAP', 'OM']
        supported_targeted = ['LLC', "RLLC", 'ILLC', 'JSMA', 'TMIFGSM', 'BLB', 'CW2', 'EAD']
        assert self.attack_name in supported_un_targeted or self.attack_name in supported_targeted, \
            "\nCurrently, our implementation support attacks of FGSM, RFGSM, BIM, UMIFGSM, DeepFool, LLC, RLLC, ILLC, TMIFGSM, JSMA, CW2,....\n"
        # set the Target (UA or TA) according to the AttackName
        if self.attack_name.upper() in supported_un_targeted:
            self.Targeted = False
        else:
            self.Targeted = True

        # load the raw model
        raw_model_location = '{}{}/model/{}_raw.pt'.format(RawModelLocation, self.dataset, self.dataset)
        if self.dataset == 'MNIST':
            self.raw_model = MNISTConvNet().to(device)
            self.raw_model.load(path=raw_model_location, device=device)
        else:
            self.raw_model = resnet20_cifar().to(device)
            self.raw_model.load(path=raw_model_location, device=device)

        # get the clean datasets / true_labels
        self.nature_samples = np.load('{}{}/{}_inputs.npy'.format(CleanDataLocation, self.dataset, self.dataset))
        self.labels_samples = np.load('{}{}/{}_labels.npy'.format(CleanDataLocation, self.dataset, self.dataset))

        # get the targets labels
        # prepare the targeted label (least likely label) for LLC RLLC and ILLC
        if self.attack_name.upper() in ['LLC', 'RLLC', 'ILLC']:
            self.targets_samples = np.load('{}{}/{}_llc.npy'.format(CleanDataLocation, self.dataset, self.dataset))
        else:
            self.targets_samples = np.load('{}{}/{}_targets.npy'.format(CleanDataLocation, self.dataset, self.dataset))

        # get the adversarial examples
        self.AdvExamplesDir = AdvExamplesDir + self.attack_name + '/' + self.dataset + '/'
        if os.path.exists(self.AdvExamplesDir) is False:
            print("the directory of {} is not existing, please check carefully".format(self.AdvExamplesDir))
        self.adv_samples = np.load('{}{}_AdvExamples.npy'.format(self.AdvExamplesDir, self.attack_name))
        # self.adv_labels = np.load('{}{}_AdvLabels.npy'.format(self.AdvExamplesDir, self.AttackName))

        predictions = predict(model=self.raw_model, samples=self.adv_samples, device=self.device).detach().cpu().numpy()

        def soft_max(x):
            return np.exp(x) / np.sum(np.exp(x), axis=0)

        tmp_soft_max = []
        for i in range(len(predictions)):
            tmp_soft_max.append(soft_max(predictions[i]))

        self.softmax_prediction = np.array(tmp_soft_max)

    # help function
    def successful(self, adv_softmax_preds, nature_true_preds, targeted_preds, target_flag):
        """

        :param adv_softmax_preds: the softmax prediction for the adversarial example
        :param nature_true_preds: for the un-targeted attack, it should be the true label for the nature example
        :param targeted_preds: for the targeted attack, it should be the specified targets label that selected
        :param target_flag: True if it is a targeted attack, False if it is a un-targeted attack
        :return:
        """

        if target_flag:
            if np.argmax(adv_softmax_preds) == np.argmax(targeted_preds):
                return True
            else:
                return False
        else:
            if np.argmax(adv_softmax_preds) != np.argmax(nature_true_preds):
                return True
            else:
                return False

    # 1 MR:Misclassification Rate
    def misclassification_rate(self):

        cnt = 0
        for i in range(len(self.adv_samples)):
            if self.successful(adv_softmax_preds=self.softmax_prediction[i], nature_true_preds=self.labels_samples[i],
                               targeted_preds=self.targets_samples[i], target_flag=self.Targeted):
                cnt += 1
        mr = cnt / len(self.adv_samples)
        print('MR:\t\t{:.1f}%'.format(mr * 100))
        return mr

    # 2 ACAC: average confidence of adversarial class
    def avg_confidence_adv_class(self):
        cnt = 0
        conf = 0
        for i in range(len(self.adv_samples)):
            if self.successful(adv_softmax_preds=self.softmax_prediction[i], nature_true_preds=self.labels_samples[i],
                               targeted_preds=self.targets_samples[i], target_flag=self.Targeted):
                cnt += 1
                conf += np.max(self.softmax_prediction[i])

        print('ACAC:\t{:.3f}'.format(conf / cnt))
        return conf / cnt

    # 3 ACTC: average confidence of true class
    def avg_confidence_true_class(self):

        true_labels = np.argmax(self.labels_samples, axis=1)
        cnt = 0
        true_conf = 0
        for i in range(len(self.adv_samples)):
            if self.successful(adv_softmax_preds=self.softmax_prediction[i], nature_true_preds=self.labels_samples[i],
                               targeted_preds=self.targets_samples[i], target_flag=self.Targeted):
                cnt += 1
                true_conf += self.softmax_prediction[i, true_labels[i]]
        print('ACTC:\t{:.3f}'.format(true_conf / cnt))
        return true_conf / cnt

    # 4 ALP: Average L_p Distortion
    def avg_lp_distortion(self):

        ori_r = np.round(self.nature_samples * 255)
        adv_r = np.round(self.adv_samples * 255)

        NUM_PIXEL = int(np.prod(self.nature_samples.shape[1:]))

        pert = adv_r - ori_r

        dist_l0 = 0
        dist_l2 = 0
        dist_li = 0

        cnt = 0

        for i in range(len(self.adv_samples)):
            if self.successful(adv_softmax_preds=self.softmax_prediction[i], nature_true_preds=self.labels_samples[i],
                               targeted_preds=self.targets_samples[i], target_flag=self.Targeted):
                cnt += 1
                dist_l0 += (np.linalg.norm(np.reshape(pert[i], -1), ord=0) / NUM_PIXEL)
                dist_l2 += np.linalg.norm(np.reshape(self.nature_samples[i] - self.adv_samples[i], -1), ord=2)
                dist_li += np.linalg.norm(np.reshape(self.nature_samples[i] - self.adv_samples[i], -1), ord=np.inf)

        adv_l0 = dist_l0 / cnt
        adv_l2 = dist_l2 / cnt
        adv_li = dist_li / cnt

        print('**ALP:**\n\tL0:\t{:.3f}\n\tL2:\t{:.3f}\n\tLi:\t{:.3f}'.format(adv_l0, adv_l2, adv_li))
        return adv_l0, adv_l2, adv_li

    # 4 ASS: Average Structural Similarity
    def avg_SSIM(self):

        ori_r_channel = np.transpose(np.round(self.nature_samples * 255), (0, 2, 3, 1)).astype(dtype=np.float32)
        adv_r_channel = np.transpose(np.round(self.adv_samples * 255), (0, 2, 3, 1)).astype(dtype=np.float32)

        totalSSIM = 0
        cnt = 0

        """
        For SSIM function in skimage: http://scikit-image.org/docs/dev/api/skimage.measure.html

        multichannel : bool, optional If True, treat the last dimension of the array as channels. Similarity calculations are done 
        independently for each channel then averaged.
        """
        for i in range(len(self.adv_samples)):
            if self.successful(adv_softmax_preds=self.softmax_prediction[i], nature_true_preds=self.labels_samples[i],
                               targeted_preds=self.targets_samples[i], target_flag=self.Targeted):
                cnt += 1
                totalSSIM += SSIM(X=ori_r_channel[i], Y=adv_r_channel[i], multichannel=True)

        print('ASS:\t{:.3f}'.format(totalSSIM / cnt))
        return totalSSIM / cnt

    # 6: PSD: Perturbation Sensitivity Distance
    def avg_PSD(self):

        psd = 0
        cnt = 0

        for outer in range(len(self.adv_samples)):
            if self.successful(adv_softmax_preds=self.softmax_prediction[outer], nature_true_preds=self.labels_samples[outer],
                               targeted_preds=self.targets_samples[outer], target_flag=self.Targeted):
                cnt += 1

                image = self.nature_samples[outer]
                pert = abs(self.adv_samples[outer] - self.nature_samples[outer])

                for idx_channel in range(image.shape[0]):
                    image_channel = image[idx_channel]
                    pert_channel = pert[idx_channel]

                    image_channel = np.pad(image_channel, 1, 'reflect')
                    pert_channel = np.pad(pert_channel, 1, 'reflect')

                    for i in range(1, image_channel.shape[0] - 1):
                        for j in range(1, image_channel.shape[1] - 1):
                            psd += pert_channel[i, j] * (1.0 - np.std(np.array(
                                [image_channel[i - 1, j - 1], image_channel[i - 1, j], image_channel[i - 1, j + 1], image_channel[i, j - 1],
                                 image_channel[i, j], image_channel[i, j + 1], image_channel[i + 1, j - 1], image_channel[i + 1, j],
                                 image_channel[i + 1, j + 1]])))
        print('PSD:\t{:.3f}'.format(psd / cnt))
        return psd / cnt

    # 7 NTE: Noise Tolerance Estimation
    def avg_noise_tolerance_estimation(self):

        nte = 0
        cnt = 0
        for i in range(len(self.adv_samples)):
            if self.successful(adv_softmax_preds=self.softmax_prediction[i], nature_true_preds=self.labels_samples[i],
                               targeted_preds=self.targets_samples[i], target_flag=self.Targeted):
                cnt += 1
                sort_preds = np.sort(self.softmax_prediction[i])
                nte += sort_preds[-1] - sort_preds[-2]

        print('NTE:\t{:.3f}'.format(nte / cnt))
        return nte / cnt

    # 8 RGB: Robustness to Gaussian Blur
    def robust_gaussian_blur(self, radius=0.5):

        total = 0
        num_gb = 0

        if self.Targeted is True:
            for i in range(len(self.adv_samples)):
                if np.argmax(self.softmax_prediction[i]) == np.argmax(self.targets_samples[i]):
                    total += 1
                    adv_sample = self.adv_samples[i]
                    gb_sample = gaussian_blur_transform(AdvSample=adv_sample, radius=radius, oriDataset=self.dataset)
                    gb_pred = predict(model=self.raw_model, samples=np.array([gb_sample]), device=self.device).detach().cpu().numpy()
                    if np.argmax(gb_pred) == np.argmax(self.targets_samples[i]):
                        num_gb += 1

        else:
            for i in range(len(self.adv_samples)):
                if np.argmax(self.softmax_prediction[i]) != np.argmax(self.labels_samples[i]):
                    total += 1
                    adv_sample = self.adv_samples[i]
                    gb_sample = gaussian_blur_transform(AdvSample=adv_sample, radius=radius, oriDataset=self.dataset)
                    gb_pred = predict(model=self.raw_model, samples=np.array([gb_sample]), device=self.device).detach().cpu().numpy()
                    if np.argmax(gb_pred) != np.argmax(self.labels_samples[i]):
                        num_gb += 1

        print('RGB:\t{:.3f}'.format(num_gb / total))
        return num_gb, total, num_gb / total

    # 9 RIC: Robustness to Image Compression
    def robust_image_compression(self, quality):

        total = 0
        num_ic = 0

        # prepare the save dir for the generated image(png or jpg)
        image_save = os.path.join(self.AdvExamplesDir, 'image')
        if os.path.exists(image_save):
            shutil.rmtree(image_save)
        os.mkdir(image_save)
        print('\nNow, all adversarial examples are saved as PNG and then compressed using *Guetzli* in the {} fold ......\n'.format(image_save))

        if self.Targeted is True:
            for i in range(len(self.adv_samples)):
                if np.argmax(self.softmax_prediction[i]) == np.argmax(self.targets_samples[i]):
                    total += 1
                    adv_sample = self.adv_samples[i]

                    ic_sample = image_compress_transform(IndexAdv=i, AdvSample=adv_sample, dir_name=image_save, quality=quality,
                                                         oriDataset=self.dataset)

                    ic_pred = predict(model=self.raw_model, samples=np.array([ic_sample]), device=self.device).detach().cpu().numpy()
                    if np.argmax(ic_pred) == np.argmax(self.targets_samples[i]):
                        num_ic += 1

        else:
            for i in range(len(self.adv_samples)):
                if np.argmax(self.softmax_prediction[i]) != np.argmax(self.labels_samples[i]):
                    total += 1
                    adv_sample = self.adv_samples[i]

                    ic_sample = image_compress_transform(IndexAdv=i, AdvSample=adv_sample, dir_name=image_save, quality=quality,
                                                         oriDataset=self.dataset)

                    ic_pred = predict(model=self.raw_model, samples=np.array([ic_sample]), device=self.device).detach().cpu().numpy()
                    if np.argmax(ic_pred) != np.argmax(self.labels_samples[i]):
                        num_ic += 1
        print('RIC:\t{:.3f}'.format(num_ic / total))
        return num_ic, total, num_ic / total


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

    # Get training parameters, set up model frames and then get the train_loader and test_loader
    dataset = args.dataset.upper()
    assert dataset == 'MNIST' or dataset == 'CIFAR10'

    attack = AttackEvaluate(DataSet=dataset, AttackName=args.attack, RawModelLocation=args.modelDir, CleanDataLocation=args.cleanDir,
                            AdvExamplesDir=args.adv_saver, device=device)

    attack.raw_model.eval()

    attack.misclassification_rate()
    attack.avg_confidence_adv_class()
    attack.avg_confidence_true_class()
    attack.avg_lp_distortion()
    attack.avg_SSIM()
    attack.avg_PSD()
    attack.avg_noise_tolerance_estimation()
    attack.robust_gaussian_blur(radius=0.5)
    attack.robust_image_compression(quality=90)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluation of Adversarial Attacks')
    # common arguments
    parser.add_argument('--dataset', type=str, default='CIFAR10', help='the dataset should be MNIST or CIFAR10')
    parser.add_argument('--seed', type=int, default=100, help='the default random seed for numpy and torch')
    parser.add_argument('--gpu_index', type=str, default='0', help="gpu index to use")

    parser.add_argument('--modelDir', type=str, default='../RawModels/', help='the directory for the raw model')
    parser.add_argument('--cleanDir', type=str, default='../CleanDatasets/', help='the directory for the clean dataset that will be attacked')
    parser.add_argument('--adv_saver', type=str, default='../AdversarialExampleDatasets/',
                        help='the directory used to save the generated adversarial examples')

    parser.add_argument('--attack', type=str, default='CW2', help='the attack name')

    arguments = parser.parse_args()
    main(arguments)
