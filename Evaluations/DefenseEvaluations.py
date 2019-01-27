#!/usr/bin/env python
# -*- coding: utf-8 -*-
# **************************************
# @Time    : 2018/9/17 1:23
# @Author  : Jiaxu Zou & Xiang Ling
# @Lab     : nesa.zju.edu.cn
# @File    : DefenseEvaluations.py
# **************************************

import argparse
import os
import random
import sys
import warnings

import numpy as np
import scipy.stats as jslib
import torch
import torch.nn.functional as F

sys.path.append('%s/../' % os.path.dirname(os.path.realpath(__file__)))
from RawModels.Utils.dataset import get_mnist_test_loader, get_cifar10_test_loader
from RawModels.ResNet import resnet20_cifar
from RawModels.MNISTConv import MNISTConvNet


# help functions
def pd_prediction(model, dataset, data_loader, epsilon, device):
    from Defenses.DefenseMethods.PD import PixelDefend
    pd = PixelDefend(model=model, defense_name='PD', dataset=dataset, pixel_cnn_dir='../Defenses/', device=device)
    model.eval()
    predicted_defended = []
    with torch.no_grad():
        for index, (images, labels) in enumerate(data_loader):
            images = images.detach().cpu().numpy()

            purified_images = pd.de_noising_samples(samples=images, batch_size=images.shape[0], eps=epsilon)

            rt_logits = model(torch.from_numpy(purified_images).to(device))
            rt_predicted = F.softmax(rt_logits, dim=1).cpu().numpy()
            predicted_defended.extend(rt_predicted)
        return np.array(predicted_defended)


def rt_prediction(model, dataset, data_loader, final_size, device):
    from Defenses.DefenseMethods.RT import RTDefense
    rt = RTDefense(model=model, defense_name='RT', dataset=dataset, device=device)
    model.eval()
    predicted_defended = []
    with torch.no_grad():
        for index, (images, labels) in enumerate(data_loader):
            transformed_images = rt.randomization_transformation(samples=images, original_size=images.shape[-1], final_size=final_size)
            transformed_images = transformed_images(device)

            rt_logits = model(transformed_images)
            rt_predicted = F.softmax(rt_logits, dim=1).cpu().numpy()
            predicted_defended.extend(rt_predicted)
    return np.array(predicted_defended)


def te_prediction(model, data_loader, level, device):
    from Defenses.DefenseMethods.TE import thermometer_encoding
    model.eval()
    predicted_defended = []
    with torch.no_grad():
        for index, (images, labels) in enumerate(data_loader):
            therm_inputs = thermometer_encoding(samples=images.to(device), level=level, device=device)
            therm_inputs = torch.from_numpy(therm_inputs).to(device)

            te_logits = model(therm_inputs)
            te_predicted = F.softmax(te_logits, dim=1).cpu().numpy()
            predicted_defended.extend(te_predicted)
    return np.array(predicted_defended)


def defense_utility_measure(pred_def, pred_raw, true_label):
    # compute the classification accuracy of raw model
    correct_prediction_raw = np.equal(np.argmax(pred_raw, axis=1), true_label)
    acc_raw = np.mean(correct_prediction_raw.astype(float))

    # compute the classification accuracy of defense-enhanced model
    correct_prediction_def = np.equal(np.argmax(pred_def, axis=1), true_label)
    acc_def = np.mean(correct_prediction_def.astype(float))

    # compute the Classification Accuracy Variance(CAV)
    cav_result = acc_def - acc_raw

    # find the index of correct predicted examples by defence-enhanced model and raw model
    idx_def = np.squeeze(np.argwhere(correct_prediction_def == True))
    idx_raw = np.squeeze(np.argwhere(correct_prediction_raw == True))
    idx = np.intersect1d(idx_def, idx_raw, assume_unique=True)

    # compute the Classification Rectify Ratio(CRR) & Classification Sacrifice Ratio(CSR)
    num_rectify = len(idx_def) - len(idx)
    crr_result = num_rectify / len(pred_def)

    num_sacrifice = len(idx_raw) - len(idx)
    csr_result = num_sacrifice / len(pred_def)

    # filter the correct prediction results
    pred_def_filter = pred_def[idx]
    pred_raw_filter = pred_raw[idx]

    # compute the Classification Confidence Variance(CCV)
    confidence_def = np.max(pred_def_filter, axis=1)
    confidence_raw = np.max(pred_raw_filter, axis=1)
    ccv_result = np.mean(np.absolute(confidence_def - confidence_raw))

    # compute the Classification Output Stability(COS)
    M = (pred_def_filter + pred_raw_filter) / 2.
    js_total = 0
    for i in range(len(M)):
        js = 0.5 * jslib.entropy(pred_def_filter[i], M[i]) + 0.5 * jslib.entropy(pred_raw_filter[i], M[i])
        js_total += js
    cos_result = js_total / len(M)

    return acc_raw, acc_def, cav_result, crr_result, csr_result, ccv_result, cos_result


def prediction(model, test_loader, device):
    print('\nThe #{}# model is evaluated on the testing dataset loader ...'.format(model.model_name))

    model = model.to(device)
    model.eval()

    prediction = []
    true_labels = []
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            logits = model(images)
            predicted = F.softmax(logits, dim=1).cpu().numpy()

            prediction.extend(predicted)
            true_labels.extend(labels)
    prediction = np.array(prediction)
    true_labels = np.array(true_labels)
    return prediction, true_labels


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

    # load the raw model / testing dataset loader
    raw_model_location = '{}{}/model/{}_raw.pt'.format('../RawModels/', dataset, dataset)
    if dataset == 'MNIST':
        raw_model = MNISTConvNet().to(device)
        raw_model.load(path=raw_model_location, device=device)
        test_loader = get_mnist_test_loader(dir_name='../RawModels/MNIST/', batch_size=30)
    else:
        raw_model = resnet20_cifar().to(device)
        raw_model.load(path=raw_model_location, device=device)
        test_loader = get_cifar10_test_loader(dir_name='../RawModels/CIFAR10/', batch_size=25)
    raw_model.eval()

    # get predictions of the raw model on test datasets
    predicted_raw, true_label = prediction(model=raw_model, test_loader=test_loader, device=device)

    re_train_defenses = {'NAT', 'EAT', 'PAT', 'DD', 'IGR'}
    input_transformation_defenses = {'EIT', 'RT', 'PD', 'TE'}
    other_defenses = {'RC'}

    defense_name = args.defense.upper().strip()
    if defense_name in re_train_defenses:
        print('\nthe ##{}## defense is a kind of complete defenses that retrain the model'.format(defense_name))
        # load the defense-enhanced model
        defended_model_location = '{}/{}/{}_{}_enhanced.pt'.format('../DefenseEnhancedModels', defense_name, dataset, defense_name)
        defended_model = MNISTConvNet().to(device) if dataset == 'MNIST' else resnet20_cifar().to(device)
        defended_model.load(path=defended_model_location, device=device)
        defended_model.eval()
        predicted_defended, _ = prediction(model=defended_model, test_loader=test_loader, device=device)
        raw_acc, def_acc, cav, crr, csr, ccv, cos = defense_utility_measure(predicted_defended, predicted_raw, true_label)

    elif defense_name in input_transformation_defenses:
        print('\nthe ##{}## defense is a kind of complete defense hat need to transform the images ... '.format(defense_name))
        if defense_name == 'EIT':

            from Defenses.DefenseMethods.EIT import EITDefense, TransformedDataset
            eit_params = {
                'crop_size': args.crop_size,
                'lambda_tv': args.lambda_tv,
                'JPEG_quality': args.JPEG_quality,
                'bit_depth': args.bit_depth
            }
            defended_model = MNISTConvNet().to(device) if dataset == 'MNIST' else resnet20_cifar().to(device)
            EIT = EITDefense(model=defended_model, defense_name=defense_name, dataset=dataset, re_training=False, training_parameters=None,
                             device=device, **eit_params)
            transformed_test_data_numpy, test_label_numpy = EIT.transforming_dataset(data_loader=test_loader)
            transformed_test_dataset = TransformedDataset(images=torch.from_numpy(transformed_test_data_numpy),
                                                          labels=torch.from_numpy(test_label_numpy), dataset=dataset, transform=None)
            transformed_test_loader = torch.utils.data.DataLoader(transformed_test_dataset, batch_size=100, shuffle=False)

            defended_model_location = '{}/{}/{}_{}_enhanced.pt'.format('../DefenseEnhancedModels', defense_name, dataset, defense_name)
            defended_model = defended_model.to(device)
            defended_model.load(path=defended_model_location, device=device)
            defended_model.eval()

            predicted_defended, _ = prediction(model=defended_model, test_loader=transformed_test_loader, device=device)
            raw_acc, def_acc, cav, crr, csr, ccv, cos = defense_utility_measure(predicted_defended, predicted_raw, true_label)

        elif defense_name == 'RT':
            final_size = args.resize
            assert isinstance(final_size, int)
            warnings.warn(message='For the RT defense, the #resize# parameter is specified as {}, please check ...'.format(final_size))
            predicted_defended = rt_prediction(model=raw_model, dataset=dataset, data_loader=test_loader, final_size=final_size, device=device)
            # test the utility performance of defended model
            raw_acc, def_acc, cav, crr, csr, ccv, cos = defense_utility_measure(predicted_defended, predicted_raw, true_label)
        elif defense_name == 'PD':
            epsilon = args.epsilon
            warnings.warn(message='For the PixelDefend defense, the #epsilon# parameter is specified as {}, please check ...'.format(epsilon))
            predicted_defended = pd_prediction(model=raw_model, dataset=dataset, data_loader=test_loader, epsilon=epsilon, device=device)
            raw_acc, def_acc, cav, crr, csr, ccv, cos = defense_utility_measure(predicted_defended, predicted_raw, true_label)
        else:
            level = args.level
            assert defense_name == 'TE' and isinstance(level, int)
            warnings.warn(message='For the TE defense, the #level# parameter is specified as {}, please check ...'.format(level))

            # load the defense-enhanced model (for TE)
            defended_model_location = '{}/{}/{}_{}_enhanced.pt'.format('../DefenseEnhancedModels', defense_name, dataset, defense_name)
            defended_model = MNISTConvNet(thermometer=True, level=level) if dataset == 'MNIST' else resnet20_cifar(thermometer=True, level=level)
            defended_model = defended_model.to(device)
            defended_model.load(path=defended_model_location, device=device)
            defended_model.eval()
            predicted_defended = te_prediction(model=defended_model, data_loader=test_loader, level=level, device=device)
            raw_acc, def_acc, cav, crr, csr, ccv, cos = defense_utility_measure(predicted_defended, predicted_raw, true_label)

    else:
        if defense_name == 'RC':
            print('\n##{}## defense is a kind of region-based classification defenses ... '.format(defense_name))
            from Defenses.DefenseMethods.RC import RCDefense
            num_points = 1000
            radius = args.radius
            rc = RCDefense(model=raw_model, defense_name='RC', dataset=dataset, device=device, num_points=num_points)

            predicted_defended = []
            with torch.no_grad():
                for index, (images, labels) in enumerate(test_loader):
                    rc_labels = rc.region_based_classification(samples=images, radius=radius)
                    predicted_defended.extend(rc_labels)
            predicted_defended = np.array(predicted_defended)

            # classification accuracy of defense-enhanced model
            correct_prediction_def = np.equal(predicted_defended, true_label)
            def_acc = np.mean(correct_prediction_def.astype(float))
            # classification accuracy of raw model
            correct_prediction_raw = np.equal(np.argmax(predicted_raw, axis=1), true_label)
            raw_acc = np.mean(correct_prediction_raw.astype(float))
            # Classification Accuracy Variance(CAV)
            cav = def_acc - raw_acc

            # Find the index of correct predicted examples by defence-enhanced model and raw model
            idx_def = np.squeeze(np.argwhere(correct_prediction_def == True))
            idx_raw = np.squeeze(np.argwhere(correct_prediction_raw == True))
            idx = np.intersect1d(idx_def, idx_raw, assume_unique=True)

            # Compute the Classification Rectify Ratio(CRR) & Classification Sacrifice Ratio(CSR)
            crr = (len(idx_def) - len(idx)) / len(predicted_raw)
            csr = (len(idx_raw) - len(idx)) / len(predicted_raw)
            ccv = cos = 0

        else:
            raise ValueError('{} is not supported!!!'.format(defense_name))

    print("****************************")
    print("The utility evaluation results of the {} defense for {} Dataset are as follow:".format(defense_name, dataset))
    print('Acc of Raw Model:\t\t{:.2f}%'.format(raw_acc * 100))
    print('Acc of {}-enhanced Model:\t{:.2f}%'.format(defense_name, def_acc * 100))
    print('CAV: {:.2f}%'.format(cav * 100))
    print('CRR: {:.2f}%'.format(crr * 100))
    print('CSR: {:.2f}%'.format(csr * 100))
    print('CCV: {:.2f}%'.format(ccv * 100))
    print('COS: {:.4f}'.format(cos))
    print("****************************")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='CIFAR10', help='the dataset should be MNIST or CIFAR10')
    parser.add_argument('--gpu_index', type=str, default='0', help="gpu index to use")
    parser.add_argument('--seed', type=int, default=100, help='the default random seed for numpy and torch')

    parser.add_argument('--defense', type=str, default='PD', help="the defense method to be evaluated")

    # default argument for the EIT defense (only crop_size is different)
    parser.add_argument('--crop_size', type=int, default=30, help='the cropping size (26 for mnist and 30 for cifar10)')
    parser.add_argument('--bit_depth', type=int, default=4, help='the quantization level of pixel value')
    parser.add_argument('--JPEG_quality', type=int, default=85, help='the JPEG quality to compress with')
    parser.add_argument('--lambda_tv', type=float, default=0.03, help='the total variance minimization weight')

    # default argument for the RT defense in MNIST (final_size=31) or in CIFAR10 (final_size=36)
    parser.add_argument('--resize', type=int, default=36, help='the final size parameter only for the RT defense')
    # default argument for the PD defense in MNIST (epsilon=76.5) or in CIFAR10 (epsilon=16)
    parser.add_argument('--epsilon', type=float, default=0.0627, help="radius of e_ball")
    # default argument for the TE defense (level=16)
    parser.add_argument('--level', type=int, default=16, help='the discretization level of pixel value')
    # default argument for the RC defense in MNIST (radius=0.3) or in CIFAR10 (radius=0.02)
    parser.add_argument('--radius', type=float, default=0.02, help='in the case of not search radius r, we set the radius of the hypercube')

    arguments = parser.parse_args()
    main(arguments)
