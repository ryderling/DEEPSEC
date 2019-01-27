#!/usr/bin/env python
# -*- coding: utf-8 -*-
# **************************************
# @Time    : 2018/10/12 11:30
# @Author  : Xiang Ling
# @Lab     : nesa.zju.edu.cn
# @File    : TrainTest.py 
# **************************************


import torch
import torch.nn.functional as F


# help functions for training and testing

def model_weight_decay(model=None):
    decay = None
    for (name, param) in model.named_parameters():
        if name.lower().find('conv') > 0:
            if decay is None:
                decay = param.norm(2)
            else:
                decay = decay + param.norm(2)
    if decay is None:
        decay = 0
    return decay


# train the model in one epoch
def train_one_epoch(model, train_loader, optimizer, epoch, device):
    """

    :param model:
    :param train_loader:
    :param optimizer:
    :param epoch:
    :param device:
    :return:
    """

    # Sets the model in training mode
    model.train()
    for index, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)
        # forward the nn
        outputs = model(images)
        loss = F.cross_entropy(outputs, labels)

        # backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print('\rTrain Epoch{:>3}: [batch:{:>4}/{:>4}({:>3.0f}%)]  \tLoss: {:.4f} ===> '. \
              format(epoch, index, len(train_loader), index / len(train_loader) * 100.0, loss.item()), end=' ')


# evaluate the model using validation dataset
def validation_evaluation(model, validation_loader, device):
    """

    :param model:
    :param validation_loader:
    :param device:
    :return:
    """
    model = model.to(device)
    model.eval()

    total = 0.0
    correct = 0.0
    with torch.no_grad():
        for index, (inputs, labels) in enumerate(validation_loader):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total = total + labels.size(0)
            correct = correct + (predicted == labels).sum().item()
        ratio = correct / total
    print('validation dataset accuracy is {:.4f}'.format(ratio))
    return ratio


# evaluate the model using testing dataset
def testing_evaluation(model, test_loader, device):
    """

    :param model:
    :param test_loader:
    :param device:
    :return:
    """
    print('\n#####################################')
    print('#### The {} model is evaluated on the testing dataset loader ...... '.format(model.model_name))
    # Sets the module in evaluation mode.
    model = model.to(device)
    model.eval()

    total = 0.0
    correct = 0.0
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total = total + labels.size(0)
            correct = correct + (predicted == labels).sum().item()
        ratio = correct / total
    print('#### Accuracy of the loaded model on the testing dataset: {:.1f}/{:.1f} = {:.2f}%'.format(correct, total, ratio * 100))
    print('#####################################\n')

    return ratio
