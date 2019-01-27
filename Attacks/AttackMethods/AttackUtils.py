#!/usr/bin/env python
# -*- coding: utf-8 -*-
# **************************************
# @Time    : 2018/9/7 23:05
# @Author  : Xiang Ling
# @Lab     : nesa.zju.edu.cn
# @File    : AttackUtils.py
# **************************************



import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.autograd import Variable


def tensor2variable(x=None, device=None, requires_grad=False):
    """

    :param x:
    :param device:
    :param requires_grad:
    :return:
    """
    x = x.to(device)
    return Variable(x, requires_grad=requires_grad)


def predict(model=None, samples=None, device=None):
    """

    :param model:
    :param samples:
    :param device:
    :return:
    """
    model.eval()
    model = model.to(device)
    copy_samples = np.copy(samples)
    var_samples = tensor2variable(torch.from_numpy(copy_samples), device=device, requires_grad=True)
    predictions = model(var_samples.float())
    return predictions
