#!/usr/bin/env python
# -*- coding: utf-8 -*-
# **************************************
# @Time    : 2018/9/7 23:05
# @Author  : Xiang Ling
# @Lab     : nesa.zju.edu.cn
# @File    : attacks.py
# **************************************

from abc import ABCMeta
from abc import abstractmethod


class Attack(object):
    __metaclass__ = ABCMeta

    def __init__(self, model):
        self.model = model

    @abstractmethod
    def perturbation(self):
        print("Abstract Method of Attacks is not implemented")
        raise NotImplementedError
