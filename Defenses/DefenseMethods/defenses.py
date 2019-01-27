#!/usr/bin/env python
# -*- coding: utf-8 -*-
# **************************************
# @Time    : 2018/9/9 15:52
# @Author  : Xiang Ling
# @Lab     : nesa.zju.edu.cn
# @File    : defenses.py
# **************************************

import os
from abc import ABCMeta
from abc import abstractmethod


class Defense(object):
    __metaclass__ = ABCMeta

    def __init__(self, model=None, defense_name=None):
        self.model = model
        self.defense_name = defense_name

        defense_dir = '../DefenseEnhancedModels/{}'.format(self.defense_name)
        if self.defense_name not in os.listdir('../DefenseEnhancedModels/'):
            os.mkdir(defense_dir)
            print('creating the {} folder for storing the {} defense'.format(defense_dir, self.defense_name))
        else:
            print('the storing {} folder is already existing'.format(defense_dir))

    @abstractmethod
    def defense(self):
        print("abstract method of 'Defenses' is not implemented")
        raise NotImplementedError
