'''
# !/usr/bin/python3
# -*- coding: utf-8 -*-
@Time : 2022/5/13 14:57
@Author : Qiufen.Chen
@FileName: evaluation.py
@Software: PyCharm
'''

import torch.nn as nn
import torch as th
import scipy.stats


def PCC(y_pred, y_true):
    """
    :param y_pred: prediction value
    :param y_true: true value
    :return: Pearson's correlation coefficient
    """
    x = y_pred
    y = y_true
    vx = x - th.mean(x)
    vy = y - th.mean(y)
    pcc = th.sum(vx * vy) / (th.sqrt(th.sum(vx ** 2)) * th.sqrt(th.sum(vy ** 2)))
    return pcc


def R_Square(y_pred, y_true):
    """
    :param y_pred: prediction value
    :param y_true: true value
    :return: coefficient of determination
    """
    mean = th.mean(y_true)
    ss_tot = th.sum((y_true - mean) ** 2)
    ss_res = th.sum((y_true - y_pred) ** 2)
    r2 = 1 - ss_res / ss_tot
    return r2


def RMSE(y_pred, y_true):
    """
    :param y_pred: prediction value
    :param y_true: true value
    :return: Root Mean Square Error
    """
    mse = nn.MSELoss()
    return th.sqrt(mse(y_pred, y_true))


def MAE(y_pred, y_true):
    """
    :param y_pred: prediction value
    :param y_true: true value
    :return: Mean Absolute Error
    """
    mae = nn.L1Loss()
    return mae(y_pred, y_true)


def MSE(y_pred, y_true):
    """
    :param y_pred: prediction value
    :param y_true: true value
    :return: Mean Square Error
    """
    mse = nn.MSELoss()
    return mse(y_pred, y_true)

def Spearman(y_pred, y_true):
    res = scipy.stats.spearmanr(y_pred, y_true)[0]
    return res