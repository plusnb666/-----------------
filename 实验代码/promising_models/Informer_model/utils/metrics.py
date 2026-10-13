import numpy as np


"""Informer 常用预测评价指标。

所有函数都接收 ``pred`` 和 ``true``，支持 NumPy 数组。MAPE/MSPE 在真实
值为零时会产生除零问题，使用前应根据数据特点做掩码或加上稳定项。
"""


def RSE(pred, true):
    """相对平方误差，按真实值的离散程度归一化。"""
    return np.sqrt(np.sum((true-pred)**2)) / np.sqrt(np.sum((true-true.mean())**2))

def CORR(pred, true):
    """计算预测值与真实值的平均相关系数。"""
    u = ((true-true.mean(0))*(pred-pred.mean(0))).sum(0) 
    d = np.sqrt(((true-true.mean(0))**2*(pred-pred.mean(0))**2).sum(0))
    return (u/d).mean(-1)

def MAE(pred, true):
    """平均绝对误差。"""
    return np.mean(np.abs(pred-true))

def MSE(pred, true):
    """均方误差。"""
    return np.mean((pred-true)**2)

def RMSE(pred, true):
    """均方根误差。"""
    return np.sqrt(MSE(pred, true))

def MAPE(pred, true):
    """平均绝对百分比误差。"""
    return np.mean(np.abs((pred - true) / true))

def MSPE(pred, true):
    """均方百分比误差。"""
    return np.mean(np.square((pred - true) / true))

def metric(pred, true):
    """一次性返回 MAE、MSE、RMSE、MAPE 和 MSPE。"""
    mae = MAE(pred, true)
    mse = MSE(pred, true)
    rmse = RMSE(pred, true)
    mape = MAPE(pred, true)
    mspe = MSPE(pred, true)
    
    return mae,mse,rmse,mape,mspe