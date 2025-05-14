# LaPON, GPL-3.0 license

import os, sys
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

if __name__ == "__main__":
    sys.path.append(os.path.abspath("../"))
# else:
#     from pathlib import Path
#     FILE = Path(__file__).resolve()
#     ROOT = FILE.parents[0]  # LaPON root directory
#     if str(ROOT) not in sys.path:
#         sys.path.append(str(ROOT))  # add ROOT to PATH
#     ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative
from cfd.general import get_vorticity

from sklearn.metrics import ( 
    mean_squared_error, 
    mean_absolute_error, 
    mean_absolute_percentage_error, 
    r2_score
    )


def torch_reduce(x, reduction='ts mean', channel_reduction=False, keepdim=False):
    """
    input shape: b/t, c, h, w
    case:
        error = torch.rand((10, 2, 64, 64))
        error = torch_reduce(error, reduction='ts mean', channel_reduction=False, keepdim=False)
        error.shape
    """
    b, c, h, w = x.shape
    
    if channel_reduction:
        x = torch.mean(x, dim=1, keepdim=True)
        c = 1
        
    if reduction == 'ts mean': # temporal spatial mean
        x = torch.mean(
            x.transpose(0, 1).reshape((c, -1)), 
            dim=-1,
            keepdim=keepdim) 
        return x if not keepdim else x.reshape((1, c, 1, 1)) # shape: c / 1, c, 1, 1
    elif reduction == 'temporal mean':
        x = torch.mean(x, dim=0, keepdim=keepdim) 
        return x if not keepdim else x.reshape((1, c, h, w)) # shape: c, h, w / 1, c, h, w
    elif reduction == 'spatial mean':
        x = torch.mean(
            x.reshape((b, c, -1)), 
            dim=-1,
            keepdim=keepdim) 
        return x if not keepdim else x.reshape((b, c, 1, 1)) # shape: b, c / b, c, 1, 1
    elif reduction == 'none':
        return x # shape: b, c, h, w
    else:
        raise ValueError("reduction must be in ['ts mean', 'temporal mean', 'spatial mean', 'none'], "
                         f"but given {reduction}.")


def torch_mean_squared_error(y_true, y_pred, reduction='ts mean', channel_reduction=False, keepdim=False):
    """
    input shape: b/t, c, h, w
    case:
        y_true = torch.rand((10, 2, 64, 64))
        y_pred = torch.rand((10, 2, 64, 64))
        error  = torch_mean_squared_error(y_true, y_pred, reduction='ts mean', channel_reduction=False, keepdim=False)
        error.shape
    """
    e = torch.square(y_true - y_pred)
    return torch_reduce(e, reduction, channel_reduction, keepdim)


def torch_r2_score(y_true, y_pred, reduction='ts mean', channel_reduction=False, keepdim=False):
    """
    Calculate R-squared score.
    :param y_true: Tensor, true target values.
    :param y_pred: Tensor, predicted values.
    :return: Tensor, R-squared score.
    
    input shape: b/t, c, h, w
    case:
        y_true = torch.rand((10, 2, 64, 64))
        y_pred = torch.rand((10, 2, 64, 64))
        error  = torch_r2_score(y_true, y_pred, reduction='ts mean', channel_reduction=False, keepdim=False)
        error.shape
    """
    y_true_mean = torch_reduce(y_true, reduction, channel_reduction, True)
    total_variance = torch_mean_squared_error(y_true, y_true_mean, reduction, channel_reduction, keepdim)
    residual_variance =  torch_mean_squared_error(y_true, y_pred, reduction, channel_reduction, keepdim)
    r2 = 1 - (residual_variance / total_variance)
    return r2


def get_corrcoef(var1, var2, not_reduce=False):
    """
    input shape (each):
        n, c, h, w
    output:
        tuple or batch: mean, std of correlation; or all batch no reduce (n,)
    
    +1-完全正相关
    +0.8-强正相关
    +0.6-中等正相关
    0-无关联
    -0.6-中度负相关
    -0.8-强烈的负相关
    -1-完全负相关
    
    case:
        y = np.random.random((10, 1, 64, 64))
        y_ = np.random.random((10, 1, 64, 64))
        mean, std = get_corrcoef(y, y_)
    """
    correlation = []
    for v1, v2 in zip(var1, var2):
        temp = np.corrcoef(
            v1.flatten(), 
            v2.flatten(), 
                # A 1-D or 2-D array containing multiple variables and observations. 
                # Each row of x represents a variable, 
                # and each column a single observation of all those variables.
            # rowvar=False 
                # True(default): 皮尔逊(Pearson)相关系数(线性关联); 
                # False: 斯皮尔曼相关系数(非线性关联)
            )[0, 1]
        correlation += [temp]
    correlation = np.array(correlation)
    # print(correlation.shape)
    
    if not not_reduce:
        return correlation.mean(), correlation.std()
    else:
        return correlation


def get_corrcoef_abs(var1, var2):
    return tuple(np.abs(get_corrcoef(var1, var2)))


def get_vorticity_correlation(frame, label, dx, dy, 
                              boundary_mode="newman", boundary_value=0, 
                              mandatory_use_cuda=False, return_np=True,
                              not_reduce=False):
    """
    input shape:
        frame & label: n, c=2, h, w
        dx & dy: a scalar
    output:
        tuple: mean, std of vorticity correlation
        
    case:
        f  = torch.randn(10, 2, 64, 64).to("cuda")
        l  = torch.randn(10, 2, 64, 64).to("cuda")
        dx = dy = 0.006
        mean, std = get_vorticity_correlation(f, l, dx, dy, boundary_mode="periodic", boundary_value=0)
        print(mean, std)
    """
    omiga1 = get_vorticity(frame, dx, dy, 
                          boundary_mode=boundary_mode, boundary_value=boundary_value,
                          mandatory_use_cuda=mandatory_use_cuda, return_np=return_np)
    omiga2 = get_vorticity(label, dx, dy, 
                          boundary_mode=boundary_mode, boundary_value=boundary_value,
                          mandatory_use_cuda=mandatory_use_cuda, return_np=return_np)
    return get_corrcoef(omiga1, omiga2, not_reduce=not_reduce) # return mean, std or all batch no reduce

