# LaPON, GPL-3.0 license
# data augmentations & various data pre-processing

import math
import random

import os, sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# import cv2 # resize
from scipy.ndimage import zoom # resize

if __name__ == "__main__":
    sys.path.append(os.path.abspath("../"))
# else:
#     from pathlib import Path
#     FILE = Path(__file__).resolve()
#     ROOT = FILE.parents[0]  # LaPON root directory
#     if str(ROOT) not in sys.path:
#         sys.path.append(str(ROOT))  # add ROOT to PATH
#     ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative
from models.common import BoundaryPadding
from models.lapon_base import SpatialDerivative


def torch_resize(frames, new_shape=None, scale_factor=None, mode='bicubic', align_corners=False):
    # frame shape: b, t, c(u_xyz/p/force), h(y), w(x) or b, c(u_xyz/p/force), h(y), w(x)
    """
    case1:
        a = torch.arange(0, 12).reshape(1,1,1,3,4).float()
        b = torch_resize(a,new_shape=(3,2))
        c = torch_resize(a,new_shape=(3,7))
    """
    l = len(frames.shape)
    assert l in [4, 5], f"len(frame.shape) == {l} must be in [4, 5]!"
    
    h, w = frames.shape[-2:]
    
    assert not (new_shape and scale_factor), f"new_shape({new_shape}) and scale_factor({scale_factor}) can only be used for one!"
    
    if new_shape:
        if isinstance(new_shape, int):
            new_shape = (new_shape, new_shape)
    elif scale_factor:
        new_shape = int(scale_factor * h), int(scale_factor * w)
    else:
        raise ValueError(f"new_shape({new_shape}) and scale_factor({scale_factor}) are both None!")
    
    if tuple(new_shape) == (h, w): # not resize
        return frames, 1
    else:
        if h < new_shape[0] or w < new_shape[1]: # resize-magnify
            if l == 5:
                b, t, c, h, w = frames.shape
                frames = frames.reshape(-1, c, h, w)
                frames = F.interpolate(frames, 
                                       size=new_shape,
                                       # scale_factor=scale_factor, 
                                       mode=mode, align_corners=align_corners)
                nh, nw = frames.shape[-2:]
                frames = frames.reshape(b, t, c, nh, nw)
            else:
                frames = F.interpolate(frames, 
                                       size=new_shape,
                                       # scale_factor=scale_factor, 
                                       mode=mode, align_corners=align_corners)
                nh, nw = frames.shape[-2:]
        else: # resize-reduce (NOTE: not interpolation)
            step0 = 1 + ((h // new_shape[0]) - 1)
            step1 = 1 + ((w // new_shape[1]) - 1)
            frames = frames[
                ...,
                step0-1::step0,
                step1-1::step1
                ][..., :new_shape[0], :new_shape[1]]
            nh, nw = frames.shape[-2:]
        
        real_ratio = nh / h
        return frames, real_ratio

def letterbox(frame, new_shape=(640, 640), pad_mode="newman", pad_value=0, auto=True, scaleFill=False, stride=32):
    # Resize and pad frame while meeting stride-multiple constraints
    """
    case0.1:
        a = torch.arange(0, 49).reshape(1,1,1,7,7).float()
        a, letterbox(a, 2), letterbox(a, 2)[0].shape
    
    case0.2:
        a = torch.arange(0, 14).reshape(1,1,1,2,7).float()
        a, letterbox(a, 2), letterbox(a, 2)[0].shape
        
    case1:
        a = torch.arange(0, 12).reshape(1,1,1,3,4).float()
        letterbox(a, 6), letterbox(a, 6)[0].shape
    
    case2:
        a = np.arange(0, 12).reshape(1,1,1,3,4).float()
        letterbox(a, 6), letterbox(a, 6)[0].shape
    
    case3:
        a = np.arange(0, 12).reshape(1,1,3,4).float()
        letterbox(a, 6), letterbox(a, 6)[0].shape
        
    case4:
        a = np.arange(0, 2*2*603*603).reshape(1, 2, 2, 603, 603).float()
        letterbox(a, 6), letterbox(a, 6)[0].shape
    
    case5:
        a = np.arange(0, 1*603*603).reshape(1, 1, 603, 603).float()
        letterbox(a, 6), letterbox(a, 6)[0].shape
    """
    
    bc_mode = {"dirichlet": "constant", # y=C
               "newman": "nearest", # for y'=0
               "periodic": "grid-wrap", # y(0,t)=y(L,t)
               "symmetry": "reflect" # y(-x,t)=y(x,t)
               }
    
    # frame shape: b, t, c(u_xyz/p/force), h(y), w(x) or b, c(u_xyz/p/force), h(y), w(x)
    l = len(frame.shape)
    assert l in [4, 5], f"len(frame.shape) == {l} must be in [4, 5]!"
    
    shape = frame.shape[-2:]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])

    # Compute padding
    ratio = r, r  # height, width ratios
    new_unpad = int(round(shape[0] * r)), int(round(shape[1] * r))
    dw, dh = new_shape[1] - new_unpad[1], new_shape[0] - new_unpad[0]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2
    
    # convert 
    if isinstance(frame, np.ndarray):
        frame = torch.from_numpy(frame.astype("float32"))
    
    # NOTE resize (by interpolation)
    if shape != new_unpad[::-1]:  
        # frame = cv2.resize(frame, (w * 2, h * 2), interpolation=cv2.INTER_CUBIC) # INTER_CUBIC # INTER_LANCZOS4, INTER_AREA 
        # frame = nn.functional.interpolate(frame, size=(w * 2, h * 2), mode='bilinear', align_corners=False)
        # frame = nn.functional.interpolate(frame, scale_factor=2.0, mode='bicubic', align_corners=False, recompute_scale_factor=True)
        
        # frame = zoom(frame, # too slow!!!
        #           (1, 1, 1, r, r) if l == 5 else (1, 1, r, r), # scale_factor
        #           order=5, # 0：最近邻插值（Nearest）; 1：双线性插值（Bilinear）; 3：三次样条插值（Cubic）; 5：五次样条插值（Quintic）
        #           mode=bc_mode[pad_mode.lower()], # boundary condition: ‘reflect’, ‘grid-mirror’, ‘constant’, ‘grid-constant’, ‘nearest’, ‘mirror’, ‘grid-wrap’, ‘wrap’
        #           cval=0.0, # boundary condition value (in ‘constant’ mode)
        #           prefilter=True, # 样条插值前需要平滑滤波
        #           grid_mode=False # True: 对应有限体积; False: 对应有限差分
        #           )  
        # https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.zoom.html#zoom
        
        if shape[::-1][0] < new_unpad[0] or shape[::-1][1] < new_unpad[1]: # magnify
            frame, _ = torch_resize(frame, 
                                    new_unpad,
                                    # scale_factor=r
                                    )
        else: # NOTE reduce: not interpolation
            step0 = 1 + ((shape[0] // new_unpad[0]) - 1)
            step1 = 1 + ((shape[1] // new_unpad[1]) - 1)
            frame = frame[
                ...,
                step0-1::step0,
                step1-1::step1
                ][..., :new_unpad[0], :new_unpad[1]]
        
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    # print((left, right, top, bottom))
    
    # padding
    if (dw, dh) != (0.0, 0.0):
        bp = BoundaryPadding()
        if l == 5: 
            frame_ = []
            for i in range(frame.shape[1]):
                temp = bp(frame[:, i, ...], 
                          padding=(left, right, top, bottom), mode=pad_mode, value=pad_value)
                # frame_ += [temp[:, None, ...]]
                frame_ += [temp]
            # frame = torch.cat(frame_, dim=1)
            frame = torch.stack(frame_, 1) # torch.stack 将一系列张量沿着一个新的维度堆叠起来
        else:
            frame = bp(frame, padding=(left, right, top, bottom), mode=pad_mode, value=pad_value)
        del bp
    
    frame = frame.detach().cpu().numpy()
    return frame, ratio, (dw, dh)

def cutout(frame, p=0.5):
    # Applies frame cutout augmentation https://arxiv.org/abs/1708.04552
    if random.random() < p:
        h, w = frame.shape[-2:]
        scales = [0.5] * 1 + [0.25] * 2 + [0.125] * 4 + [0.0625] * 8 + [0.03125] * 16  # frame size fraction
        for s in scales:
            mask_h = random.randint(1, int(h * s))  # create random masks
            mask_w = random.randint(1, int(w * s))

            # box
            xmin = max(0, random.randint(0, w) - mask_w // 2)
            ymin = max(0, random.randint(0, h) - mask_h // 2)
            xmax = min(w, xmin + mask_w)
            ymax = min(h, ymin + mask_h)

            # apply random value mask
            for i in range(frame.shape[-3]):
                fmin = frame[... , i, :, :].min()
                fmax = frame[... , i, :, :].max()
                vmin, vmax = 64 / 255 * (fmax-fmin) + fmin, 191 / 255 * (fmax-fmin) + fmin
                frame[... , i, ymin:ymax, xmin:xmax] = random.uniform(vmin, vmax) 

    return frame

def add_noise(frame, level=0.01):
    # add noise but try not to change the original data distribution
    # frame shape: b, t, c(u_xyz/p/force), h(y), w(x) or b, c(u_xyz/p/force), h(y), w(x)
    mean, std = frame.mean(), frame.std()
    # weighted sum
    return (1-level) * frame + level * np.random.normal(mean, std, frame.shape)

def spatial_derivative(frame, grid, # 2D scalar or vector field
                       boundary_mode="newman", boundary_value=0): 
    """
    input_shape: 
        x: batch_size, (t,) c, h, w
        grid: a scalar or a Tensor (shape as x)
    output_shape(each): 
        batch_size, (t,) c, h, w
        
    case1: # 2D scalar field
        f    = torch.arange(1, 10*1*64*64+1, dtype=torch.float32).reshape(10, 1, 64, 64)
        grid = torch.ones(10, 2, 64, 64)
        spatial_derivative(f, grid, boundary_mode="periodic", boundary_value=0)[0].shape
    
    case2: # 2D vector field
        f    = torch.arange(1, 10*2*64*64+1, dtype=torch.float32).reshape(10, 2, 64, 64)
        grid = torch.ones(10, 2, 64, 64)
        spatial_derivative(f, grid, boundary_mode="periodic", boundary_value=0)[0].shape
    
    case3: # 2D vector field
        f    = torch.arange(1, 10*1*2*64*64+1, dtype=torch.float32).reshape(10, 1, 2, 64, 64)
        grid = 2 * np.pi / 1023
        spatial_derivative(f, grid, boundary_mode="periodic", boundary_value=0)[0].shape
    """
    assert len(frame.shape) in [4, 5], f"len(frame.shape)=={len(frame.shape)} must be in [4, 5]"
    flag = False
    if len(frame.shape) == 5:
        flag = True
        b, t, c, h, w = frame.shape
        frame = frame.reshape(-1, c, h, w)
        if not isinstance(grid, (int, float)):
            grid = grid.reshape(-1, c, h, w)
    
    sde = SpatialDerivative() # input_shape: b, c, h, w
    _, _, dfdx, dfdy = sde(frame, grid,
                boundary_mode=boundary_mode, boundary_value=boundary_value)
    
    if flag:
        dfdx = dfdx.reshape(b, t, c, h, w)
        dfdy = dfdy.reshape(b, t, c, h, w)
        
    return dfdx, dfdy

