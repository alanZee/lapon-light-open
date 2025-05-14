# LaPON, GPL-3.0 license
# Conventional ML trainer for LaPON

import os, glob, random, math, re, sys, time
from datetime import datetime, timedelta
import _pickle as cPickle
from copy import deepcopy
from pathlib import Path
from tqdm import tqdm 
import yaml
import logging

import numpy as np, pandas as pd, matplotlib.pyplot as plt
from PIL import Image
# from sklearn.preprocessing import MinMaxScaler, StandardScaler
# from sklearn.metrics import mean_squared_error, mean_absolute_error

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch import optim

from torch.utils.tensorboard import SummaryWriter

if __name__ == "__main__":
    sys.path.append(os.path.abspath("../"))
# else:
#     from pathlib import Path
#     FILE = Path(__file__).resolve()
#     ROOT = FILE.parents[0]  # LaPON root directory
#     if str(ROOT) not in sys.path:
#         sys.path.append(str(ROOT))  # add ROOT to PATH
#     ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative
from models.lapon4ns import PDEModule_NS, LaPON
from utils.augmentations import torch_resize, letterbox, cutout, add_noise, spatial_derivative
from cfd.iteration_method import PoissonEqSolver
from cfd.general import fd8, get_vorticity, get_energy_spectrum2d
from utils.metrics import (
    torch_reduce, 
    torch_mean_squared_error, 
    torch_r2_score, 
    get_vorticity_correlation
    )
from utils.visualization import plot_heat
from utils.general import ( 
    colorstr,
    emojis,
    increment_path,
    LOGGER, 
    ROOT, 
    DATA_PATH, 
    HYP_CFG_PATH, 
    DEFAULT_CFG_PATH
    )
from utils.torch_utils import ( 
    init_seeds,
    intersect_dicts,
    de_parallel,
    one_cycle,
    convert_optimizer_state_dict_to_fp16,
    ModelEMA,
    EarlyStopping
    )
import models
from utils.dataloader_jhtdb import create_dataloader, LoadJHTDB
from utils.dataloader_jaxcfd import create_dataloader_jaxcfd, LoadJAXCFD

from engine.base_trainer import BaseTrainer
from models.ml_models import YOLO, DeepONetCNN


#%%
class TrainerML(BaseTrainer):
    # Conventional ML trainer engine
    """
    TrainerML.

    A class for creating trainers for pure ML model training.

    Attributes:
        model (nn.Module): Model instance.
        save_dir (Path): Directory to save results.
        wdir (Path): Directory to save weights.
        last (Path): Path to the last checkpoint.
        best (Path): Path to the best checkpoint.
        start_epoch (int): Starting epoch for training.
        scaler (amp.GradScaler): Gradient scaler for AMP.
        train_loader (torch.utils.data.Dataloader): Training dataloader.
        val_loader (torch.utils.data.Dataloader): Validation dataloader.
        train_loader.dataset (torch.utils.data.Dataset): Training dataset.
        val_loader.dataset (torch.utils.data.Dataset): Validation dataset.
        ema (nn.Module): EMA (Exponential Moving Average) of the model.
        loss_fn (nn.Module): Loss function.
        lf (function): learning rate scheduler function.
        scheduler (torch.optim.lr_scheduler._LRScheduler): Learning rate scheduler.
        best_metrics (float): The best metrics value achieved.
        best_epoch (int): The epoch corresponding to the best metrics.
        metrics (float): Current metrics value.
    """

    def __init__(self,
                 cfg=DEFAULT_CFG_PATH, # cfg file path for train etc.
                 yolo_or_deeponet='yolo', # choose ml model
                 ):
        self.yolo_or_deeponet = yolo_or_deeponet
        super().__init__(cfg)
        
    def setup_model(self, evaluation=False):
        """Create (& load) model."""
        if self.yolo_or_deeponet == 'yolo':
            Model = YOLO
        elif self.yolo_or_deeponet == 'deeponet':
            Model = DeepONetCNN
        else:
            raise ValueError(f"yolo_or_deeponet(={self.yolo_or_deeponet}) must be in ['yolo', 'deeponet']")
        
        # create model
        self.model = Model(
            (None, self.cfg["history_num"] + 1, 2, self.cfg["framesz"], self.cfg["framesz"]), 
            (None, self.cfg["history_num"] + 1, 2, self.cfg["framesz"], self.cfg["framesz"]), 
            (None, 1, self.cfg["history_num"] + 1), 
            (None, 1, 3), 
            use_isotropic_conv = self.cfg["use_isotropic_conv"],
            stride = int(self.cfg["stride"]),
            depth_multiple = float(self.cfg["depth_multiple"]),
            width_multiple = float(self.cfg["width_multiple"]),
            )
        
        # get ckpt path
        ckpt_path = None
        if self.cfg["resume"]: # resume # high priority
            ckpt_path = self.last
        elif self.cfg["model_weights"]: # pretrain # low priority
            ckpt_path = Path(self.cfg["model_weights"])
        
        # check & load ckpt file
        if ckpt_path is not None:
            if ckpt_path.exists():
                self.load_model(ckpt_path)
            else:
                LOGGER.error(f"ERROR ❌ the ckpt {colorstr('red', 'bold', ckpt_path)} "
                             f"does {colorstr('red', 'bold', 'not')} exist." + '\n')
                raise FileExistsError
        else:
            LOGGER.warning(f"WARNING ⚠️ the ckpt {colorstr('red', 'bold', ckpt_path)} "
                           f"is {colorstr('red', 'bold', 'None')}.\n" 
                           f"Model {colorstr('red', 'bold', 'weights')} "
                           f"are {colorstr('red', 'bold', 'randomly')} generated." + '\n')
        
        # to device
        self.model = self.model.to(self.cfg["device"])
        
        # Freeze
        # freeze = [f'model.{x}.' for x in (freeze if len(freeze) > 1 else range(freeze[0]))]  # layers to freeze
        # for k, v in model.named_parameters():
        #     v.requires_grad = True  # train all layers
        #     if any(x in k for x in freeze):
        #         LOGGER.info(f'freezing {k}' + '\n')
        #         v.requires_grad = False
        
        self.model.no_ni = True # Disable the normalization and inverse normalization of the inputs and outputs of each ANN module
        self.model.out_mode = 'normal' # output: next_frame, current_pressure
            
