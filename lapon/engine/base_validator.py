# LaPON, GPL-3.0 license
# Validator(Base) of LaPON 

import os, shutil, random, math, sys, time
from datetime import datetime
from copy import deepcopy
from pathlib import Path
from tqdm import tqdm 
import yaml
import logging
from typing import Any, Dict, Mapping, Optional, Tuple, Union
import xarray as xr

import numpy as np, matplotlib.pyplot as plt
# from sklearn.preprocessing import MinMaxScaler, StandardScaler
# from sklearn.metrics import mean_squared_error, mean_absolute_error

import torch
import torch.nn as nn
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
import utils.xarray_utils as xru
from utils.augmentations import torch_resize, letterbox, cutout, add_noise, spatial_derivative
from utils.general import ( 
    colorstr,
    increment_path,
    LOGGER, 
    DATA_PATH, 
    HYP_CFG_PATH, 
    DEFAULT_CFG_PATH
    )
from utils.torch_utils import ( 
    init_seeds,
    intersect_dicts,
    one_cycle,
    convert_optimizer_state_dict_to_fp16,
    ModelEMA,
    EarlyStopping
    )
import models
from utils.dataloader_jhtdb import create_dataloader, LoadJHTDB
# from utils.dataloader_jaxcfd import create_dataloader_jaxcfd, LoadJAXCFD
create_dataloader_jaxcfd, LoadJAXCFD = create_dataloader, LoadJHTDB

from engine.base_trainer import BaseTrainer

from cfd.general import fd8, get_vorticity, get_energy_spectrum2d
from utils.metrics import (
    torch_reduce, 
    torch_mean_squared_error, 
    torch_r2_score, 
    get_corrcoef,
    get_vorticity_correlation
    )
from utils.visualization import *


# DEFAULT_CFG_PATH = DEFAULT_CFG_PATH.parent / (DEFAULT_CFG_PATH.with_suffix('').name + '_validate.yaml')

FILTER_PERIOD = { 
    '(1024, 1024)': 4,
    '(512, 512)': 6,
    '(256, 256)': 12,
    '(128, 128)': 16,
    '(64, 64)': 18,
    '(32, 32)': 100,
    'lapon': 20,
                 }


#%%
class BaseValidator:
    # LaPON validator engine 
    """
    Base Validator. Contains a series of validation process baselines.

    A class for creating validator.
    
    For time-consuming process(validation on the entire dataset -- eval_loop):
    Auto-saving prediction results to the local file.
    And then only need to load the data in the file during post-processing, 
    avoiding repeated deduction, greatly saving time and for flexible post-processing.

    Attributes:
        model (nn.Module): Model instance.
        save_dir (Path): Directory to save results.
        save_dir_replot (Path): Directory to save replotting figs from plotting data.
        wdir (Path): Directory to save weights.
        loader (torch.utils.data.Dataloader): Validation dataloader.
        loader.dataset (torch.utils.data.Dataset): Validation dataset.
        loaders (list): List of all validation dataloaders (self.loaders[0] = self.loader). 
    """

    def __init__(self, 
                 cfg=DEFAULT_CFG_PATH, # cfg file path for validate etc.
                 ):
        # cfg
        with open(cfg, errors='ignore') as f:
            self.cfg = yaml.safe_load(f)  # load cfg dict
        assert self.cfg, f'ERROR ❌No cfg ({cfg}) found'
        
        init_seeds(self.cfg['seed'], deterministic=self.cfg['deterministic'])
        
        # hyp
        with open(self.cfg["hyp"] if self.cfg["hyp"] else HYP_CFG_PATH, errors='ignore') as f:
            self.hyp = yaml.safe_load(f)  # load hyps dict
        assert self.hyp, f'ERROR ❌ No hyp ({self.cfg["hyp"]}) found'
        
        # Directories & files
        exist_ok = True if self.cfg["resume_val"] else False
        self.save_dir = increment_path(Path(self.cfg["project_val"]) / self.cfg["name_val"],
                                       exist_ok=exist_ok, mkdir=True)
        self.wdir = self.save_dir / 'weights'  # weights dir
        self.wdir.mkdir(parents=True, exist_ok=True)  # make dir
        self.weight = self.wdir / 'weight.pt'
        self.save_dir_results = self.save_dir / 'results_data' # auto_save_results()
        self.save_dir_results.mkdir(parents=True, exist_ok=True)  # make dir
        
        # set LOGGER
        log_dir = self.save_dir / "log_run"
        log_dir.mkdir(parents=True, exist_ok=True)  # make dir
        file_handler = logging.FileHandler(log_dir / "logfile.log")
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(
            logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
        LOGGER.addHandler(file_handler)
        
        # The main preparations - model (Create & load & save)
        self.setup_model()
        
        # The main preparations - dataloader
        self.get_dataloader()
        
        # clean cache
        try:
            del self.ckpt
        except:
            pass

        # Save run settings
        with open(self.save_dir / 'cfg.yaml', 'w') as f:
            yaml.safe_dump(self.cfg, f, sort_keys=False)
        with open(self.save_dir / 'hyp.yaml', 'w') as f:
            yaml.safe_dump(self.hyp, f, sort_keys=False)
        
        # Log
        LOGGER.info(f'Training/Validation with {colorstr("cfg")}: \n' + ', '.join(f'{k}={v}' for k, v in self.cfg.items()) +
                    f'\nTraining/Validation with {colorstr("hyp")}: \n' + ', '.join(f'{k}={v}' for k, v in self.hyp.items()) + '\n' +
                    f'\nsave_dir: {colorstr(self.save_dir)} \n' + '\n')
    
    
    def auto_save_results(
            self, 
            file_name: str, # i.e. 'example.nc'
            dx: float, # attrs
            dt: float, # attrs
            prediction: np.array, # item value
            target: np.array, # item value
            keys: str = ['prediction_velocity', 'target_velocity'], # item name
            cat_dim: int = 1, # concat dim
            extra_cat_dim_0: bool = False, # extra concat dim_0
            ) -> int:
        dims = ['sample', 'time', 'channel', 'x', 'y'] # 'x'(h), 'y'(w)
        grid_size = prediction.shape[-1]
        file = self.save_dir_results / file_name
            # (Path(file_name).with_suffix('').name + f'_{grid_size}x{grid_size}.nc')
        file_ = file.with_suffix('._.nc')
        file_temp = file.with_suffix('.temp.nc')
        file_temp_ = file.with_suffix('.temp_.nc')
        """
        file: for concat sample
        file_temp: for concat time temporarily
        """

        # Creat DataArray1
        da1 = xr.DataArray(
            prediction, 
            # coords=[lat, lon], 
            dims=dims
            ) #.rename(keys[0])
        
        # Creat DataArray2
        da2 = xr.DataArray(
            target, 
            # coords=[lat, lon], 
            dims=dims
            ) #.rename(keys[1])
        
        # Creat Dataset
        ds = xr.Dataset({
            keys[0]: da1,
            keys[1]: da2,
            })
        ds.coords['channel'] = ['u', 'v']
        ds.coords['sample'] = np.arange(ds[keys[0]].shape[0]) #.astype(np.int32)
        ds.coords['time'] = np.linspace(dt, ds[keys[0]].shape[1] * dt, ds[keys[0]].shape[1]).astype(np.float32)
        ds.coords['x'] = np.linspace(dx, (ds[keys[0]].shape[3]-1) * dx, ds[keys[0]].shape[3]).astype(np.float32)
        ds.coords['y'] = ds.coords['x'].values
        # da1.attrs['ndim'] = 3 # 添加属性
        ds.attrs.update(dict(
            ndim = 2, # 2D physical field
            grid_size = grid_size, # i.e. 128
            dx = dx, # i.e. 0.0491
            dt = dt, # i.e. 2e-3
            # domain_size = (self.loader.dataset.origin_dx * 
            #                (self.loader.dataset.resolution_x - 1)), # i.e. 2 * np.pi
            # origin_dx = self.loader.dataset.origin_dx, # i.e. 2 * np.pi / (1024-1)
            # origin_dt = self.loader.dataset.origin_dt, # i.e. 2e-4
            # rho = self.loader.dataset.rho, # 1.29
            # diss = self.loader.dataset.diss, # 0.103
            # nu = self.loader.dataset.nu, # 0.000185
            ))
        
        # load existing dataset & concat datasets
        if file_temp.exists():
            ds0 = xr.open_dataset(file_temp)
            ds = xr.concat([ds0, ds], dim=dims[cat_dim])
            ds.coords['time'] = np.array(
                ds0.coords['time'].values.tolist() +
                [ds0.coords['time'].values[-1] + dt]
                ).astype(np.float32) # reset coords
            # np.linspace(dt, ds[keys[0]].shape[1] * dt, ds[keys[0]].shape[1]).astype(np.float32) # reset coords
            # ds0.close() # Close
        
        # Save dataset to NetCDF file
        ds.to_netcdf(file_temp_) # , engine='netcdf4' 'h5netcdf'
        # Close
        try:
            ds0.close()
        except:
            pass
        # rename (overwrite the original file)
        shutil.move(file_temp_, file_temp) 
        
        # extra_cat_dim_0
        if extra_cat_dim_0:
            # load existing dataset & concat datasets
            if file.exists():
                ds0 = xr.open_dataset(file)
                ds = xr.concat([ds0, ds], dim=dims[0]) # 数据集会被读到内存中进行拼接, 过大数据集会爆显存
                # 控制体报错： MemoryError: Unable to allocate 23.5 GiB for an array with shape (94, 32, 2, 1024, 1024) and data type float32
                ds.coords['sample'] = range(ds[keys[0]].shape[0]) #.astype(np.int32) # reset coords
                # ds0.close() # Close
            
            # Save dataset to NetCDF file
            ds.to_netcdf(file_) # , engine='netcdf4' 'h5netcdf'
            # Close
            try:
                ds0.close()
            except:
                pass
            # rename (overwrite the original file)
            shutil.move(file_, file) 
            # remove 
            os.remove(file_temp)
        
        # ds.close()
        return 1
    
    
    def get_vc_on_ds(self, ds: xr.Dataset, out_resize=None):
        # get vorticity correlation between prediction and target on a xr.Dataset
        out_resize = (self.cfg['out_resize'] 
                      if self.cfg['out_resize'] is not None else 
                      self.cfg['frame_size_ls_line'][-1]
                      ) if out_resize is None else out_resize
        
        # # resize
        # ds = xru.resize_reduce_n_(ds, n=int(ds.attrs['grid_size'] / out_resize[0]))
        
        # # get vorticity
        # prediction_vorticity = xru.vorticity_2d_(ds, key='prediction_velocity').values[:, :, None, ...]
        # target_vorticity = xru.vorticity_2d_(ds, key='target_velocity').values[:, :, None, ...]
        
        # # get correlation
        # vc = []
        # for time_idx in range(prediction_vorticity.shape[1]):
        #     vc += [get_corrcoef(
        #         prediction_vorticity[:, time_idx, ...],
        #         target_vorticity[:, time_idx, ...],
        #         not_reduce=True
        #         )]
        
        
        dx = (ds.x[1] - ds.x[0]).item()
        dy = (ds.y[1] - ds.y[0]).item()
        try:
            dt = (ds.time[1] - ds.time[0]).item()
        except:
            dt = ds.time.values[0]
        
        batch_size = 32
        vc = []
        for infer_time in tqdm(
                ds.time.values,
                total=ds.dims['time'], 
                bar_format=#f"epoch {epoch}/{epochs-1} | Eval"
                f'Eval(frame_size, dt = {ds.x.shape[0], dt}): '
                '{l_bar}{bar:10}{r_bar}{bar:-10b}'
                ):
            vc_one_time = np.array([])
            for i in range(0, ds.dims['sample'], batch_size):
                i = int(i)
                j = min(i + batch_size - 1, ds.dims['sample'] - 1) # idx of last sample in current batch
                prediction_velocity = ds['prediction_velocity'].sel(sample=slice(i, j, 1), time=infer_time).values # Left closed, right closed
                target_velocity = ds['target_velocity'].sel(sample=slice(i, j, 1), time=infer_time).values
            
                # resize
                prediction_velocity_resized, real_ratio = torch_resize(prediction_velocity, new_shape=out_resize)
                target_velocity_resized, real_ratio = torch_resize(target_velocity, new_shape=out_resize)
                # updata dx dy
                dx_resized = dx / real_ratio
                dy_resized = dy / real_ratio
                
                # Vorticity correlation
                vc_temp = get_vorticity_correlation( # shape: (n,) 
                    target_velocity_resized, 
                    prediction_velocity_resized,
                    dx_resized, dy_resized,
                    not_reduce=True) 
                vc_one_time = np.concatenate( # shape: (sample,)
                    [vc_one_time, vc_temp], 
                    axis=0) 
                # Velocity R2
                # r2_temp = torch_r2_score( # shape:(n, 1)
                #     target_velocity_resized, 
                #     prediction_velocity_resized, 
                #     reduction='spatial mean', channel_reduction=True, 
                #     keepdim=False)
                # r2_temp = r2_temp.reshape(-1) # shape:(n,)
                
            # record
            vc += [vc_one_time] # len: time step; item_shape: (sample,)
            
        # mean, std, p1
        vc_mean = np.array([i.mean(axis=0) for i in vc])
        vc_std = np.array([i.std(axis=0) for i in vc])
        vc_p1 = np.array([np.percentile(i, 1, axis=0) for i in vc])
        
        return [vc_mean, vc_std, vc_p1] # shape: (No. of time step,)
    
    
    def setup_model(self, evaluation=True):
        """Create (& load & save) model."""
        model_module = eval(self.cfg["model"])
        LaPON = model_module.LaPON
        
        # create model
        self.model = LaPON(
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
        ckpt_path = self.cfg["model_weights_val"]
        
        # check & load ckpt file
        if ckpt_path is not None:
            if Path(ckpt_path).exists():
                self.load_model(Path(ckpt_path), evaluation=evaluation)
            else:
                LOGGER.error(f"ERROR ❌ the ckpt {colorstr('red', 'bold', ckpt_path)} "
                             f"does {colorstr('red', 'bold', 'not')} exist." + '\n')
                raise FileExistsError
        else:
            LOGGER.warning(f"WARNING ⚠️ the ckpt {colorstr('red', 'bold', ckpt_path)} "
                           f"is {colorstr('red', 'bold', 'None')}.\n" 
                           f"Model {colorstr('red', 'bold', 'weights')} "
                           f"are {colorstr('red', 'bold', 'randomly')} generated." + '\n')
            self.ckpt = {
                "epoch": 0, # -1 means fit is finished
                "best_metrics": 0,
                # "weight": deepcopy(self.model).half().state_dict(),
                "model": deepcopy(self.model).half(), #None, #deepcopy(self.model).half(), # resume and final checkpoints derive from EMA
                "ema": None, #deepcopy(self.ema.ema).half(),
                "updates": None, #self.ema.updates,
                "optimizer": None, #convert_optimizer_state_dict_to_fp16(deepcopy(self.optimizer.state_dict())),
                "date": datetime.now().isoformat(),
            }
        
        # to device
        # self.model = self.model.to(self.cfg["device"])
        
        # self.model.no_ni = True # Disable the normalization and inverse normalization of the inputs and outputs of each ANN module
        # self.model.out_mode = 'normal' # output: next_frame, current_pressure
        # self.model.out_mode = 'all' # output: Direct output (no inverse norm) of every neural operator etc. (for Neural operators are trained separately)
        # self.model.operator_cutoff = False # cut off gradient flow of operators (for Neural operators are trained separately)
        # self.model.no_constrain = False # remove hard condstrain of operator's output
        
        # ---------------------------------------------------------------------
        
        # save weight
        import io
        # Serialize ckpt to a byte buffer once (faster than repeated torch.save() calls)
        buffer = io.BytesIO()
        torch.save(
            self.ckpt,
            buffer,
        )
        serialized_ckpt = buffer.getvalue()  # get the serialized content to save
        # Save checkpoints
        self.weight.write_bytes(serialized_ckpt)  # save weight
    
    
    def load_model(self, ckpt_path='lapon.pt', evaluation=True):
        """Load and set model."""
        # self.model.load_state_dict(torch.load('model_weight.pth'))
        # self.model = torch.load('model.pth')
        
        # load ckpt file
        try:
            self.ckpt = torch.load(ckpt_path, map_location='cpu')  # load checkpoint to CPU to avoid CUDA memory leak
        except Exception as e:
            LOGGER.error(f"ERROR ❌ the ckpt {colorstr('red', 'bold', ckpt_path)} "
                         f"{colorstr('red', 'bold', 'failed')} to load." + '\n')
            raise e
        
        # check ckpt file
        ckpt_keys = {'best_metrics', 'date', 'ema', 'epoch', 'model', 'optimizer', 'updates'}
        if not isinstance(self.ckpt, dict):
            LOGGER.error(f"ERROR ❌ the ckpt {colorstr('red', 'bold', ckpt_path)} "
                         f"is {colorstr('red', 'bold', 'not')} dict that is expected." + '\n')
            raise ValueError
        elif not {'ema', 'model'} & set(self.ckpt.keys()):
            LOGGER.error(f"ERROR ❌ the ckpt {colorstr('red', 'bold', ckpt_path)} "
                         f"is {colorstr('red', 'bold', 'not')} the expected dict.\n"
                         f"The expected keys of ckpt dict must contain {colorstr('red', 'bold', 'ema or model')},\n"
                         f"but the loaded ckpt file is {colorstr('red', 'bold', set(self.ckpt.keys()))}." + '\n')
            raise ValueError
            
        # NOTE replace self.model (NOT set model weight (state_dict))
        # csd = (self.ckpt.get("ema") or self.ckpt["model"]).float().state_dict()  # FP32 model # Priority EMA
        # csd = intersect_dicts(csd, self.model.state_dict())  # intersect
        # # self.model.load_state_dict(csd, strict=False)  # load
        # self.model.load_state_dict(csd, strict=True)  # load
        # # clean cache
        # del csd
        self.model = (self.ckpt.get("ema") or self.ckpt["model"]).float()

        # Model compatibility updates
        self.model.pt_path = ckpt_path  # attach *.pt file path to model
        if not hasattr(self.model, "stride"):
            self.model.stride = torch.tensor([32.0])

        # Module updates
        for m in self.model.modules():
            if isinstance(m, nn.Upsample) and not hasattr(m, "recompute_scale_factor"):
                m.recompute_scale_factor = None  # torch 1.11.0 compatibility

        # evaluation mode (different on behavior of dropout & batch normalization etc.)
        if evaluation: 
            self.model.eval()
        
        LOGGER.info(f"the ckpt {colorstr('green', 'bold', ckpt_path)} "
                    f"loads {colorstr('green', 'bold', 'successfully')}." + '\n')
        
        
    def get_dataloader(self):
        """
        Get dataloader.
        """
        self.loaders = []
        
        path = self.cfg["data_set_val"] if self.cfg["data_set_val"] else DATA_PATH
        if 'jax' not in path: # jhtdb dataset
            create = create_dataloader
        else:
            create = create_dataloader_jaxcfd
        # print(colorstr('green', 'bold', path))
        self.loader = create(
            path, # data root dir
            frame_size=self.cfg["framesz"], # preset frame shape; >= 128 if hyp.multi_scale == [0.5,1.5]
            batch_size=self.cfg["batch_val"][str((self.cfg["framesz"],)*2)],
            history_num=self.cfg["history_num_val"], # 8 
            augment=self.cfg["augment_val"], # augmentation (just preset state)
            hyp=self.cfg["hyp"] if self.cfg["hyp"] else HYP_CFG_PATH, # hyperparameters
            rect=self.cfg["rect"], # Rectangular Training(rectangular batches)
            stride=int(self.cfg["stride"]), # stride for Rectangular Training
            pad_mode="newman", #self.cfg["pad_mode"], # boundary condition
            pad_value=0, #self.cfg["pad_value"], # boundary condition
            field_axis=self.cfg["field_axis"], # The axis of the field slice, xy or xz or yz
            workers=self.cfg["workers"], # 8 / 0
            shuffle=False,
            time_interval=1, 
            time_len=130, 
            )
        self.loaders += [self.loader]
        
        for path_i in [
                self.cfg["data_set_val1"],
                self.cfg["data_set_val2"],
                self.cfg["data_set_val3"],
                self.cfg["data_set_val4"],
                ]:
            if 'isotropic1024coarse' in path_i: # jhtdb dataset
                create = create_dataloader
            else:
                create = create_dataloader_jaxcfd
            # print(colorstr('green', 'bold', path_i))
            loader = create(
                 path_i if path_i is not None else DATA_PATH, # data root dir
                 frame_size=self.cfg["framesz"], # preset frame shape; >= 128 if hyp.multi_scale == [0.5,1.5]
                 batch_size=self.cfg["batch_val"][str((self.cfg["framesz"],)*2)],
                 history_num=self.cfg["history_num_val"], # 8 
                 augment=self.cfg["augment_val"], # augmentation (just preset state)
                 hyp=self.cfg["hyp"] if self.cfg["hyp"] else HYP_CFG_PATH, # hyperparameters
                 rect=self.cfg["rect"], # Rectangular Training(rectangular batches)
                 stride=int(self.cfg["stride"]), # stride for Rectangular Training
                 pad_mode="newman", #self.cfg["pad_mode"], # boundary condition
                 pad_value=0, #self.cfg["pad_value"], # boundary condition
                 field_axis=self.cfg["field_axis"], # The axis of the field slice, xy or xz or yz
                 workers=self.cfg["workers"], # 8 / 0
                 shuffle=False,
                 time_interval=1, 
                 time_len=130, 
                 )
            self.loaders += [loader]
        
    
    def validate(self): # validate = predict (+ calculate + metrics) + plot + LOGGER
        """NOTE Main method : Complete standardized validation process of the model, 
        predict, calculate vorticity, get metrics and plot figs."""
        
        # Start 
        self.time_start = time.time()
        # Log 
        LOGGER.info(f"Logging results to {colorstr('bold', self.save_dir)}\n"
                    f'Starting(validate)...' + '\n')
        
        # accuracy ------------------------------------------------------------
        fig_names = [
            "accuracy_heat.png" if not self.cfg['contour'] else "accuracy_contour.png",
            "accuracy_correlation.png",
            "accuracy_spectrum.png",
            ]
        fig_sizes = [ # w, h
            (7 * 4 / 3, 7),
            (7 * 0.8, 7),
            (7 * 0.8, 7),
            ]
        self.accuracy_vorticity(figname=fig_names[0], figsize=fig_sizes[0]) 
        self.accuracy_vorticity_correlation(figname=fig_names[1], figsize=fig_sizes[1])
        self.accuracy_energy_spectrum(figname=fig_names[2], figsize=fig_sizes[2])
        # concatenate figs
        concat_fig(img_path_ls=[self.save_dir / i for i in fig_names], 
                   orientation='h')
        # ---------------------------------------------------------------------
        
        # stability -----------------------------------------------------------
        fig_names = [
            "stability_heat.png" if not self.cfg['contour'] else "stability_contour.png",
            "stability_correlation.png",
            "stability_noise_immunity.png",
            "stability_data_missing_resistance.png",
            "stability_noise_cutout_vis.png"
            ]
        fig_sizes = [ # w, h
            (7 * 4 / 3, 7),
            (7 * (4 / 3 + 0.5), 7 * (4 / 3 + 0.5) * 0.618), # (12.833, 7.931)
            (7 * 0.8, 7),
            (7 * (0.3 + 1/2.2), 7 * (4 / 3 + 0.5) * 0.618), # (5.282, 7.931)
            (7 / 2.2, 7) 
            ]
        self.stability_vorticity(figname=fig_names[0], figsize=fig_sizes[0]) 
        self.stability_vorticity_correlation(figname=fig_names[1], figsize=fig_sizes[1])
        self.stability_noise_immunity(figname=fig_names[2], figsize=fig_sizes[2])
        self.stability_data_missing_resistance(figname=fig_names[3], figsize=fig_sizes[3])
        self.stability_noise_cutout_vis(figname=fig_names[4], figsize=fig_sizes[4])
        # concatenate figs
        concat_fig(img_path_ls=[self.save_dir / i for i in [
                                fig_names[0], fig_names[2], fig_names[4]
                                ]], 
                   orientation='h',
                   out_figname='stability_concat_temp1.png')
        concat_fig(img_path_ls=[self.save_dir / i for i in [
                                fig_names[1], fig_names[3]
                                ]], 
                   orientation='h',
                   out_figname='stability_concat_temp2.png')
        concat_fig(img_path_ls=[self.save_dir / 'stability_concat_temp1.png',
                                self.save_dir / 'stability_concat_temp2.png'], 
                   orientation='v')
        # ---------------------------------------------------------------------
        
        # generalization  -----------------------------------------------------
        # loader = self.loaders[1] # specify the dataset
        # loader = self.loaders[2] # specify the dataset
        # loader = self.loaders[3] # specify the dataset
        loader = self.loaders[4] # specify the dataset
        
        fig_names = [
            "generalization_heat.png" if not self.cfg['contour'] else "generalization_contour.png",
            "generalization_correlation.png",
            "generalization_spectrum.png",
            ]
        fig_sizes = [ # w, h
            (7 * 4 / 3, 7),
            (7 * 0.8, 7),
            (7 * 0.8, 7),
            ]
        # test accuracy on other dataset
        self.accuracy_vorticity(figname=fig_names[0], figsize=fig_sizes[0],
                                loader=loader, generalization=True) 
        self.accuracy_vorticity_correlation(figname=fig_names[1], figsize=fig_sizes[1],
                                loader=loader, generalization=True)
        self.accuracy_energy_spectrum(figname=fig_names[2], figsize=fig_sizes[2],
                                loader=loader, generalization=True)
        # concatenate figs
        concat_fig(img_path_ls=[self.save_dir / i for i in fig_names], 
                   orientation='h')
        # ---------------------------------------------------------------------
        
        #  Log 
        print("✅ " + colorstr('green', 'bold', 'underline', 'finish!'))
        LOGGER.info(colorstr('Validation finish:\n') + 
                    f'Completed in {(time.time() - self.time_start) / 3600:.3f} hours.\n' + '\n')
        
        # Clean up the cache of GPU
        torch.cuda.empty_cache()
        
    
    def accuracy_vorticity(
            self,
            save_dir: Path = None,
            figname: str = "accuracy_heat.png",
            figsize: Tuple[float, ...] = None,
            top_title: Tuple[str, ...] = None,
            left_title: Tuple[str, ...] = None,
            loader=None, # specify the dataset
            generalization=False, # use cfg of generalization validation, not accuracy
            ):
        """ predict (+ calculate + metrics) + plot """
        save_dir = self.save_dir if save_dir is None else save_dir
        top_title = ((self.cfg['dt_label_ls_acc_heat'] if not generalization else 
                      self.cfg['dt_label_ls_gnr_heat'])
                     if top_title is None else top_title)
        left_title = self.cfg['model_name_ls_heat'] if left_title is None else left_title 
        figsize = (7 * len(top_title) / len(left_title), 7) if figsize is None else figsize
        
        LOGGER.info(f"Run {colorstr('green', 'bold', 'accuracy_vorticity')}...\n")
        
        ### predict + calculate vorticity
        idx_init = self.cfg['one_sample_idx']
        results = []
        vorticity_target = []
        for model_idx, (frame_size, only_numerical) in enumerate(zip(
                self.cfg['frame_size_ls_heat'],
                self.cfg['only_numerical_ls_heat'])): 
            results_one_model = []
            for dt_multiple in (self.cfg['dt_multiple_ls_acc_heat'] if not generalization else 
                                self.cfg['dt_multiple_ls_gnr_heat']):
                file_name = (
                    f"{Path(figname).with_suffix('').name}_{'DS' if only_numerical else 'LaPON'}_"
                    f"{frame_size[0]}x{frame_size[1]}_{dt_multiple}dt.nc"
                    )
                
                if (self.save_dir_results / file_name).exists() and not self.cfg['recal_results']:
                    # Get vorticity correlation between prediction and target on a existing xr.Dataset
                    ds = xr.open_dataset(self.save_dir_results / file_name) 
                    # resize
                    out_resize = (self.cfg['out_resize'] 
                                  if self.cfg['out_resize'] is not None else 
                                  self.cfg['frame_size_ls_line'][-1])
                    ds = xru.resize_reduce_n_(ds, n=int(ds.attrs['grid_size'] / out_resize[0]))
                    # (sample, iter_infer_num, c, x/h, y/w) -> (list) len: iter_infer_num; shape: c, x/h, y/w
                    prediction = [ i for i in ds['prediction_velocity'].values[0] ]
                    target = [ i for i in ds['target_velocity'].values[0] ]
                    dx, dy, dt = ds.attrs['dx'], ds.attrs['dx'], ds.attrs['dt']
                    ds.close()
                else:
                    # forward
                    [initial_frame, prediction, target, dx_unresized, dy_unresized, dx, dy, dt] = self.one_sample_continuously_infer(
                        idx_init=idx_init, 
                        only_numerical=only_numerical, 
                        frame_size=frame_size, 
                        dt_multiple=dt_multiple,
                        continuously_infer_num=1, 
                        cutout=0.0,
                        noise=0.0,
                        loader=loader,
                        file_name=file_name, 
                        )
                    del initial_frame, dt
                # calculate vorticity & record (prediction)
                results_one_model += [get_vorticity( # ls_shape: dt; shape: h, w
                    prediction[0][None, ...], dx, dy, 
                    boundary_mode="newman", boundary_value=0, 
                    mandatory_use_cuda=False, return_np=True)[0, 0, ...]]
                del prediction # Clean up the cache
                # calculate vorticity & record (target) 
                if model_idx == 0:
                    vorticity_target += [get_vorticity( # ls_shape: dt; item_shape: h, w
                        target[0][None, ...], dx, dy, 
                        boundary_mode="newman", boundary_value=0, 
                        mandatory_use_cuda=False, return_np=True)[0, 0, ...]]
                del target, dx, dy # Clean up the cache
            # record (prediction)
            results += [results_one_model] # ls_shape: model, dt; item_shape: h, w
        
        # prepare data for plot
        data = np.array( 
            [vorticity_target] + results 
            )
        shape = data.shape
        # shape: model, dt, h, w -> idx(model*dt), h, w
        data = data.reshape(-1, shape[-2], shape[-1])
        
        ### plot
        plot_accuracy_heat_or_contour(
            data,
            top_title=top_title,
            left_title=left_title,
            save_dir=save_dir,
            figname=figname,
            figsize=figsize,
            contour=self.cfg['contour'], # heat or contour
            contourf=self.cfg['contourf'], # plot contourf when contour=True
            # cmap=seaborn.cm.icefire,
            )
        del data # Clean up the cache
    
    
    def accuracy_vorticity_correlation(
            self,
            save_dir: Path = None,
            figname: str = "accuracy_correlation.png",
            figsize: Tuple[float, ...] = None,
            line_label: Tuple[str, ...] = None,
            legend: bool = False,
            loader=None, # specify the dataset
            generalization=False, # use cfg of generalization validation, not accuracy
            ):
        """ predict (+ calculate + metrics) + plot """
        save_dir = self.save_dir if save_dir is None else save_dir
        line_label = self.cfg['model_name_ls_line'] if line_label is None else line_label 
        figsize = (7 * 0.8, 7) if figsize is None else figsize
        
        LOGGER.info(f"Run {colorstr('green', 'bold', 'accuracy_vorticity_correlation')}...\n")
        
        ### predict + get metrics
        results = []
        for model_idx, (frame_size, only_numerical) in enumerate(zip(
                self.cfg['frame_size_ls_line'],
                self.cfg['only_numerical_ls_line'])): 
            results_one_model = []
            dt_ls = []
            for dt_multiple in (self.cfg['dt_multiple_ls_acc_line'] if not generalization else 
                                self.cfg['dt_multiple_ls_gnr_line']):
                file_name = (
                    f"{Path(figname).with_suffix('').name}_{'DS' if only_numerical else 'LaPON'}_"
                    f"{frame_size[0]}x{frame_size[1]}_{dt_multiple}dt.nc"
                    )
                    
                if (self.save_dir_results / file_name).exists() and not self.cfg['recal_results']:
                    # Get vorticity correlation between prediction and target on a existing xr.Dataset
                    ds = xr.open_dataset(self.save_dir_results / file_name) 
                    [vc_mean, vc_std, vc_p1] = self.get_vc_on_ds(ds, out_resize=None) # out_resize see cfg
                    dx, dy, dt = ds.attrs['dx'], ds.attrs['dx'], ds.attrs['dt']
                    ds.close()
                else:
                    # eval
                    [vc_mean, vc_std, vc_p1, r2_mean, r2_std, r2_p1, dx, dy, dx_resized, dy_resized, dt] = self.eval_loop(
                        only_numerical=only_numerical, 
                        frame_size=frame_size, 
                        dt_multiple=dt_multiple,
                        continuously_infer_num=1, # 1 means no iter, just infer once
                        cutout=0.0, # cutout probability
                        noise=0.0, # noise level
                        loader=loader,
                        file_name=file_name, 
                        )
                # record
                results_one_model += [vc_mean[-1]]
                dt_ls += [
                    np.round(dt / 1e-3 # 1e-3: corresponding to xlabel beblow
                             , 1)
                    ]
            # record 
            results += [results_one_model] # ls_shape: model, dt
        
        # prepare data for plot
        data = np.array(results) # shape: model, dt
        
        ### plot
        xvline = (self.cfg['xvline_acc_line'] if not generalization else 
            self.cfg['xvline_gnr_line']) 
        plot_accuracy_correlation(
            dt_ls,
            data,
            line_label=line_label,
            xvline=xvline / 1e-3 if xvline is not None else None, # 1e-3: corresponding to xlabel beblow
            yscale_log=(self.cfg['yscale_log_acc_line'] if not generalization else 
                        self.cfg['yscale_log_gnr_line']),
            xlabel='The size of $\Delta t$ (×$10^{-3}$ s)',
            save_dir=save_dir,
            figname=figname,
            figsize=figsize,
            legend=legend,
            )
        
    
    def accuracy_energy_spectrum(
            self,
            save_dir: Path = None,
            figname: str = "accuracy_spectrum.png",
            figsize: Tuple[float, ...] = None,
            line_label: Tuple[str, ...] = None,
            legend: bool = True,
            loader=None, # specify the dataset
            generalization=False, # use cfg of generalization validation, not accuracy
            ):
        """ predict (+ calculate + metrics) + plot """
        save_dir = self.save_dir if save_dir is None else save_dir
        line_label = self.cfg['model_name_ls_line'] + ['Target'] if line_label is None else line_label 
        figsize = (7 * 0.8, 7) if figsize is None else figsize
        
        LOGGER.info(f"Run {colorstr('green', 'bold', 'accuracy_energy_spectrum')}...\n")
        
        ### predict + calculate energy spectrum
        idx_init = (self.cfg['one_sample_idx_acc_spec'] if not generalization else 
                    self.cfg['one_sample_idx_gnr_spec'])
        results = []
        for model_idx, (frame_size, only_numerical) in enumerate(zip(
                self.cfg['frame_size_ls_line'],
                self.cfg['only_numerical_ls_line'])): 
            file_name = (
                f"{Path(figname).with_suffix('').name}_{'DS' if only_numerical else 'LaPON'}_"
                f"{frame_size[0]}x{frame_size[1]}.nc"
                )
            
            if (self.save_dir_results / file_name).exists() and not self.cfg['recal_results']:
                # Get vorticity correlation between prediction and target on a existing xr.Dataset
                ds = xr.open_dataset(self.save_dir_results / file_name) 
                # resize
                out_resize = (self.cfg['out_resize'] 
                              if self.cfg['out_resize'] is not None else 
                              self.cfg['frame_size_ls_line'][-1])
                ds = xru.resize_reduce_n_(ds, n=int(ds.attrs['grid_size'] / out_resize[0]))
                # (sample, iter_infer_num, c, x/h, y/w) -> (list) len: iter_infer_num; shape: c, x/h, y/w
                prediction = [ i for i in ds['prediction_velocity'].values[0] ]
                target = [ i for i in ds['target_velocity'].values[0] ]
                dx, dy, dt = ds.attrs['dx'], ds.attrs['dx'], ds.attrs['dt']
                ds.close()
            else:
                # forward
                [initial_frame, prediction, target, dx_unresized, dy_unresized, dx, dy, dt] = self.one_sample_continuously_infer(
                    idx_init=idx_init, 
                    only_numerical=only_numerical, 
                    frame_size=frame_size, 
                    dt_multiple=(self.cfg['dt_multiple_acc_spec'] if not generalization else 
                                 self.cfg['dt_multiple_gnr_spec']),
                    continuously_infer_num=(
                        self.cfg['continuously_infer_num_acc_spec'] if not generalization else 
                        self.cfg['continuously_infer_num_gnr_spec']),
                    cutout=0.0,
                    noise=0.0,
                    loader=loader,
                    file_name=file_name, 
                    filter_output=(
                        (
                            True if only_numerical else False
                            )
                        if 
                        (self.cfg['continuously_infer_num_acc_spec'] 
                         if not generalization else 
                         self.cfg['continuously_infer_num_gnr_spec'])
                        <= 21 else None), 
                    )
                del initial_frame, dt
            # calculate energy spectrum & record (prediction)
            knyquist, wavenumber, tke_spectrum = get_energy_spectrum2d( 
                u=prediction[-1][0, ...], 
                v=prediction[-1][1, ...], 
                lx=dx*(prediction[-1].shape[2]-1), 
                ly=dy*(prediction[-1].shape[1]-1))
            results += [ [wavenumber, tke_spectrum] ] # ls_shape: model, wavenumber or tke_spectrum
            # clean up the cache
            del prediction
        # calculate energy spectrum & record (target) 
        knyquist, wavenumber, tke_spectrum = get_energy_spectrum2d( 
            u=target[-1][0, ...], 
            v=target[-1][1, ...], 
            lx=dx*(target[-1].shape[2]-1), 
            ly=dy*(target[-1].shape[1]-1))
        results += [ [wavenumber, tke_spectrum] ] # ls_shape: model, wavenumber or tke_spectrum
        # clean up the cache
        del target, dx, dy
        
        # prepare data for plot
        # transpose: model, wavenumber or tke_spectrum -> wavenumber or tke_spectrum, model
        results = list(zip(*results))
        
        ### plot
        plot_accuracy_spectrum(
            results[0],
            results[1],
            line_label=line_label,
            xscale_log=(self.cfg['xscale_log_acc_spec'] if not generalization else 
                        self.cfg['xscale_log_gnr_spec']),
            yscale_log=(self.cfg['yscale_log_acc_spec'] if not generalization else 
                        self.cfg['yscale_log_gnr_spec']),
            ylabel="Energy spectrum $E(\\kappa)$ $(m^3/s^2)$",
            xvline=knyquist, # knyquist: 有效波数截止区间
            save_dir=save_dir,
            figname=figname,
            figsize=figsize,
            legend=legend,
            )
        del results # Clean up the cache
        
    
    def stability_vorticity(
            self,
            save_dir: Path = None,
            figname: str = "stability_heat.png",
            figsize: Tuple[float, ...] = None,
            top_title: Tuple[str, ...] = None,
            left_title: Tuple[str, ...] = None,
            loader=None, # specify the dataset
            ):
        """ predict (+ calculate + metrics) + plot """
        save_dir = self.save_dir if save_dir is None else save_dir
        top_title = self.cfg['dt_label_ls_stb_heat'] if top_title is None else top_title 
        left_title = self.cfg['model_name_ls_heat'] if left_title is None else left_title 
        figsize = (7 * len(top_title) / len(left_title), 7) if figsize is None else figsize
        
        LOGGER.info(f"Run {colorstr('green', 'bold', 'stability_vorticity')}...\n")
        
        ### predict + calculate vorticity
        idx_init = self.cfg['one_sample_idx']
        results = []
        for model_idx, (frame_size, only_numerical) in enumerate(zip(
                self.cfg['frame_size_ls_heat'],
                self.cfg['only_numerical_ls_heat'])): 
            file_name = (
                f"{Path(figname).with_suffix('').name}_{'DS' if only_numerical else 'LaPON'}_"
                f"{frame_size[0]}x{frame_size[1]}.nc"
                )
            
            if (self.save_dir_results / file_name).exists() and not self.cfg['recal_results']:
                # Get vorticity correlation between prediction and target on a existing xr.Dataset
                ds = xr.open_dataset(self.save_dir_results / file_name) 
                # resize
                out_resize = (self.cfg['out_resize'] 
                              if self.cfg['out_resize'] is not None else 
                              self.cfg['frame_size_ls_line'][-1])
                ds = xru.resize_reduce_n_(ds, n=int(ds.attrs['grid_size'] / out_resize[0]))
                # (sample, iter_infer_num, c, x/h, y/w) -> (list) len: iter_infer_num; shape: c, x/h, y/w
                prediction = [ i for i in ds['prediction_velocity'].values[0] ]
                target = [ i for i in ds['target_velocity'].values[0] ]
                dx, dy, dt = ds.attrs['dx'], ds.attrs['dx'], ds.attrs['dt']
                ds.close()
                
                # load initial_frame separately
                with open((self.save_dir_results / file_name).with_suffix('.initial.pkl'), 
                          'rb') as fi: 
                    initial_frame = cPickle.load(fi)
                    
            else:
                # forward
                [initial_frame, prediction, target, dx_unresized, dy_unresized, dx, dy, dt] = self.one_sample_continuously_infer(
                    idx_init=idx_init, 
                    only_numerical=only_numerical, 
                    frame_size=frame_size, 
                    dt_multiple=self.cfg['dt_multiple_ls_stb_heat'],
                    continuously_infer_num=self.cfg['continuously_infer_num_ls_stb_heat'][-1], 
                    cutout=0.0,
                    noise=0.0,
                    loader=loader,
                    file_name=file_name, 
                    )
                
                # save initial_frame separately
                with open((self.save_dir_results / file_name).with_suffix('.initial.pkl'), 
                          'wb') as fi: 
                    cPickle.dump(initial_frame, fi)
            
            # calculate vorticity & record (target) 
            if model_idx == 0:
                results_target = []
                results_target += [get_vorticity( 
                    initial_frame[None, ...], dx, dy, 
                    boundary_mode="newman", boundary_value=0, 
                    mandatory_use_cuda=False, return_np=True)[0, 0, ...]]
                for nt in self.cfg['continuously_infer_num_ls_stb_heat']: # No. of time step
                    results_target += [get_vorticity( 
                        target[nt-1][None, ...], dx, dy, 
                        boundary_mode="newman", boundary_value=0, 
                        mandatory_use_cuda=False, return_np=True)[0, 0, ...]]
                results += [results_target]
                
            results_one_model = []
            # calculate vorticity & record (initial frame) 
            results_one_model += [get_vorticity( 
                initial_frame[None, ...], dx, dy, 
                boundary_mode="newman", boundary_value=0, 
                mandatory_use_cuda=False, return_np=True)[0, 0, ...]]
            for nt in self.cfg['continuously_infer_num_ls_stb_heat']: # No. of time step
                # calculate vorticity & record (prediction)
                results_one_model += [get_vorticity( # shape: h, w
                    prediction[nt-1][None, ...], dx, dy, 
                    boundary_mode="newman", boundary_value=0, 
                    mandatory_use_cuda=False, return_np=True)[0, 0, ...]]
            results += [results_one_model] # ls_shape: model, No. of time step; item_shape: h, w
            
            del initial_frame, prediction, target, dx, dy, dt # Clean up the cache
            
        # prepare data for plot
        data = np.array(results)
        shape = data.shape
        # shape: model, No. of time step, h, w -> idx(model*No. of time step), h, w
        data = data.reshape(-1, shape[-2], shape[-1])
        
        ### plot
        plot_accuracy_heat_or_contour(
            data,
            top_title=top_title,
            left_title=left_title,
            save_dir=save_dir,
            figname=figname,
            figsize=figsize,
            contour=self.cfg['contour'], # heat or contour
            contourf=self.cfg['contourf'], # plot contourf when contour=True
            # cmap=seaborn.cm.icefire,
            )
        del data # Clean up the cache
        
    
    def stability_vorticity_correlation(
            self,
            save_dir: Path = None,
            figname: str = "stability_correlation.png",
            figsize: Tuple[float, ...] = None,
            line_label: Tuple[str, ...] = None,
            legend: bool = False,
            alignment_value_xcoordinate: bool = None, # In order to make the x-coordinate values of multiple subgraphs consistent, the data is revalued
            loader=None, # specify the dataset
            ):
        """ predict (+ calculate + metrics) + plot """
        save_dir = self.save_dir if save_dir is None else save_dir
        line_label = self.cfg['model_name_ls_line'] if line_label is None else line_label 
        figsize = (7 * 0.8, 7) if figsize is None else figsize
        alignment_value_xcoordinate = self.cfg['alignment_value_xcoordinate_stb_line'] if alignment_value_xcoordinate is None else alignment_value_xcoordinate
        
        LOGGER.info(f"Run {colorstr('green', 'bold', 'stability_vorticity_correlation')}...\n")
        
        ### predict + get metrics
        results_vc_mean = []
        results_vc_std = []
        nt_ls = [] # No. of time step
        st_ls = [] # Simulation time (Corresponds to No. of time step)
        for model_idx, (frame_size, only_numerical) in enumerate(zip(
                self.cfg['frame_size_ls_line'],
                self.cfg['only_numerical_ls_line'])): 
            results_one_model_vc_mean = []
            results_one_model_vc_std = []
            nt_ls_one_model = []
            st_ls_one_model = []
            for dt_multiple in self.cfg['dt_multiple_ls_stb_line']:
                file_name = (
                    f"{Path(figname).with_suffix('').name}_{'DS' if only_numerical else 'LaPON'}_"
                    f"{frame_size[0]}x{frame_size[1]}_{dt_multiple}dt.nc"
                    )
                    
                if (self.save_dir_results / file_name).exists() and not self.cfg['recal_results']:
                    # Get vorticity correlation between prediction and target on a existing xr.Dataset
                    ds = xr.open_dataset(self.save_dir_results / file_name) 
                    [vc_mean, vc_std, vc_p1] = self.get_vc_on_ds(ds, out_resize=None) # out_resize see cfg
                    dx, dy, dt = ds.attrs['dx'], ds.attrs['dx'], ds.attrs['dt']
                    ds.close()
                else:
                    # eval
                    [vc_mean, vc_std, vc_p1, r2_mean, r2_std, r2_p1, dx, dy, dx_resized, dy_resized, dt] = self.eval_loop(
                        only_numerical=only_numerical, 
                        frame_size=frame_size, 
                        dt_multiple=dt_multiple,
                        continuously_infer_num=int( # 1 means no iter, just infer once
                            self.cfg['maxdt_continuously_infer_num'] * 
                            self.cfg['dt_multiple_ls_stb_line'][-1][-1] / 
                            dt_multiple[-1]
                            ),
                        cutout=0.0, # cutout probability
                        noise=0.0, # noise level
                        loader=loader,
                        file_name=file_name, 
                        )
                # record
                vc_mean = vc_mean[:int(self.cfg['maxdt_continuously_infer_num'] * self.cfg['dt_multiple_ls_stb_line'][-1][0] / dt_multiple[0])]
                vc_std = vc_std[:int(self.cfg['maxdt_continuously_infer_num'] * self.cfg['dt_multiple_ls_stb_line'][-1][0] / dt_multiple[0])]
                results_one_model_vc_mean += [vc_mean]
                results_one_model_vc_std += [vc_std]
                # record
                nt_ls_one_model += [
                    [nt + 1 for nt in range(vc_mean.shape[0])]
                    ]
                st_ls_one_model += [
                    [np.round((nt+1) * dt / 1e-3 # 1e-3: corresponding to x1label beblow
                              , 1) for nt in range(vc_mean.shape[0])]
                    ]
            # record 
            results_vc_mean += [results_one_model_vc_mean] # ls_shape: model, dt; item_shape: (No. of time step,)
            results_vc_std += [results_one_model_vc_std] # ls_shape: model, dt; item_shape: (No. of time step,)
            nt_ls += [nt_ls_one_model] # ls_shape: model, dt, No. of time step
            st_ls += [st_ls_one_model] # ls_shape: model, dt, Simulation time (Corresponds to No. of time step)
        results_vc_mean = list(zip(*results_vc_mean)) # (transpose) ls_shape: dt, model; item_shape: (No. of time step,)
        results_vc_std = list(zip(*results_vc_std)) # (transpose) ls_shape: dt, model; item_shape: (No. of time step,)
        nt_ls = list(zip(*nt_ls)) # (transpose) ls_shape: dt, model, No. of time step
        st_ls = list(zip(*st_ls)) # (transpose) ls_shape: dt, model, Simulation time (Corresponds to No. of time step)
        
        ### prepare data for plot
        
        # In order to make the x-coordinate values of multiple subgraphs consistent, the data is revalued
        if alignment_value_xcoordinate: 
            for i, dt_multiple in enumerate(self.cfg['dt_multiple_ls_stb_line']):
                idx_interval = int(self.cfg['dt_multiple_ls_stb_line'][-1][-1] / dt_multiple[-1])
                # Only these 'No. of time step' are shown in the plotting 
                results_vc_mean[i] = [r[::-1][::idx_interval][::-1] for r in results_vc_mean[i]]
                results_vc_std[i] = [r[::-1][::idx_interval][::-1] for r in results_vc_std[i]]
                nt_ls[i] = [r[::-1][::idx_interval][::-1] for r in nt_ls[i]]
                st_ls[i] = [r[::-1][::idx_interval][::-1] for r in st_ls[i]]
        
        # ls_shape: dt, model; item_shape: (No. of time step,) 
        # -> ls_shape: dt; item_shape: (model, No. of time step) 
        results_vc_mean = [np.array(r) for r in results_vc_mean]
        results_vc_std = [np.array(r) for r in results_vc_std]
        # ls_shape: dt, model, No. of time step 
        # -> ls_shape: dt; item_shape: (model, No. of time step)
        nt_ls = [np.array(r) for r in nt_ls]
        # ls_shape: dt, model, Simulation time (Corresponds to No. of time step) 
        # -> ls_shape: dt; item_shape: (model, Simulation time (Corresponds to No. of time step))
        st_ls = [np.array(r) for r in st_ls]
        
        # debug
        # self.temp = [results_vc_mean, results_vc_std, nt_ls, st_ls]
        
        ### plot
        plot_stability_correlation(
            st_ls,
            nt_ls,
            results_vc_mean,
            results_vc_std,
            nrows=2,
            ncols=2,
            line_label=line_label,
            x1label='Simulation time ($\\times10^{-3}$ s)',
            x2label='No. of time step',
            ylabel='Vorticity correlation',
            subfig_label = self.cfg['subfig_label_stb_line'],
            xvline=self.cfg['xvline_stb_line'] / 1e-3 if self.cfg['xvline_stb_line'] is not None else None, # 1e-3: corresponding to x1label above
            yhline=0.95,
            yscale_log=self.cfg['yscale_log_stb_line'],
            save_dir=save_dir,
            figname=figname,
            figsize=figsize,
            legend=legend,
            )
    
    
    def stability_noise_immunity(
            self,
            save_dir: Path = None,
            figname: str = "stability_noise_immunity.png",
            figsize: Tuple[float, ...] = None,
            line_label: Tuple[str, ...] = None,
            legend: bool = True,
            loader=None, # specify the dataset
            ):
        """ predict (+ calculate + metrics) + plot """
        save_dir = self.save_dir if save_dir is None else save_dir
        line_label = self.cfg['model_name_ls_line'] if line_label is None else line_label 
        figsize = (7 * 0.8, 7) if figsize is None else figsize
        
        LOGGER.info(f"Run {colorstr('green', 'bold', 'stability_noise_immunity')}...\n")
        
        ### predict + get metrics
        results = []
        for model_idx, (frame_size, only_numerical) in enumerate(zip(
                self.cfg['frame_size_ls_line'],
                self.cfg['only_numerical_ls_line'])): 
            results_one_model = []
            for noise in self.cfg['noise_ls']:
                LOGGER.info('noise'.upper() + f': {noise}')
                file_name = (
                    f"{Path(figname).with_suffix('').name}_{'DS' if only_numerical else 'LaPON'}_"
                    f"{frame_size[0]}x{frame_size[1]}_{noise}noise.nc"
                    )
                    
                if (self.save_dir_results / file_name).exists() and not self.cfg['recal_results']:
                    # Get vorticity correlation between prediction and target on a existing xr.Dataset
                    ds = xr.open_dataset(self.save_dir_results / file_name) 
                    [vc_mean, vc_std, vc_p1] = self.get_vc_on_ds(ds, out_resize=None) # out_resize see cfg
                    dx, dy, dt = ds.attrs['dx'], ds.attrs['dx'], ds.attrs['dt']
                    ds.close()
                else:
                    # eval
                    [vc_mean, vc_std, vc_p1, r2_mean, r2_std, r2_p1, dx, dy, dx_resized, dy_resized, dt] = self.eval_loop(
                        only_numerical=only_numerical, 
                        frame_size=frame_size, 
                        dt_multiple=self.cfg['dt_multiple_stb_nc'],
                        continuously_infer_num=self.cfg['continuously_infer_num_stb_nc'], # 1 means no iter, just infer once
                        cutout=0.0, # cutout probability
                        noise=noise, # noise level
                        loader=loader,
                        file_name=file_name, 
                        )
                # record
                results_one_model += [vc_mean[-1]] 
            # record 
            results += [results_one_model] # ls_shape: model, noise
        
        # prepare data for plot
        data = np.array(results) # shape: model, noise
        noise_ls = np.array(self.cfg['noise_ls']) # shape: noise
        
        ### plot
        plot_stability_noise_immunity(
            noise_ls,
            data,
            line_label=line_label,
            xlabel='Noise (%)',
            ylabel='Vorticity correlation', 
            xscale_log=self.cfg['xscale_log_stb_noise'],
            save_dir=save_dir,
            figname=figname,
            figsize=figsize,
            legend=legend,
            )
        
    
    def stability_data_missing_resistance(
            self,
            save_dir: Path = None,
            figname: str = "stability_data_missing_resistance.png",
            figsize: Tuple[float, ...] = None,
            loader=None, # specify the dataset
            ):
        """ predict (+ calculate + metrics) + plot """
        save_dir = self.save_dir if save_dir is None else save_dir
        sub_titles = ['Original flow field', 'Add noise', 'Locally missing']
        figsize = (7 / 2.2, 7) if figsize is None else figsize
        
        LOGGER.info(f"Run {colorstr('green', 'bold', 'stability_data_missing_resistance')}...\n")
        
        ### predict + get metrics
        results = []
        for model_idx, (frame_size, only_numerical) in enumerate(zip(
                self.cfg['frame_size_ls_line'][-2:],
                self.cfg['only_numerical_ls_line'][-2:])): 
            results_one_model = []
            for cutout in [0.0, 1.0]:
                file_name = (
                    f"{Path(figname).with_suffix('').name}_{'DS' if only_numerical else 'LaPON'}_"
                    f"{frame_size[0]}x{frame_size[1]}_{cutout}cutout.nc"
                    )
                    
                if (self.save_dir_results / file_name).exists() and not self.cfg['recal_results']:
                    # Get vorticity correlation between prediction and target on a existing xr.Dataset
                    ds = xr.open_dataset(self.save_dir_results / file_name) 
                    [vc_mean, vc_std, vc_p1] = self.get_vc_on_ds(ds, out_resize=None) # out_resize see cfg
                    dx, dy, dt = ds.attrs['dx'], ds.attrs['dx'], ds.attrs['dt']
                    ds.close()
                else:
                    # eval
                    [vc_mean, vc_std, vc_p1, r2_mean, r2_std, r2_p1, dx, dy, dx_resized, dy_resized, dt] = self.eval_loop(
                        only_numerical=only_numerical, 
                        frame_size=frame_size, 
                        dt_multiple=self.cfg['dt_multiple_stb_nc'],
                        continuously_infer_num=self.cfg['continuously_infer_num_stb_nc'], # 1 means no iter, just infer once
                        cutout=cutout, # cutout probability
                        noise=0.0, # noise level
                        loader=loader,
                        file_name=file_name, 
                        )
                # record
                results_one_model += [vc_mean[-1]] 
            # record 
            results += [results_one_model] # ls_shape: model, cutout
        
        # prepare data for plot
        data = list(zip(*results)) # (transpose) shape: model, cutout -> cutout, model 
        var1, var2 = data
        
        ### plot
        plot_stability_data_missing_resistance(
            var1,
            var2,
            cluster_label=[
                self.cfg['model_name_ls_line'][-2] + ' ',
                self.cfg['model_name_ls_line'][-1]
                ],
            bar_label=(
                'No data missing',
                'Data missing'),
            ylabel='Vorticity correlation',
            fig_label='(d)',
            save_dir=save_dir,
            figname=figname,
            figsize=figsize,
            )
        
        
    def stability_noise_cutout_vis(
            self,
            save_dir: Path = None,
            figname: str = "stability_noise_cutout_vis.png",
            figsize: Tuple[float, ...] = None,
            loader=None, # specify the dataset
            ):
        """ get data + plot """
        loader = self.loader if loader is None else loader
        save_dir = self.save_dir if save_dir is None else save_dir
        sub_titles = ['Original flow field', 'Add noise', 'Locally missing']
        figsize = (7 / 2.2, 7) if figsize is None else figsize
        
        LOGGER.info(f"Run {colorstr('green', 'bold', 'stability_noise_cutout_vis')}...\n")
        
        ### set loader(dataset) & get data
        loader.dataset.frame_size = (1024, 1024)
        loader.dataset.crop_ratio = [-1, -1]
        loader.dataset.dt_multiple = [1, 1]
        loader.dataset.flipud = 0.0
        loader.dataset.fliplr = 0.0
        
        sample_idx = int(self.cfg['one_sample_idx'])
        
        data = []
        data += [ loader.dataset[sample_idx][0][-1, 0, ...] ] # u: [-1, 0, ...]; v: [-1, 1, ...]
        
        loader.dataset.augment = True
        loader.dataset.noise = 0.35
        loader.dataset.cutout = 0.0
        data += [ loader.dataset[sample_idx][0][-1, 0, ...] ] # u: [-1, 0, ...]; v: [-1, 1, ...]
        
        loader.dataset.noise = 0.0
        loader.dataset.cutout = 1.0
        data += [ loader.dataset[sample_idx][0][-1, 0, ...] ] # u: [-1, 0, ...]; v: [-1, 1, ...]
        
        # prepare data for plot
        data = [d.numpy() for d in data]
        data = np.array(data)
        
        ### plot
        plot_heat_or_contour_vertical(
            data,
            sub_titles=sub_titles,
            save_dir=save_dir,
            figname=figname,
            figsize=figsize,
            contour=False, # heat or contour
            contourf=False, # plot contourf when contour=True
            use_cbar=False, 
            # cmap=seaborn.cm.icefire,
            )
        
        # restore loader state
        loader.dataset.augment = False
        loader.dataset.noise = 0.0
        loader.dataset.cutout = 0.0
    
    
    # @torch.no_grad()
    def one_sample_continuously_infer(
        self,
        idx_init=0, # self.cfg['one_sample_idx']
        only_numerical=False, 
        frame_size=(64, 64), 
        dt_multiple=[1, 1],
        continuously_infer_num=1, # 1 means no iter, just infer once
        cutout=0.0, # cutout probability
        noise=0.0, # noise level
        out_resize=None, # unify the final output frame_size for models of different frame_size for easy comparison
        loader=None, # specify the dataset
        model=None,
        file_name='fig_stability_vc_DS_128x128_1dt.nc', # save prediction results
        clean_cache=True,
        filter_output=None,
        ):
        '''
        NOTE Core method 1/3: infer continuously on one sample for validation, 
        return frames (which are resized to a uniform specified size) etc.
        '''
        model = self.model if model is None else model
        loader = self.loader if loader is None else loader
        out_resize = (self.cfg['out_resize'] 
                      if self.cfg['out_resize'] is not None else 
                      self.cfg['frame_size_ls_line'][-1]
                      ) if out_resize is None else out_resize
        
        ### set loader(dataset) & model
        loader.dataset.frame_size = tuple(frame_size)
        loader.dataset.crop_ratio = [-1, -1]
        loader.dataset.dt_multiple = dt_multiple
        loader.dataset.continuously_infer_num = continuously_infer_num
        if cutout != 0 or noise != 0:
            loader.dataset.augment = True
        loader.dataset.cutout = cutout
        loader.dataset.noise = noise
        
        model = model.to(self.cfg["device"])
        model.eval() 
        # model.no_ni = False # Disable the normalization and inverse normalization of the inputs and outputs of each ANN module
        # model.out_mode = 'normal' # output: next_frame, current_pressure
        # model.out_mode = 'all' # output: Direct output (no inverse norm) of every neural operator etc. (for Neural operators are trained separately)
        # model.operator_cutoff = True # cut off gradient flow of operators (for Neural operators are trained separately)
        # model.no_constrain = False # remove hard condstrain of operator's output
        
        """
        NOTE main process (get data & forward)
        dt_m_cumsum = dt_multiple[0] (represent the end point of the next infer) (where dt_multiple[0] = dt_multiple[1])
        for infer_idx in continuously_infer_num:
            dt_m_cumsum += dt_multiple[0]
            if is first:
                get_init_data
            model.infer ..
            if not last:
                if dt_multiple[0] = dt_multiple[1] is int: (i.e. dt_multiple=[1,1])
                    pass
                elif dt_multiple[0] = dt_multiple[1] < 1: (i.e. dt_multiple=[0.1,0.1])
                    if dt_m_cumsum % 1 != 0: (i.e. dt_m_cumsum=0.8 or 1.2)
                        idx_next_in = dt_m_cumsum // 1 + idx_init
                        loader.dataset.dt_multiple = dt_m_cumsum % 1
                        ..
                    elif dt_m_cumsum % 1 == 0: (i.e. dt_m_cumsum=1 or 3)
                        idx_next_in = dt_m_cumsum // 1 - 1 + idx_init
                        loader.dataset.dt_multiple = 1
                        ..
                replace input (concat frames_in) ..
        """
    
        ### get data & forward (continuously iter infer)
        prediction = []
        target = []
        try:
            model.init4infer_new() # NOTE init for new continuously infer when iter start
        except:
            pass
        dt_m_cumsum = dt_multiple[0] # represent the end point of the next infer #dt[-1] / loader.dataset.origin_dt # cumsum of dt_base's multiple number (loader.dataset.origin_dt = ORIGIN_DT)
        for infer_idx in range(int(continuously_infer_num)):
            dt_m_cumsum += dt_multiple[0] # cumsum of dt_base's multiple number for next 
            
            ###  get initial data
            if infer_idx == 0:
                sample_batch = [ loader.dataset[idx_init] ] #(NOTE from Dataset, not DataLoader)
                sample_batch = loader.dataset.collate_fn(sample_batch) # data preprocess (trans form for model infer)
                
                # data preprocess 
                sample_batch = self.preprocess_batch(sample_batch, False)
                # unpacking
                [frames_in, grid, dt, param, force, pressure, 
                 frame_label_cpu, dfdx_cpu, dfdy_cpu, dfdt_xi_cpu,
                 file, crop_box, ratio, pad] = sample_batch
            
            # # debug
            # print('infer_idx', infer_idx)
            # if infer_idx == 0:
            #     print(idx_init, dt)
            # else:
            #     print(idx_next_in, dt)
            
            ### model forward
            with torch.no_grad():
                outputs = model(
                  frames_in, grid, dt, param, 
                  force,
                  pressure,
                  boundary_mode=self.cfg["pad_mode"], 
                  boundary_value=self.cfg["pad_value"], 
                  boundary_mode_pressure=self.cfg["pad_mode_p"], 
                  boundary_value_pressure=self.cfg["pad_value_p"], 
                  continuously_infer=True, # need to run model.init4infer_new() manually when iter start
                  only_numerical=only_numerical,
                  filter_output=(
                      (infer_idx + 1) % (
                          FILTER_PERIOD[str(tuple(frame_size))] if only_numerical 
                          else FILTER_PERIOD['lapon']) == 0
                      ) if filter_output is None else filter_output, 
                  filter_kernel_size=3, # 3 7
                  gauss_filter=True) 
                if model.out_mode != 'normal':
                    [dfdx_operator_output, dfdy_operator_output, dfdt_xi_operator_output, 
                     dfdx, dfdy, dfdt_xi,
                     next_frame, current_pressure] = outputs
                else:
                    [next_frame, current_pressure] = outputs
            
            ### prepare data for save & output
            if infer_idx == 0:
                dx_unresized = grid[0, -1, 0, 0, 0].item()
                dy_unresized = grid[0, -1, 1, 0, 0].item()
                dt_item = dt[0, 0, -1].item()
                    
            ### Save results to disk
            if self.cfg['save_results_to_disk']:
                self.auto_save_results(
                        file_name=file_name,
                        dx=dx_unresized,
                        dt=dt_item,
                        prediction=next_frame.detach().cpu().numpy()[:, None, ...],
                        target=frame_label_cpu.numpy()[:, None, ...],
                        keys=['prediction_velocity', 'target_velocity'], 
                        cat_dim=1,
                        extra_cat_dim_0=True if infer_idx == continuously_infer_num-1 else False,
                        )
            
            ### outputs resize (unify the final output frame_size) & record (trans shape & device etc.)
            if infer_idx == 0:
                initial_frame = frames_in[0, -1, ...].detach().cpu()
                initial_frame, real_ratio = torch_resize(initial_frame[None, ...], new_shape=out_resize)
                initial_frame = initial_frame[0, ...]
                # updata dx dy
                dx = dx_unresized / real_ratio
                dy = dy_unresized / real_ratio
            prediction += [
                torch_resize(next_frame.detach().cpu(), new_shape=out_resize)[0][0, ...]
                ]
            target += [
                torch_resize(frame_label_cpu, new_shape=out_resize)[0][0, ...]
                ]
            
            ### Clean up the cache
            del [grid, dt, param, force, pressure, 
             frame_label_cpu, dfdx_cpu, dfdy_cpu, dfdt_xi_cpu,
             crop_box, ratio, pad]
            del current_pressure, outputs
            try:
                del [dfdx_operator_output, dfdy_operator_output, dfdt_xi_operator_output, 
                 dfdx, dfdy, dfdt_xi]
            except:
                pass
            
            ### prepare data for next infer ===================================
            if infer_idx != int(continuously_infer_num) - 1:
                if dt_multiple[0] >= 1: # dt_multiple[0] = dt_multiple[1] >= 1
                    label_idx_file = sample_batch[-4][0][-1] # NOTE idx in files: loader.dataset.history_num * loader.dataset.dt_multiple[0] ~ ...
                    label_idx = int(label_idx_file - 
                                 loader.dataset.history_num * 
                                 loader.dataset.dt_multiple[0]) # NOTE idx in dataset: 0 ~ ...
                    idx_next_in = label_idx
                
                else: # dt_multiple[0] = dt_multiple[1] < 1
                    if dt_m_cumsum % 1 != 0:
                        # idx_next_in = int(
                        #         sample_batch[-4][0][-3] - # NOTE current in idx in files
                        #         np.ceil(
                        #             loader.dataset.history_num * 
                        #             loader.dataset.dt_multiple[0])) # NOTE idx in dataset: 0 ~ ...
                        idx_next_in = int(dt_m_cumsum // 1) + idx_init
                        loader.dataset.dt_multiple = [dt_m_cumsum % 1] * 2 # set dt_multiple for label
                    else: 
                        # idx_next_in = int(
                        #         sample_batch[-4][0][-3] - # NOTE current in idx in files
                        #         np.ceil(
                        #             loader.dataset.history_num * 
                        #             loader.dataset.dt_multiple[0]) +  
                        #         1) # NOTE idx in dataset: 0 ~ ...
                        idx_next_in = int(dt_m_cumsum // 1 - 1) + idx_init
                        loader.dataset.dt_multiple = [1] * 2 # NOTE set dt_multiple for label
            
                # NOTE batch-style sample: dataset[idx] & add a new dim (i.e. sample dim # sample -> batch)
                # sample_batch = [ loader.dataset[idx_init + infer_idx * dt_multiple] ] 
                sample_batch = [ loader.dataset[idx_next_in] ] # next sample_batch idx
                loader.dataset.dt_multiple = dt_multiple # NOTE recover loader set 
                sample_batch[-1][2] = sample_batch[-1][2] / sample_batch[-1][2][..., -1] * (dt_multiple[0] * loader.dataset.origin_dt) # NOTE revise dt
                    
                sample_batch = loader.dataset.collate_fn(sample_batch) # data preprocess (trans form for model infer)
                
                # data preprocess 
                sample_batch = self.preprocess_batch(sample_batch, False)
                # unpacking
                [frames_in_, grid, dt, param, force, pressure, 
                 frame_label_cpu, dfdx_cpu, dfdy_cpu, dfdt_xi_cpu,
                 file, crop_box, ratio, pad] = sample_batch
                
                ### NOTE replace input (concat frames_in)
                # frames_in = torch.cat([
                #     frames_in[:, 1:, ...],
                #     next_frame[:, None, ...]
                #     ], dim=1)
                frames_in = torch.cat([
                    frames_in_[:, :-1, ...],
                    next_frame[:, None, ...]
                    ], dim=1)
                
                ### Clean up the cache
                del frames_in_, next_frame
            ### prepare data for next infer END ===============================
            
        # restore loader state
        loader.dataset.augment = False
        loader.dataset.noise = 0.0
        loader.dataset.cutout = 0.0
        
        # Clean up the cache of GPU
        if clean_cache:
            del model
            # self.model = self.model.to('cpu')
            torch.cuda.empty_cache()
        
        # resized output
        return (initial_frame, # (tensor) shape: c, h, w
                prediction, # (list) len: iter_infer_num; shape: c, h, w
                target, # (list) len: iter_infer_num; shape: c, h, w
                dx_unresized, # dx (unresized, used when solving)
                dy_unresized, # dy (unresized, used when solving)
                dx, # dx 
                dy, # dy
                dt_item) # dt
    
    
    # @torch.no_grad()
    def eval_loop_one_by_one(self, # too SLOW!!! Use another one: eval_loop()
                  only_numerical=False, 
                  frame_size=(64, 64), 
                  dt_multiple=[1, 1],
                  continuously_infer_num=1, # 1 means no iter, just infer once
                  cutout=0.0, # cutout probability
                  noise=0.0, # noise level
                  out_resize=None, # unify the final output frame_size for models of different frame_size for easy comparison
                  loader=None, # specify the dataset
                  model=None,
                  file_name='fig_stability_vc_DS_128x128_1dt.nc', # save prediction results
                  filter_output=None,
                  ):
        '''
        NOTE Core method 2/3: validate (allow continuously_infer) on whole dataset (sample loop: one by one), 
        resize to a uniform specified size, then calculalte metrics.
        return metrics(Vorticity correlation & Velocity R2) etc.
        '''
        model = self.model if model is None else model
        loader = self.loader if loader is None else loader
        out_resize = (self.cfg['out_resize'] 
                      if self.cfg['out_resize'] is not None else 
                      self.cfg['frame_size_ls_line'][-1]
                      ) if out_resize is None else out_resize
        
        ### loop on dataset (forward + metrics + record)
        vc = [] # Vorticity correlation
        r2 = [] # Velocity R2
        pbar = tqdm( # progress bar
                range(int( # redefine length of dataset according to experiment parameters of stability validate
                    len(loader.dataset.files) - 
                    (self.cfg['dt_multiple_ls_stb_line'][-1][0] * 
                    self.cfg['maxdt_continuously_infer_num'] + 1)
                    )),
                # total, 
                bar_format=#f"epoch {epoch}/{epochs-1} | Eval"
                f'Eval(only_numerical, frame_size, dt_multiple = {only_numerical, frame_size, dt_multiple}): '
                '{l_bar}{bar:10}{r_bar}{bar:-10b}'
                )
        for idx_init in pbar: 
            ### forward (get data + predict)
            [initial_frame, prediction, target, dx_unresized, dy_unresized, dx, dy, dt] = self.one_sample_continuously_infer(
                idx_init=idx_init, 
                only_numerical=only_numerical, 
                frame_size=frame_size, 
                dt_multiple=dt_multiple,
                continuously_infer_num=continuously_infer_num, 
                cutout=cutout,
                noise=noise,
                out_resize=out_resize,
                loader=loader,
                model=model,
                file_name=file_name,
                clean_cache=False,
                filter_output=filter_output,
                )
            
            ### calculate & record metrics of one sample 
            vc_one_sample = []
            r2_one_sample = []
            for p, t in zip(prediction, target): # loop on "No. of time step"
                # Vorticity correlation
                vc_temp, vc_std_temp_ = get_vorticity_correlation( # vorticity_correlation (batch mean), vorticity_correlation_std (batch std)
                    t[None, ...].to(self.cfg['device']), 
                    p[None, ...].to(self.cfg['device']),
                    dx, dy) 
                vc_one_sample += [vc_temp]
                # Velocity R2
                r2_temp = torch_r2_score(
                    t[None, ...].to(self.cfg['device']), 
                    p[None, ...].to(self.cfg['device']), 
                    reduction='ts mean', channel_reduction=True, 
                    keepdim=False).item() 
                r2_one_sample += [r2_temp]
            vc += [vc_one_sample]
            r2 += [r2_one_sample]
        vc = np.array(vc) # shape: sample, No. of time step
        r2 = np.array(r2) # shape: sample, No. of time step
            
        ### metrics on dataset 
        
        ## Vorticity correlation
        # mean & std  
        vc_mean, vc_std = vc.mean(axis=0), vc.std(axis=0) # shape: (No. of time step,)
        # the 1st percentile 
        vc_p1 = np.percentile(vc, 1, axis=0) # shape: (No. of time step,)
        
        ## Velocity R2
        # mean & std  
        r2_mean, r2_std = r2.mean(axis=0), r2.std(axis=0) # shape: (No. of time step,)
        # the 1st percentile 
        r2_p1 = np.percentile(r2, 1, axis=0) # shape: (No. of time step,)
        
        # Clean up the cache of GPU
        del model
        # self.model = self.model.to('cpu')
        torch.cuda.empty_cache()
            
        return [vc_mean, vc_std, vc_p1, # Vorticity correlation; shape: (No. of time step,)
                r2_mean, r2_std, r2_p1, # Velocity R2; shape: (No. of time step,)
                dx_unresized, dy_unresized, # (unresized, used when solving)
                dx, dy, dt]
    
    
    # @torch.no_grad()
    def eval_loop(self, 
                  only_numerical=False, 
                  frame_size=(64, 64), 
                  dt_multiple=[1, 1],
                  continuously_infer_num=1, # 1 means no iter, just infer once
                  cutout=0.0, # cutout probability
                  noise=0.0, # noise level
                  out_resize=None, # unify the final output frame_size for models of different frame_size for easy comparison
                  loader=None, # specify the dataset
                  model=None,
                  file_name='fig_stability_vc_DS_128x128_1dt.nc', # save prediction results
                  filter_output=None,
                  ):
        '''
        NOTE Core method 3/3: validate (allow continuously_infer) on whole dataset (batch loop), 
        resize to a uniform specified size, then calculalte metrics.
        return metrics(Vorticity correlation & Velocity R2) etc.
        '''
        model = self.model if model is None else model
        
        # Select the batch according to the frame_size and re-establish the loader if loader is None 
        path = self.cfg["data_set_val"] if self.cfg["data_set_val"] else DATA_PATH
        if 'jax' not in path: # jhtdb dataset
            create = create_dataloader
        else:
            create = create_dataloader_jaxcfd
        loader = create(
            path, # data root dir
            frame_size=tuple(frame_size), # preset frame shape; >= 128 if hyp.multi_scale == [0.5,1.5]
            batch_size=self.cfg["batch_val"][str(tuple(frame_size))],
            history_num=self.cfg["history_num_val"], # 8 
            augment=self.cfg["augment_val"], # augmentation (just preset state)
            hyp=self.cfg["hyp"] if self.cfg["hyp"] else HYP_CFG_PATH, # hyperparameters
            rect=self.cfg["rect"], # Rectangular Training(rectangular batches)
            stride=int(self.cfg["stride"]), # stride for Rectangular Training
            pad_mode="newman", #self.cfg["pad_mode"], # boundary condition
            pad_value=0, #self.cfg["pad_value"], # boundary condition
            field_axis=self.cfg["field_axis"], # The axis of the field slice, xy or xz or yz
            workers=self.cfg["workers"], # 8 / 0
            shuffle=False) if loader is None else loader
        self.loader_current_temp = loader # for debug
        
        out_resize = (self.cfg['out_resize'] 
                      if self.cfg['out_resize'] is not None else 
                      self.cfg['frame_size_ls_line'][-1]
                      ) if out_resize is None else out_resize
        
        ### set loader(dataset) & model
        loader.dataset.frame_size = tuple(frame_size)
        loader.dataset.crop_ratio = [-1, -1]
        loader.dataset.dt_multiple = dt_multiple
        loader.dataset.continuously_infer_num = continuously_infer_num
        if cutout != 0 or noise != 0:
            loader.dataset.augment = True
        loader.dataset.cutout = cutout
        loader.dataset.noise = noise
        
        model = model.to(self.cfg["device"])
        model.eval() 
        # model.no_ni = False # Disable the normalization and inverse normalization of the inputs and outputs of each ANN module
        # model.out_mode = 'normal' # output: next_frame, current_pressure
        # model.out_mode = 'all' # output: Direct output (no inverse norm) of every neural operator etc. (for Neural operators are trained separately)
        # model.operator_cutoff = True # cut off gradient flow of operators (for Neural operators are trained separately)
        # model.no_constrain = False # remove hard condstrain of operator's output
        
        ### loop on dataset (forward + metrics + record)
        vc = [] # Vorticity correlation
        r2 = [] # Velocity R2
        pbar = tqdm( # progress bar
                enumerate(loader),
                total=len(loader), 
                bar_format=#f"epoch {epoch}/{epochs-1} | Eval"
                f'Eval(only_numerical, frame_size, dt_multiple = {only_numerical, frame_size, dt_multiple}): '
                '{l_bar}{bar:10}{r_bar}{bar:-10b}'
                )
        for batch_idx, batch in pbar: 
            
            # data preprocess 
            batch = self.preprocess_batch(batch, False)
            
            # unpacking
            [frames_in, grid, dt, param, force, pressure, 
             frame_label_cpu, dfdx_cpu, dfdy_cpu, dfdt_xi_cpu,
             file, crop_box, ratio, pad] = batch
            
            ### forward (predict + metrics + prepare next frame)
            vc_one_batch = []
            r2_one_batch = []
            try:
                model.init4infer_new() # NOTE init for new continuously infer when iter start
            except:
                pass
            for infer_idx in range(int(continuously_infer_num)):
                # model forward
                with torch.no_grad():
                    outputs = model(
                      frames_in, grid, dt, param, 
                      force,
                      pressure,
                      boundary_mode=self.cfg["pad_mode"], 
                      boundary_value=self.cfg["pad_value"], 
                      boundary_mode_pressure=self.cfg["pad_mode_p"], 
                      boundary_value_pressure=self.cfg["pad_value_p"], 
                      continuously_infer=True, # need to run model.init4infer_new() manually when iter start
                      only_numerical=only_numerical,
                      filter_output=(
                          (infer_idx + 1) % (
                              FILTER_PERIOD[str(tuple(frame_size))] if only_numerical 
                              else FILTER_PERIOD['lapon']) == 0
                          ) if filter_output is None else filter_output, 
                      filter_kernel_size=3, # 3 7
                      gauss_filter=True) 
                    if model.out_mode != 'normal':
                        [dfdx_operator_output, dfdy_operator_output, dfdt_xi_operator_output, 
                         dfdx, dfdy, dfdt_xi,
                         next_frame, current_pressure] = outputs
                    else:
                        [next_frame, current_pressure] = outputs
                    
                if infer_idx == 0:
                    dx = grid[0, -1, 0, 0, 0].item()
                    dy = grid[0, -1, 1, 0, 0].item()
                    dt_item = dt[0, 0, -1].item()
                
                # Save results to disk
                if self.cfg['save_results_to_disk']:
                    self.auto_save_results(
                            file_name=file_name,
                            dx=dx,
                            dt=dt_item,
                            prediction=next_frame.detach().cpu().numpy()[:, None, ...],
                            target=frame_label_cpu.numpy()[:, None, ...],
                            keys=['prediction_velocity', 'target_velocity'], 
                            cat_dim=1,
                            extra_cat_dim_0=True if infer_idx == continuously_infer_num-1 else False,
                            )
                        
                ### output resize (unify the final output frame_size)
                # frames_in_resized, real_ratio = torch_resize(frames_in, new_shape=out_resize)
                next_frame_resized, real_ratio = torch_resize(next_frame, new_shape=out_resize)
                frame_label_cpu_resized, real_ratio = torch_resize(frame_label_cpu, new_shape=out_resize)
                # updata dx dy
                if infer_idx == 0:
                    dx_resized = dx / real_ratio
                    dy_resized = dy / real_ratio
                
                ### calculate & record metrics of one batch ##################################################
                
                frame_label_resized = frame_label_cpu_resized.to(self.cfg['device'])
                # Vorticity correlation
                vc_temp = get_vorticity_correlation( # shape: (n,) 
                    frame_label_resized, 
                    next_frame_resized,
                    dx_resized, dy_resized,
                    not_reduce=True) 
                vc_one_batch += [vc_temp] # shape: No. of time step; item_shape: (n,)
                # Velocity R2
                r2_temp = torch_r2_score( # shape:(n, 1)
                    frame_label_resized, 
                    next_frame_resized, 
                    reduction='spatial mean', channel_reduction=True, 
                    keepdim=False)
                r2_temp = r2_temp.detach().cpu().numpy().reshape(-1) # shape:(n,)
                r2_one_batch += [r2_temp] # shape: No. of time step; item_shape: (n,)
                
                ### Clean up the cache
                del frame_label_resized, next_frame_resized, frame_label_cpu_resized
                del [grid, dt, param, force, pressure, 
                 frame_label_cpu, dfdx_cpu, dfdy_cpu, dfdt_xi_cpu,
                 crop_box, ratio, pad]
                del current_pressure, outputs
                try:
                    del [dfdx_operator_output, dfdy_operator_output, dfdt_xi_operator_output, 
                     dfdx, dfdy, dfdt_xi]
                except:
                    pass
                
                ### calculate & record metrics of one batch END ##############################################
                
                ### prepare data (batch) for next infer ##################################################
                # (continuously_infer need to manually load the subsequent data one by one, will be very slow) 
                # Optimization : Duplicate samples avoids reloading
                
                # continuously infer base on dt_m_cumsum(represent the end point of the next infer) (when dt_multiple[0] = dt_multiple[1] < 1)
                if infer_idx == 0:
                    dt_m_cumsum = dt_multiple[0] #dt[-1] / loader.dataset.origin_dt # cumsum of dt_base's multiple number (loader.dataset.origin_dt = ORIGIN_DT)
                
                if infer_idx != (continuously_infer_num - 1):
                    dt_m_cumsum += dt_multiple[0] # cumsum of dt_base's multiple number 
                    
                    # get batch for next infer #############################
                    
                    # batch = []
                    # for f in file: # shape: (n, features)
                    #     label_idx_file = f[-1] # NOTE idx in files: loader.dataset.history_num * loader.dataset.dt_multiple[0] ~ ...
                    #     label_idx = int(label_idx_file - 
                    #                   loader.dataset.history_num * 
                    #                   loader.dataset.dt_multiple[0]) # NOTE idx in dataset: 0 ~ ...
                    #     batch += [ loader.dataset[label_idx] ] # list_shape: (n, features)
                    # batch = loader.dataset.collate_fn(batch) # shape: features, n, ... # data preprocess (trans form for model infer)
                    # # data preprocess 
                    # batch = self.preprocess_batch(batch, False)
                    
                    if dt_multiple[0] >= 1: # dt_multiple[0] = dt_multiple[1] >= 1
                        # Optimization 1: Duplicate samples avoids reloading when dt_multiple[0] >= 1
                        spare_num = dt_multiple[1] # num of not duplicate samples
                        batch_add = []
                        for sample_idx, fi in enumerate(file): # shape: (n, features)
                            if sample_idx >= (batch[0].shape[0] - spare_num): # filter duplicate samples
                                label_idx_file = fi[-1] # NOTE idx in files: loader.dataset.history_num * loader.dataset.dt_multiple[0] ~ ...
                                label_idx = int(label_idx_file - 
                                             loader.dataset.history_num * 
                                             loader.dataset.dt_multiple[0]) # NOTE idx in dataset: 0 ~ ...
                                idx_next_in = label_idx
                                batch_add += [ loader.dataset[idx_next_in] ] # list_shape: (n, features)
                        batch_add = loader.dataset.collate_fn(batch_add) # shape: features, n, ... # data preprocess (trans form for model infer)
                        # data preprocess 
                        batch_add = self.preprocess_batch(batch_add, False)
                        # concate on sample dim
                        feature_num = len(batch)
                        for feature_idx in range(feature_num):
                            if feature_idx not in [ # Tensor
                                    feature_num-4, feature_num-3, feature_num-2, feature_num-1]:
                                batch[feature_idx] = torch.cat([
                                    batch[feature_idx][spare_num:], 
                                    batch_add[feature_idx]
                                    ], dim=0)
                            else: # lsit
                                batch[feature_idx] = (batch[feature_idx][spare_num:] + 
                                                      list(batch_add[feature_idx]))
                                
                    else: # dt_multiple[0] = dt_multiple[1] < 1
                        batch = []
                        for sample_idx, fi in enumerate(file): # shape: (n, features)
                            if dt_m_cumsum % 1 != 0:
                                # idx_next_in = int(
                                #         fi[-3] - # fi[-3]: current in idx in files
                                #         np.ceil(
                                #             loader.dataset.history_num * 
                                #             loader.dataset.dt_multiple[0])) 
                                idx_next_in = int(dt_m_cumsum // 1) + sample_idx + batch_idx * loader.batch_size
                                loader.dataset.dt_multiple = [dt_m_cumsum % 1] * 2
                            else: 
                                # idx_next_in = int(
                                #         fi[-3] - # fi[-3]: current in idx in files
                                #         np.ceil(
                                #             loader.dataset.history_num * 
                                #             loader.dataset.dt_multiple[0]) +  
                                #         1)
                                idx_next_in = int(dt_m_cumsum // 1 - 1) + sample_idx + batch_idx * loader.batch_size
                                loader.dataset.dt_multiple = [1] * 2
                            
                            batch += [ loader.dataset[idx_next_in] ] # list_shape: (n, features)
                            loader.dataset.dt_multiple = dt_multiple # recover loader set 
                            batch[-1][2] = batch[-1][2] / batch[-1][2][..., -1] * (dt_multiple[0] * loader.dataset.origin_dt) # revise dt
                        batch = loader.dataset.collate_fn(batch) # shape: features, n, ... # data preprocess (trans form for model infer)
                        # data preprocess 
                        batch = self.preprocess_batch(batch, False)
                    
                    # get batch for next infer END #########################
                    
                    # unpacking
                    [frames_in_, grid, dt, param, force, pressure, 
                     frame_label_cpu, dfdx_cpu, dfdy_cpu, dfdt_xi_cpu,
                     file, crop_box, ratio, pad] = batch
                    
                    # replace input (concat frames_in)
                    # frames_in = torch.cat([
                    #     frames_in[:, 1:, ...],
                    #     next_frame[:, None, ...]
                    #     ], dim=1)
                    frames_in = torch.cat([
                        frames_in_[:, :-1, ...],
                        next_frame[:, None, ...]
                        ], dim=1)
                    
                    ### Clean up the cache
                    del frames_in_, next_frame
                    
                    ### prepare data (batch) for next infer END ##################################################
                
            vc += [np.array(vc_one_batch)] # shape: batch; item_shape: (No. of time step, n)
            r2 += [np.array(r2_one_batch)] # shape: batch; item_shape: (No. of time step, n)
        # clean cache
        del batch, loader
        vc = np.concatenate(vc, axis=-1) # shape: (No. of time step, samples)
        r2 = np.concatenate(r2, axis=-1) # shape: (No. of time step, samples)
            
        ### metrics on dataset 
        
        ## Vorticity correlation
        # mean & std  
        vc_mean, vc_std = vc.mean(axis=1), vc.std(axis=1) # shape: (No. of time step,)
        # the 1st percentile 
        vc_p1 = np.percentile(vc, 1, axis=1) # shape: (No. of time step,)
        
        ## Velocity R2
        # mean & std  
        r2_mean, r2_std = r2.mean(axis=1), r2.std(axis=1) # shape: (No. of time step,)
        # the 1st percentile 
        r2_p1 = np.percentile(r2, 1, axis=1) # shape: (No. of time step,)
        
        # Clean up the cache of GPU
        del model
        # self.model = self.model.to('cpu')
        torch.cuda.empty_cache()
        
        return [vc_mean, vc_std, vc_p1, # Vorticity correlation; shape: (No. of time step,)
                r2_mean, r2_std, r2_p1, # Velocity R2; shape: (No. of time step,)
                dx, dy, # (unresized, used when solving)
                dx_resized, dy_resized, dt_item]
    
    
    def preprocess_batch(self, batch, use_multi_scale=False):
        
        # TODO (additional data preprocess) ###################################
        # batch = self.preprocess_batch_trans_small(batch, size=(64, 64), idx_h=0, idx_w=0)
        # batch = self.preprocess_batch_trans_big(batch, stack_num=1)
        # TODO END ############################################################
        
        [frames_in, grid, dt, param, force, pressure, 
         frame_label, dfdx, dfdy, dfdt_xi,
         file, crop_box, ratio, pad] = batch
        
        if use_multi_scale:
            ### multi-scale train for each batch ##################################
            multi_scale = self.hyp["multi_scale"]
            if multi_scale != [1.0, 1.0]:
                ### Calculate Multi-scale ratio ###################################
                ratio = random.uniform(multi_scale[0], multi_scale[1])
                ### Revise ratio ##################################################
                # NOTE Satisfying the final shape is a multiple of stride(usually 32)
                stride = 32
                h, w = frames_in.shape[-2:]
                nh_temp, nw_temp = int(h * ratio), int(w * ratio)
                nh = int(np.ceil(nh_temp / stride) * stride) 
                nw = int(np.ceil(nw_temp / stride) * stride) 
                ratio = nh / h
                ### Resize ########################################################
                frames_in, real_ratio = torch_resize(frames_in, (nh, nw))
                force, _              = torch_resize(force, (nh, nw))
                pressure, _           = torch_resize(pressure, (nh, nw))
                # grid, _               = torch_resize(grid, scale_factor=ratio, mode="nearest") # NOTE it's const, can't use bicubic etc.
                frame_label, _        = torch_resize(frame_label, (nh, nw))
                dfdx, _               = torch_resize(dfdx, (nh, nw))
                dfdy, _               = torch_resize(dfdy, (nh, nw))
                dfdt_xi, _            = torch_resize(dfdt_xi, (nh, nw))
                ### grid needs to be handled separately ###########################
                grid = torch.full(frames_in.shape, grid[0,0,0,0,0])
                ### upgrad grid & pad #############################################
                grid = grid / real_ratio
                pad = [[int(p * real_ratio) for p in pad_s] for pad_s in pad]
        
        batch = [frames_in, grid, dt, param, force, pressure, 
                 frame_label, dfdx, dfdy, dfdt_xi, 
                 file, crop_box, ratio, pad]
        # batch[:-4] = [i.float().to(self.cfg["device"], non_blocking=True) for i in batch[:-4]]
        batch[:-8] = [i.float().to(self.cfg["device"], non_blocking=True) for i in batch[:-8]] # without label
        return batch
        # [frames_in, grid, dt, param, force, pressure, 
        #  frame_label_cpu, dfdx_cpu, dfdy_cpu, dfdt_xi_cpu,
        #  file, crop_box, ratio, pad]
    
    
    def preprocess_batch_trans_small(self, batch, size=(64, 64), idx_h=0, idx_w=0):
        """crop a batch (i.e. no resize sample) to small domain (without resize).
        i.e. 1024 x 1024 -> 64 x 64"""
        [frames_in, grid, dt, param, force, pressure, 
         frame_label, dfdx, dfdy, dfdt_xi,
         file, crop_box, ratio, pad] = batch
        
        # i1, j1, i2, j2: left-top point, right-bottom point 
        crop_box = idx_h, idx_w, idx_h + size[0] - 1, idx_w + size[1] - 1
        
        [frames_in, grid, 
         force, pressure, 
         frame_label, 
         dfdx, dfdy, dfdt_xi] = [
             LoadJHTDB.crop_domain(frame, crop_box)
             for frame in [frames_in, grid, 
                           force, pressure, 
                           frame_label, 
                           dfdx, dfdy, dfdt_xi]
             ]
        
        batch = [frames_in, grid, dt, param, force, pressure, 
                 frame_label, dfdx, dfdy, dfdt_xi, 
                 file, crop_box, ratio, pad] # crop_box, ratio, pad maybe invalid
        return batch
    
    
    def preprocess_batch_trans_big(self, batch, stack_num=1):
        """stack a batch (i.e. no resize sample) to big domain (without resize).
        i.e. 1024 x 1024 -> 2048 x 2048"""
        [frames_in, grid, dt, param, force, pressure, 
         frame_label, dfdx, dfdy, dfdt_xi,
         file, crop_box, ratio, pad] = batch
        
        for i in range(stack_num):
            [frames_in, grid, 
             force, pressure, 
             frame_label, 
             dfdx, dfdy, dfdt_xi] = [
                 torch.cat([
                     torch.cat([frame, frame], dim=-2), 
                     torch.cat([frame, frame], dim=-2)
                     ], dim=-1)
                 for frame in [frames_in, grid, 
                               force, pressure, 
                               frame_label, 
                               dfdx, dfdy, dfdt_xi]
                 ]
        
        batch = [frames_in, grid, dt, param, force, pressure, 
                 frame_label, dfdx, dfdy, dfdt_xi, 
                 file, crop_box, ratio, pad] # crop_box, ratio, pad maybe invalid
        return batch
    
    
    def preprocess_label4operator(self, dfdx, dfdy, dfdt_xi):
        # norm label if model use "norm&inverse for ANN module"
        
        label_dfdx, label_dfdy, label_dfdt_xi = (
            dfdx.to(self.cfg['device']), 
            dfdy.to(self.cfg['device']), 
            dfdt_xi.to(self.cfg['device'])
            )
        
        if not self.model.no_ni:
            self.model.eval() 
            label_dfdx = self.model.pde_module.spatial_module.spatial_operator.bni_dfdx(label_dfdx)
            label_dfdy = self.model.pde_module.spatial_module.spatial_operator.bni_dfdy(label_dfdy)
            label_dfdt_xi = self.model.t_operator.bni_dfdt(label_dfdt_xi)
            self.model.train()
        
        return label_dfdx, label_dfdy, label_dfdt_xi 
        
