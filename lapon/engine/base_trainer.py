# LaPON, GPL-3.0 license
# Base trainer for LaPON

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


#%%
class BaseTrainer:
    # LaPON base trainer engine
    """
    BaseTrainer.

    A base class for creating trainers.

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
        
        exist_ok = True if self.cfg["resume"] else False
        self.save_dir = increment_path(Path(self.cfg["project"]) / self.cfg["name"],
                                       exist_ok=exist_ok, mkdir=True)
        
        self.wdir = self.save_dir / 'weights'  # weights dir
        self.wdir.mkdir(parents=True, exist_ok=True)  # make dir
        self.last, self.best = self.wdir / 'last.pt', self.wdir / 'best.pt'
        self.csv = self.save_dir / 'results.csv'
        
        # set LOGGER
        log_dir = self.save_dir / "log_run"
        log_dir.mkdir(parents=True, exist_ok=True)  # make dir
        file_handler = logging.FileHandler(log_dir / "logfile.log")
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(
            logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
        LOGGER.addHandler(file_handler)
        
        # check cfg
        if self.cfg["resume"] and self.cfg["model_weights"]:
            LOGGER.info(f"{colorstr('black', 'bold', 'CHECK CFG')}: 'resume' and 'model_weights' exist at the same time, ignore model_weights." + '\n')
        
        # The main preparations - model
        self.setup_model()
        
        # Other utils init
        self.start_epoch = 0
        self.ema = None # EMA
        self.loss_fn = None # loss function
        self.lf = None # learning rate scheduler function (self.cfg['lrf']: final OneCycleLR learning rate (lr0 * lrf))
        self.scheduler = None 
        self.scaler = None # for AMP
        
        # Epoch level metrics
        self.best_metrics = -10.0
        self.best_epoch = None

        # Save run settings
        with open(self.save_dir / 'cfg.yaml', 'w') as f:
            yaml.safe_dump(self.cfg, f, sort_keys=False)
        with open(self.save_dir / 'hyp.yaml', 'w') as f:
            yaml.safe_dump(self.hyp, f, sort_keys=False)
        
        # Log
        LOGGER.info(f'Training/Validation with {colorstr("cfg")}: \n' + ', '.join(f'{k}={v}' for k, v in self.cfg.items()) +
                    f'\nTraining/Validation with {colorstr("hyp")}: \n' + ', '.join(f'{k}={v}' for k, v in self.hyp.items()) + '\n')
            
        # TensorBoard
        tb_path = self.save_dir
        # tb_path.mkdir(parents=True, exist_ok=True)  # make dir
        LOGGER.info(f"{colorstr('TensorBoard: ')}Start with 'tensorboard --logdir " + f'"{tb_path.parent.parent.resolve()}"' + "', view at http://localhost:6006/" + '\n')
        self.tb = SummaryWriter(str(tb_path)) 
        
    
    def update_saved_cfg_and_hyp_file(self, save_dir=None):
        """ Save current cfg & hyp to file (update saved file or save new file).
        It will run automatically in _do_fit()."""
        save_dir = self.save_dir if save_dir is None else save_dir
        # Save run settings
        with open(self.save_dir / 'cfg.yaml', 'w') as f:
            yaml.safe_dump(self.cfg, f, sort_keys=False)
        with open(self.save_dir / 'hyp.yaml', 'w') as f:
            yaml.safe_dump(self.hyp, f, sort_keys=False)
    
    
    def fit(self): # fit = train + val

        # The main preparations - train (Builds loss, optimizer, scheduler and dataloaders etc.)
        self._setup_train() # include resume
        
        # clean cache
        try:
            del self.ckpt
        except:
            pass

        self._do_fit()
    
    
    def _do_fit(self): # fit = train + val
        """Train completed, evaluate and plot if specified by arguments."""
        
        # save current cfg & hyp to file (again)
        self.update_saved_cfg_and_hyp_file()
        
        # Start training
        self.train_time_start = time.time()
        self.last_opt_step = -1 # last optimize step
        
        # Log 
        LOGGER.info(f'Frame sizes {self.cfg["framesz"]} train, {self.cfg["framesz"]} val\n'
                    f'Using {self.train_loader.num_workers} dataloader workers\n'
                    f"Logging results to {colorstr('bold', self.save_dir)}\n"
                    f'Starting training for {self.cfg["epochs"]} epochs...' + '\n')
        
        # fitting -------------------------------------------------------------
        self.optimizer.zero_grad()  # zero any resumed gradients to ensure stability on train start
        for self.epoch in range(self.start_epoch, self.cfg["epochs"]):
            
            ### fit for one epoch (Warmup, AMP, EMA, minibatch accumulate)
            ### train
            mloss_train = self.train_loop()
            
            ### Scheduler 
            self.lr_all = [x['lr'] for x in self.optimizer.param_groups]  # for loggers
            # lr = [self.lr_all[-1]]
            lr = self.lr_all
            self.scheduler.step() # updata next lr in optimizer
            
            ### EMA update_attr
            self.ema.update_attr(self.model, include=["yaml", "args", "names", "stride"]) 
            
            ### val 
            mloss_val, mmetrics_val = self.eval_loop()
            # [mr2_val, mvc_val, pseudo_vcs_val]
            
            ### Update best metrics 
            mr2_val, mvc_val, pseudo_vcs_val = mmetrics_val 
            self.metrics = mvc_val
            if self.metrics > self.best_metrics:
                self.best_metrics = self.metrics
                self.best_epoch = self.epoch
            
            ### Save log 
            vals = mloss_train + mloss_val + mmetrics_val + lr 
            info = dict(zip(['train/loss', 'val/loss', 
                             'val/r2', 'val/VorticityCorrelation', 'val/VC_pseudo_std', 
                             ] + [f'lr/{ii}' for ii in range(len(lr))], 
                            vals)) 
            # csv
            n = len(info) + 1  # number of cols
            s = '' if self.csv.exists() else (('%23s,' * n % tuple(['epoch'] + list(info.keys()))).rstrip(',') + '\n')  # add header
            with open(self.csv, 'a') as f:
                f.write(s + ('%23.5g,' * n % tuple([self.epoch] + vals)).rstrip(',') + '\n')
            # TensorBoard
            for k, v in info.items():
                self.tb.add_scalar(k, v, self.epoch)
            
            ### Save model (last, best and epoch_i ckpt)
            self.auto_save_model()
            
            ### Save figs (last, best and epoch_i) (disk & TensorBoard)
            if self.cfg["save_period_fig"] >= 0:
                self.auto_save_fig()
            
            ### Stop
            if self.epoch != self.cfg["epochs"] - 1:
                if self.stopper(self.epoch, self.metrics): 
                    LOGGER.info(f'Early stop triggered, current epoch={self.epoch}.' + '\n')
                    break
                if self.cfg['time'] and (time.time() - self.train_time_start) > (self.cfg['time'] * 3600):
                    LOGGER.info(f'Training timed out, stop automatically, current epoch={self.epoch}.' + '\n')
                    break
            
        # end fitting ---------------------------------------------------------
        
        #  Log 
        print("✅ ", end='')
        LOGGER.info(colorstr('Fit finish:\n') + 
                    f'{self.epoch - self.start_epoch + 1} epochs completed in {(time.time() - self.train_time_start) / 3600:.3f} hours.\n'
                    f'The best epoch is {self.best_epoch}, and the val_metrics is {self.best_metrics}.' + '\n')
        
        # TensorBoard
        # self.tb.add_image(tag, img_tensor, global_step=self.epoch, dataformats='HWC')
        
        # Clean up the cache of GPU
        torch.cuda.empty_cache()

    
    def setup_model(self, evaluation=False):
        """Create (& load) model."""
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
        
        # self.model.pde_module.eval() # with operator_weights = [0, 0, 1.0], for t_operator train only
        self.model.no_ni = True # Disable the normalization and inverse normalization of the inputs and outputs of each ANN module
        self.model.out_mode = 'normal' # output: next_frame, current_pressure
        # self.model.out_mode = 'all' # output: Direct output (no inverse norm) of every neural operator etc. (for Neural operators are trained separately)
        self.model.operator_cutoff = False # cut off gradient flow of operators (for Neural operators are trained separately)
        self.model.no_constrain = False # remove hard condstrain of operator's output
        

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
        elif self.cfg["resume"]: # resume
            if not ckpt_keys <= set(self.ckpt.keys()):
                LOGGER.error(f"ERROR ❌ the ckpt {colorstr('red', 'bold', ckpt_path)} "
                             f"is {colorstr('red', 'bold', 'not')} the expected dict.\n"
                             f"The expected keys of ckpt dict must contain {colorstr('red', 'bold', ckpt_keys)},\n"
                             f"but the loaded ckpt file is {colorstr('red', 'bold', set(self.ckpt.keys()))}." + '\n')
                raise ValueError
        elif self.cfg["model_weights"]: # pretrain
            if not {'ema', 'model'} & set(self.ckpt.keys()):
                LOGGER.error(f"ERROR ❌ the ckpt {colorstr('red', 'bold', ckpt_path)} "
                             f"is {colorstr('red', 'bold', 'not')} the expected dict.\n"
                             f"The expected keys of ckpt dict must contain {colorstr('red', 'bold', 'ema or model')},\n"
                             f"but the loaded ckpt file is {colorstr('red', 'bold', set(self.ckpt.keys()))}." + '\n')
                raise ValueError
            
        # set model weight (state_dict)
        csd = (self.ckpt.get("ema") or self.ckpt["model"]).float().state_dict()  # FP32 model # Priority EMA
        csd = intersect_dicts(csd, self.model.state_dict())  # intersect
        self.model.load_state_dict(csd, strict=False)  # load
        # clean cache
        del csd

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
    
    
    def auto_save_model(self):
        """Save model training checkpoints (last, best, epoch_i) with additional metadata."""
        # torch.save(self.model.state_dict(), 'model_weight.pt')
        # torch.save(self.model, 'model.pt')
        # torch.save(ckpt, self.last)
        import io

        # Serialize ckpt to a byte buffer once (faster than repeated torch.save() calls)
        buffer = io.BytesIO()
        torch.save(
            {
                "epoch": -1 if self.epoch == self.cfg["epochs"] - 1 else self.epoch, # -1 means fit is finished
                "best_metrics": self.best_metrics,
                # "weight": deepcopy(self.model).half().state_dict(),
                "model": None,  #deepcopy(self.model).half(), # resume and final checkpoints derive from EMA
                "ema": deepcopy(self.ema.ema).half(),
                "updates": self.ema.updates,
                "optimizer": convert_optimizer_state_dict_to_fp16(deepcopy(self.optimizer.state_dict())),
                "date": datetime.now().isoformat(),
            },
            buffer,
        )
        serialized_ckpt = buffer.getvalue()  # get the serialized content to save

        # Save checkpoints
        self.last.write_bytes(serialized_ckpt)  # save last.pt
        if self.best_metrics == self.metrics:
            self.best.write_bytes(serialized_ckpt)  # save best.pt
        if (self.cfg["save_period"] > 0) and (self.epoch >= 0) and (self.epoch % self.cfg["save_period"] == 0):
            (self.wdir / f"epoch{self.epoch}.pt").write_bytes(serialized_ckpt)  # save epoch_i, i.e. 'epoch3.pt'


    def auto_save_fig(self):
        """Save figs (last, best and epoch_i) (disk & TensorBoard)."""

        ### get data & predict
        if not hasattr(self, 'pre_existed_batch'):
            self.pre_existed_batch = next(iter(self.val_loader))
            self.pre_existed_batch = [ i[:1] for i in self.pre_existed_batch ] # one sample
        batch = self.pre_existed_batch
        batch = self.preprocess_batch(batch, False) 
        [frames_in, grid, dt, param, force, pressure, 
         frame_label_cpu, dfdx_cpu, dfdy_cpu, dfdt_xi_cpu,
         file, crop_box, ratio, pad] = batch
        
        model = self.model
        model.eval() 
        with torch.no_grad():
            outputs = model(
              *batch[:4], 
              force,
              pressure,
              boundary_mode=self.cfg["pad_mode"], 
              boundary_value=self.cfg["pad_value"], 
              boundary_mode_pressure=self.cfg["pad_mode_p"], 
              boundary_value_pressure=self.cfg["pad_value_p"], 
              continuously_infer=False,
              only_numerical=False)
            if self.model.out_mode != 'normal':
                [dfdx_operator_output, dfdy_operator_output, dfdt_xi_operator_output, 
                 dfdx, dfdy, dfdt_xi,
                 next_frame, current_pressure] = outputs
            else:
                [next_frame, current_pressure] = outputs
        
        frame, pred, target, dx, dy = [
            frames_in[:, -1, ...].cpu(),
            next_frame.detach().cpu(), 
            frame_label_cpu,
            grid[0, -1, 0, 0, 0].item(),
            grid[0, -1, 1, 0, 0].item()
            ]
        
        ### get vorticity & plot
        omiga_frame, omiga_pred, omiga_target = [
            get_vorticity(
                i, dx, dy, 
                boundary_mode="newman", boundary_value=0, 
                mandatory_use_cuda=False, return_np=True)[0, 0, ...]
            for i in [frame, pred, target]]
        
        fig, axes = plot_heat([omiga_frame, omiga_pred, omiga_target],
                             show_but_noreturn = False)
        
        ### save
        
        # TensorBoard
        if self.cfg['tensor_board_fig']:
            import io
            # Serialize img to a byte buffer once (faster than nomal save)
            buffer = io.BytesIO()
            plt.savefig(buffer, format='png')
            buffer.seek(0) # 将buf的文件读写指针移动到起始位置
            image = Image.open(buffer)
            image_tensor = torch.tensor(np.array(image))
            try: # maybe error: AttributeError: module 'PIL.Image' has no attribute 'ANTIALIAS'
                tag = 'Vorticity heat'
                self.tb.add_image(tag, image_tensor, global_step=self.epoch, 
                                  dataformats='HWC' # 'HWC' 'CHW' 'HW'...
                                  )
            except:
                pass
        
        # disk
        data = [omiga_frame, omiga_pred, omiga_target]
        filename = self.save_dir / "heat_last.png"
        fig.savefig(filename, dpi=109, bbox_inches='tight') # save last
        with open(filename.with_suffix('.pkl'), 'wb') as fi: 
            cPickle.dump(data, fi) # save data
        if self.best_metrics == self.metrics:
            filename = self.save_dir / "heat_best.png"
            fig.savefig(filename, dpi=109, bbox_inches='tight') # save best
            with open(filename.with_suffix('.pkl'), 'wb') as fi: 
                cPickle.dump(data, fi) # save data
        if (self.cfg["save_period_fig"] > 0) and (self.epoch >= 0) and (self.epoch % self.cfg["save_period_fig"] == 0):
            filename = self.save_dir / f"heat_epoch{self.epoch}.png"
            fig.savefig(filename, dpi=109, bbox_inches='tight') # save epoch_i, i.e. 'epoch50'
            with open(filename.with_suffix('.pkl'), 'wb') as fi: 
                cPickle.dump(data, fi) # save data
        
        plt.close()
        
    
    def _setup_train(self):
        """Builds loss, optimizer, scheduler and dataloaders etc. (include resume)"""
        
        # Loss
        self.loss_fn = nn.SmoothL1Loss(reduction='mean', beta=1.0)
        # self.loss_fn = nn.HuberLoss(reduction='mean', delta=1.0) 
        
        # dataloaders
        self.get_dataloader()
        
        # Optimizer
        nbs = self.cfg["nbs"]  # nominal batch size (it will accumulate when batch < nbs)
        self.accumulate = max(round(nbs / self.cfg["batch"]), 1)  # accumulate loss before optimizing
        weight_decay = float(self.cfg['weight_decay']) * self.cfg["batch"] * self.accumulate / nbs  # scale weight_decay
        iterations = math.ceil(len(self.train_loader.dataset) / max(self.cfg["batch"], nbs)) * self.cfg["epochs"]
        self.optimizer = self.build_optimizer(
            model=self.model,
            name=self.cfg['optimizer'],
            lr=float(self.cfg['lr0']),
            momentum=float(self.cfg['momentum']),
            decay=weight_decay,
            iterations=iterations,
        )
        # Scheduler (Scheduler: return a multiplier that adjusts the learning rate)
        self._setup_scheduler()
        # early stop
        self.stopper = EarlyStopping(patience=self.cfg['patience']) 
        
        # EMA
        self.ema = ModelEMA(self.model)
        
        # Resume
        if self.cfg['resume']:
            self.resume_set() 
        
        # Warmup
        self.num_warup = max(round(self.cfg['warmup_epochs'] * len(self.train_loader)), 100)  # number of warmup iterations, max(3 epochs, 100 iterations)
        # Scheduler
        self.scheduler.last_epoch = self.start_epoch - 1
        # Scaler (torch.cuda.amp.GradScaler + torch.cuda.amp.autocast for AMP) 
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.cfg['amp'])
        
    
    def resume_set(self): 
        # hyp
        self.cfg["hyp"] = str(self.save_dir / 'hyp.yaml')  # replace
        with open(self.cfg["hyp"], errors='ignore') as f:
            self.hyp = yaml.safe_load(f)  # replace
            self.train_loader.hyp = self.hyp  # replace
            self.val_loader.hyp = self.hyp  # replace
            
        # Optimizer
        if self.ckpt['optimizer'] is not None:
            self.optimizer.load_state_dict(self.ckpt['optimizer'])
            self.best_metrics = self.ckpt['best_metrics']

        # EMA
        if self.ema and self.ckpt.get('ema'):
            self.ema.ema.load_state_dict(self.ckpt['ema'].float().state_dict())
            self.ema.updates = self.ckpt['updates']

        # Epoch
        assert self.ckpt['epoch'] >= 0, f'ERROR ❌ {self.last} training to {self.cfg["epochs"]} epochs is finished, nothing to resume.'
        self.start_epoch = self.ckpt['epoch'] + 1
        if self.cfg["epochs"] < self.start_epoch:
            LOGGER.info(f'{self.last} has been trained for {self.ckpt["epoch"]} epochs. Fine-tuning for {self.cfg["epochs"]} more epochs.' + '\n')
            self.cfg["epochs"] += self.ckpt['epoch']  # finetune additional epochs
        
        
    def get_dataloader(self):
        """
        Get train, val dataloader.
        """
        path = self.cfg["data_train"] if self.cfg["data_train"] else DATA_PATH
        if 'jax' not in path: # jhtdb dataset
            create = create_dataloader
        else:
            create = create_dataloader_jaxcfd
        self.train_loader = create(
            path, # data root dir
            frame_size=self.cfg["framesz"], # preset frame shape; >= 128 if hyp.multi_scale == [0.5,1.5]
            batch_size=self.cfg["batch"],
            history_num=self.cfg["history_num"], # 8 
            augment=self.cfg["augment"], # augmentation
            hyp=self.cfg["hyp"] if self.cfg["hyp"] else HYP_CFG_PATH, # hyperparameters
            rect=self.cfg["rect"], # Rectangular Training(rectangular batches)
            stride=int(self.cfg["stride"]), # stride for Rectangular Training
            pad_mode="newman", #self.cfg["pad_mode"], # boundary condition
            pad_value=0, #self.cfg["pad_value"], # boundary condition
            field_axis=self.cfg["field_axis"], # The axis of the field slice, xy or xz or yz
            workers=self.cfg["workers"], # 8 / 0
            shuffle=True)
        
        path = self.cfg["data_val"] if self.cfg["data_val"] else DATA_PATH
        if 'jax' not in path: # jhtdb dataset
            create = create_dataloader
        else:
            create = create_dataloader_jaxcfd
        self.val_loader = create(
            path, # data root dir
            frame_size=self.cfg["framesz"], # preset frame shape; >= 128 if hyp.multi_scale == [0.5,1.5]
            batch_size=self.cfg["batch"],
            history_num=self.cfg["history_num"], # 8 
            augment=False, # augmentation
            hyp=self.cfg["hyp"] if self.cfg["hyp"] else HYP_CFG_PATH, # hyperparameters 
            rect=False, # Rectangular Training(rectangular batches)
            stride=int(self.cfg["stride"]), # stride for Rectangular Training
            pad_mode="newman", #self.cfg["pad_mode"], # boundary condition
            pad_value=0, #self.cfg["pad_value"], # boundary condition
            field_axis=self.cfg["field_axis"], # The axis of the field slice, xy or xz or yz
            workers=self.cfg["workers"], # 8 / 0
            shuffle=False)
    
    
    def build_optimizer(self, model, name="auto", lr=0.001, momentum=0.9, decay=1e-5, iterations=1e5):
        """
        Constructs an optimizer for the given model, based on the specified optimizer name, learning rate, momentum,
        weight decay, and number of iterations.

        Args:
            model (torch.nn.Module): The model for which to build an optimizer.
            name (str, optional): The name of the optimizer to use. If 'auto', the optimizer is selected
                based on the number of iterations. Default: 'auto'.
            lr (float, optional): The learning rate for the optimizer. Default: 0.001.
            momentum (float, optional): The momentum factor for the optimizer. Default: 0.9.
            decay (float, optional): The weight decay for the optimizer. Default: 1e-5.
            iterations (float, optional): The number of iterations, which determines the optimizer if
                name is 'auto'. Default: 1e5.

        Returns:
            (torch.optim.Optimizer): The constructed optimizer.
        """

        g = [], [], []  # optimizer parameter groups
        bn = tuple(v for k, v in nn.__dict__.items() if "Norm" in k)  # normalization layers, i.e. BatchNorm2d()
        if name == "auto":
            LOGGER.info(
                f"{colorstr('optimizer:')} 'optimizer=auto' found, "
                f"ignoring 'lr0'={self.cfg['lr0']} and 'momentum'={self.cfg['momentum']} and "
                f"determining best 'optimizer', 'lr0' and 'momentum' automatically... " + '\n'
            )
            rate = 1 # 5 / (4 + nc) in yolov8
            lr_fit = round(0.002 * rate, 6)  # lr0 fit equation to 6 decimal places
            name, lr, momentum = ("SGD", 0.01, 0.9) if iterations > 10000 else ("AdamW", lr_fit, 0.9)
            self.cfg['warmup_bias_lr'] = 0.0  # no higher than 0.01 for Adam

        for module_name, module in model.named_modules():
            for param_name, param in module.named_parameters(recurse=False):
                fullname = f"{module_name}.{param_name}" if module_name else param_name
                if "bias" in fullname:  # g2: bias (no decay)
                    g[2].append(param)
                elif isinstance(module, bn):  # g1: Norm weight (no decay)
                    g[1].append(param)
                else:  # g0: weight (with decay)
                    g[0].append(param)

        # self.optimizer = torch.optim.SGD(self.model.parameters(), 
        #                                  lr=float(self.cfg['lr0']), 
        #                                  momentum=float(self.cfg['momentum']))
        
        if name in {"Adam", "Adamax", "AdamW", "NAdam", "RAdam"}:
            optimizer = getattr(optim, name, optim.Adam)(g[2], lr=lr, betas=(momentum, 0.999), weight_decay=0.0)
        elif name == "RMSProp":
            optimizer = optim.RMSprop(g[2], lr=lr, momentum=momentum)
        elif name == "SGD":
            optimizer = optim.SGD(g[2], lr=lr, momentum=momentum, nesterov=True)
        elif name == 'L-BFGS-B':
            optimizer = optim.LBFGS(g[2], lr=lr) 
        else:
            raise NotImplementedError(
                f"Optimizer '{name}' not found in list of available optimizers "
                "[Adam, AdamW, NAdam, RAdam, RMSProp, SGD, L-BFGS-B, auto]."
            )

        optimizer.add_param_group({"params": g[0], "weight_decay": decay})  # add g0 with weight_decay
        optimizer.add_param_group({"params": g[1], "weight_decay": 0.0})  # add g1 (BatchNorm2d weights)
        LOGGER.info(
            f"{colorstr('optimizer:')} {type(optimizer).__name__}(lr={lr}, momentum={momentum}) with parameter groups "
            f'{len(g[1])} weight(decay=0.0), {len(g[0])} weight(decay={decay}), {len(g[2])} bias(decay=0.0)' + '\n'
        )
        return optimizer
    
    
    def _setup_scheduler(self):
        """Initialize training learning rate scheduler."""
        if self.cfg['cos_lr']:
            self.lf = one_cycle(1, self.cfg['lrf'], self.cfg['epochs'])  # cosine (1->hyp['lrf'])
        else:
            self.lf = lambda x: max(1 - x / self.cfg['epochs'], 0) * (1.0 - self.cfg['lrf']) + self.cfg['lrf']  # linear (1->hyp['lrf'])
        self.scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=self.lf) 
    
    
    def train_loop(self):
        epoch, epochs = self.epoch, self.cfg["epochs"]
        dataloader, model, loss_fn, optimizer = self.train_loader, self.model, self.loss_fn, self.optimizer
        
        # Start training
        model.train() # Set the model to training mode - important for batch normalization and dropout layers
        ns = len(dataloader.dataset) # num of sample
        nb = len(dataloader) # num of batch
        mloss = torch.zeros(1, device=self.cfg["device"])  # mean loss/losses
        LOGGER.info(('\n' + '%11s' * 4) % ('Epoch', 'gpu_mem', 'mean_loss', 'frame_size'))
        pbar = tqdm(enumerate(dataloader), total=nb, 
                    bar_format=#f"Epoch {epoch}/{epochs-1} | train"
                    '{l_bar}{bar:10}{r_bar}{bar:-10b}') # progress bar
        for batch_idx, batch in pbar:
            # batch -----------------------------------------------------------
            batch = self.preprocess_batch(batch, self.cfg["use_multi_scale"]) # standardization & Multi-scale
            [frames_in, grid, dt, param, force, pressure, 
             frame_label_cpu, dfdx_cpu, dfdy_cpu, dfdt_xi_cpu,
             file, crop_box, ratio, pad] = batch
            
            # Warmup (warmup accumulate, lr & momentum)
            ni = batch_idx + nb * epoch  # number integrated batches (since train start)
            if ni <= self.num_warup:
                xi = [0, self.num_warup]  # x interp
                # linear interp for updata accumulate in Warmup
                self.accumulate = max(1, np.interp(ni, xi, [1, self.cfg["nbs"] / self.cfg["batch"]]).round()) 
                for i, x in enumerate(optimizer.param_groups):
                    # bias lr falls from 0.1 to lr0 * lf(epoch), all other lrs rise from 0.0 to lr0 * lf(epoch)
                    x['lr'] = np.interp(ni, xi, [self.cfg['warmup_bias_lr'] if i == 0 else 0.0, x['initial_lr'] * self.lf(epoch)])
                    if 'momentum' in x:
                        x['momentum'] = np.interp(ni, xi, [self.cfg['warmup_momentum'], self.cfg['momentum']])

            # Forward & Loss (AMP)
            with torch.cuda.amp.autocast(self.cfg['amp']): 
                outputs = model(
                  *batch[:4], 
                  force,
                  pressure,
                  boundary_mode=self.cfg["pad_mode"], 
                  boundary_value=self.cfg["pad_value"], 
                  boundary_mode_pressure=self.cfg["pad_mode_p"], 
                  boundary_value_pressure=self.cfg["pad_value_p"], 
                  continuously_infer=False,
                  only_numerical=False)
                if self.model.out_mode != 'normal':
                    [dfdx_operator_output, dfdy_operator_output, dfdt_xi_operator_output, 
                     dfdx, dfdy, dfdt_xi,
                     next_frame, current_pressure] = outputs
                else:
                    [next_frame, current_pressure] = outputs
                frame_label = frame_label_cpu.to(self.cfg['device'])
                loss = loss_fn(next_frame, frame_label) # return a tensor
                loss_item = loss.item() # return a float
    
            # Backward (AMP)
            # loss.backward() # backpropagation
            self.scaler.scale(loss).backward() # backpropagation
            
            # Optimize (& clip gradients & AMP & EMA & minibatch accumulate) 
            if ni - self.last_opt_step >= self.accumulate:
                self.optimizer_step()
                self.last_opt_step = ni
            
            # Log & Pbar
            mloss = (mloss * batch_idx + loss_item) / (batch_idx + 1)  # update mean losses
            mem = f'{torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0:.3g}G'  # (GB)
            pbar.set_description(('%11s' * 2 + '%11.4g' * 2) %
                                 (f'{epoch}/{epochs - 1}', mem, *mloss, frames_in.shape[-1]))
            # pbar.set_postfix(iteration_result=iteration_result) # 更新进度条的后置文本
            # end batch -------------------------------------------------------
        
        return [mloss.item()]
    
    
    def optimizer_step(self):
        """
        Perform a single step of the training optimizer with gradient clipping and EMA update.
        Optimize (AMP) - https://pytorch.org/docs/master/notes/amp_examples.html
        """
        # optimizer.step()
        # optimizer.zero_grad()
        
        self.scaler.unscale_(self.optimizer)  # unscale gradients
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10.0)  # clip gradients
        
        self.scaler.step(self.optimizer) # optimizer.step
        self.scaler.update() # update the scale factor in scaler
        self.optimizer.zero_grad()
        if self.ema:
            self.ema.update(self.model)
    
    
    # @torch.no_grad()
    def eval_loop(self):
        epoch, epochs = self.epoch, self.cfg["epochs"]
        dataloader, model, loss_fn = self.val_loader, self.model, self.loss_fn
        
        model.eval() # Set the model to evaluation mode - important for batch normalization and dropout layers
        nb = len(dataloader) # num of batch
        mloss_val = 0 # mean val loss 
        mr2_val = 0 # mean val R2 
        mvc_val = 0 # mean of val vorticity_correlation
        pseudo_vcs_val = 0 # pseudo std of val vorticity_correlation (Avg of all batch std)
        
        # Evaluating the model with torch.no_grad() ensures that no gradients are computed during test mode
        # also serves to reduce unnecessary gradient computations and memory usage for tensors with requires_grad=True
        with torch.no_grad():
            pbar = tqdm(dataloader, total=nb, 
                        bar_format=#f"epoch {epoch}/{epochs-1} | Eval"
                        'Eval: '
                        '{l_bar}{bar:10}{r_bar}{bar:-10b}') # progress bar
            for batch in pbar:
                batch = self.preprocess_batch(batch, False) # standardization
                [frames_in, grid, dt, param, force, pressure, 
                 frame_label_cpu, dfdx_cpu, dfdy_cpu, dfdt_xi_cpu,
                 file, crop_box, ratio, pad] = batch
                
                outputs = model(
                  *batch[:4], 
                  force,
                  pressure,
                  boundary_mode=self.cfg["pad_mode"], 
                  boundary_value=self.cfg["pad_value"], 
                  boundary_mode_pressure=self.cfg["pad_mode_p"], 
                  boundary_value_pressure=self.cfg["pad_value_p"], 
                  continuously_infer=False,
                  only_numerical=False)
                if self.model.out_mode != 'normal':
                    [dfdx_operator_output, dfdy_operator_output, dfdt_xi_operator_output, 
                     dfdx, dfdy, dfdt_xi,
                     next_frame, current_pressure] = outputs
                else:
                    [next_frame, current_pressure] = outputs
                frame_label = frame_label_cpu.to(self.cfg['device'])
                mloss_val += loss_fn(next_frame, frame_label).item() # return a float
                # metrics (r2)
                mr2_val += torch_r2_score(
                    frame_label, next_frame, 
                    reduction='ts mean', channel_reduction=True, 
                    keepdim=False).item() # return a float
                # metrics (vorticity_correlation)
                vc, vcs = get_vorticity_correlation( # vorticity_correlation (batch mean), vorticity_correlation_std (batch std)
                    frame_label, next_frame,
                    grid[0, -1, 0, 0, 0].item(), grid[0, -1, 1, 0, 0].item()) 
                mvc_val += vc
                pseudo_vcs_val += vcs
    
        mloss_val /= nb # Val Error (Avg loss)
        mr2_val /= nb # Val Metrics (Avg r2)
        mvc_val /= nb # Val Metrics (Avg vorticity_correlation)
        pseudo_vcs_val /= nb # Val Metrics (pseudo std, Avg of all batch std)
        
        # Log 
        LOGGER.info(f"Val Error: \n Avg loss: {mloss_val:>8f} \n"
                    f"Val Metrics: \n Avg r2: {mr2_val:>8f} \n"
                    f"Val Metrics: \n Avg vorticity correlation: {mvc_val:>8f} \n"
                    f"Val Metrics: \n Avg of vorticity correlation std: {pseudo_vcs_val:>8f} \n")
        
        return [mloss_val], [mr2_val, mvc_val, pseudo_vcs_val]
    
    
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
                # NOTE The loss is calculated without considering the padding boundary (pad = dw, dh)(include 2 sides)
        
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
        
