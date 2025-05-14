# LaPON, GPL-3.0 license
# Dataloaders and dataset utils for JHTDB

"""
demond1
    pkl data (np.array) shape: z(1), y(h), x(w), u_xyz/p [or z, y(1), x, u_xyz/p or z, y, x(1), u_xyz/p]

demond2 
    velocity & pressure data storage:
        root
        ├── .*_u_.*
        |   ├── .*1.pkl
        |   ├── .*2.pkl  
        |   ├── .*3.pkl  
        |   └── ...
        └── .*_p_.*
            ├── .*1.pkl
            ├── .*2.pkl
            ├── .*3.pkl
            └── ...
    example:
        lapon/data/demo/isotropic1024coarse
        ├── datadir_u_z1
        |   ├── frame0001.pkl
        |   ├── frame0002.pkl  
        |   ├── frame0003.pkl  
        |   └── ...
        └── datadir_p_z1
            ├── frame0001.pkl
            ├── frame0002.pkl
            ├── frame0003.pkl
            └── ...
"""

import os, glob, random, re, sys
import _pickle as cPickle
from pathlib import Path
from tqdm import tqdm 
import yaml

import numpy as np, matplotlib.pyplot as plt

import torch
from torch.utils.data import DataLoader, Dataset


if __name__ == "__main__":
    sys.path.append(os.path.abspath("../"))
# else:
#     from pathlib import Path
#     FILE = Path(__file__).resolve()
#     ROOT = FILE.parents[0]  # LaPON root directory
#     if str(ROOT) not in sys.path:
#         sys.path.append(str(ROOT))  # add ROOT to PATH
#     ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative
from utils.augmentations import torch_resize, letterbox, cutout, add_noise, spatial_derivative
from cfd.iteration_method import PoissonEqSolver
from cfd.general import fd8, cubic_hermite_interpolate, linear_interpolate
from utils.general import ( 
    DATA_PATH, 
    HYP_CFG_PATH, 
    )


# TODO param of dataset

FRAME_FORMATS = 'pkl'

RESOLUTION_X = 1024
RESOLUTION_T = 5028

ORIGIN_DX = 2*np.pi / (RESOLUTION_X-1) # 0.00614192112138767
ORIGIN_DT = 0.002

RHO = 1.29 # rho_water: 1e3; rho_air: 1.29
DISS = 0.103 # Dissipation (epsilon)
NU = 0.000185 # Viscosity


class LoadJHTDB(Dataset):
    # LaPON train_loader/val_loader, loads frames and label frames for training and validation
    resolution_x = RESOLUTION_X
    resolution_t = RESOLUTION_T
    origin_dx = ORIGIN_DX
    origin_dt = ORIGIN_DT
    rho = RHO
    diss = DISS
    nu = NU

    def __init__(self,
                 path, # like yolov5: dir or dir_ls or txt_file or txt_file_ls (dir: File directories at any level, all files will be traversed recursively)
                 frame_size=640, # preset frame shape(preset_shape_in_DatasetClass); >= 128 if hyp.multi_scale == [0.5,1.5]
                 history_num=1, # 8
                 augment=False, # augmentation
                 hyp=None, # hyperparameters
                 rect=False, # Rectangular Training(rectangular batches)
                 stride=32, # stride for Rectangular Training
                 pad_mode="newman", # boundary condition
                 pad_value=0.0, # boundary condition
                 field_axis="xy", # The axis of the field slice, xy or xz or yz
                 **kwargs
                 ):
        print('='*30, '\nLoadJHTDB NOTE\nOnly one "Time Continuous" dataset (and "Spatial Consistency") is supported, if there are multiple datasets, please create multiple dataloaders!', '\n'+'='*30)
        self.augment = augment
        self.rect = rect
        self.stride = stride
        self.pad_mode = pad_mode
        self.pad_value = pad_value
        
        assert field_axis in ["xy", "xz", "yz"], f'ERROR ❌ field_axis={field_axis} must be in ["xy", "xz", "yz"]'
        self.field_axis = field_axis
        
        assert hyp, 'ERROR ❌ please give hyp parameter'
        if not isinstance(hyp, dict): # str, Path
            with open(hyp, errors='ignore') as f:
                self.hyp = yaml.safe_load(f)  # load hyps dict
            assert self.hyp, f'ERROR ❌ No hyp ({hyp}) found'
        else: # dict
            self.hyp = hyp
        
        ### NOTE Parameters that control the style of the data for Model Validation
        # self.augment = augment # See above 
        # self.frame_size = frame_size # See below 
        self.crop_ratio = self.hyp['crop_ratio']
        self.dt_multiple = self.hyp['dt_multiple']
        self.history_num = history_num
        self.continuously_infer_num = 1 # 1 means no iter, just infer once(Only change the len of the dataset, do not affect the others (the output is still a single label))
        self.cutout = self.hyp['cutout']
        self.noise = self.hyp['noise']
        self.flipud = self.hyp['flipud']
        self.fliplr = self.hyp['fliplr']
        
        self.path = path 
        
        # self.files = glob.glob(os.path.join(self.path, "frame*"))
        # self.files = glob.glob(os.path.join(self.path, "*"))
        # self.files.sort()
        # 'path' can be file directories at any level, all files will be traversed recursively
        try:
            f = []  # frame files
            for p in path if isinstance(path, list) else [path]:
                p = Path(p)  # os-agnostic
                if p.is_dir():  # dir
                    f += glob.glob(str(p / '**' / '*.*'), recursive=True)
                    # f = list(p.rglob('*.*'))  # pathlib
                elif p.is_file():  # txt file
                    with open(p) as t:
                        t = t.read().strip().splitlines()
                        parent = str(p.parent) + os.sep
                        f += [x.replace('./', parent) if x.startswith('./') else x for x in t]  # local to global path
                else:
                    raise Exception(f'{p} does not exist')
            self.files = sorted(x.replace('/', os.sep) for x in f if x.split('.')[-1].lower() in FRAME_FORMATS)
            assert self.files, f'ERROR ❌ No frames found (path={path})'
        except Exception as e:
            raise Exception(f'Error loading data from {path}: {e}\n')
        
        ### pressure & force
        self.files_pressure = [re.findall("(.*)_u_.*", file)[0] + # prefix
                               "_p_" + 
                               re.findall(".*_u_(.*)", file)[0]   # suffix
                               for file in self.files]
        
        self.files_force    = [re.findall("(.*)_u_.*", file)[0] + 
                               "_force_" + 
                               re.findall(".*_u_(.*)", file)[0] 
                               for file in self.files[:-1]] # NOTE drop the last
        
        pressure_dir = os.path.dirname(self.files_pressure[0])
        force_dir = os.path.dirname(self.files_force[0])
        
        if not os.path.exists(pressure_dir):
            # prepare pressure
            from cfd.iteration_method import PoissonEqSolver 
            
            pass
            raise NotImplementedError("prepare pressure function not implemented in LoadJHTDB")
        
        if not os.path.exists(force_dir):
            # prepare force
            print("There is no force dir, the force is being calculated...")
            os.makedirs(force_dir)
            self.prepare_force() # NOTE Calculate force based on the most recent next frame, not based on label
        else:
            # check data
            if not self.check_files(self.files_force):
                print("WARNING ⚠️ The force does not match velocity, "
                      "all frames are being recalculated. "
                      "But It is recommended to MANUALLY remove all files and re-run the program.")
                self.prepare_force()
        
        # check data
        assert self.check_files(self.files_pressure), "ERROR ❌ The pressure does not match velocity, please check!"
        
        # set preset frame shape
        if self.rect: # if Rectangular Training
            assert isinstance(frame_size, int), "ERROR ❌ frame_size (long side) type must be int when rect=True !"
            
            # shape
            _, (h0, w0) = self.load_frame(0)
            
            # temp short side
            if h0 < w0:
                short_side_temp = int(h0 * (frame_size / w0))
            else:
                short_side_temp = int(w0 * (frame_size / h0))

            # Set training image shapes
            final_short_side = int(np.ceil(short_side_temp / stride) * stride) # Satisfying the shape is a multiple of stride
            
            if h0 < w0:
                self.frame_size = (final_short_side, frame_size)
            else:
                self.frame_size = (frame_size, final_short_side)
        else:
            self.frame_size = (frame_size, frame_size) if isinstance(frame_size, int) else frame_size
        
    def __len__(self):
        return (len(self.files) - 
                int(np.ceil(self.history_num * self.dt_multiple[0])) - 
                int(np.ceil(self.continuously_infer_num * self.dt_multiple[0])))
        # NOTE the first and last frame cannot form sample pairs
        # so the sample index is in range [0, self.__len__() - 1]
    
    def __getitem__(self, index):
        # if not 0 <= index <= self.__len__()-1:
        #     raise ValueError(f'index = {index} must be in 0~{self.__len__()-1}')
        
        # hyp
        hyp = self.hyp
        
        ### index map: from sample idx to file idx ############################
        index = index + int(np.ceil(self.history_num * self.dt_multiple[0]))

        ### Select history & label index ######################################
        history_idx_ls, history_dt_m_ls = self.assign_idx(index, self.dt_multiple, history=True)
        label_idx, label_dt_m = self.assign_idx(index, self.dt_multiple, history=False)
        
        ### Load frame (frame0) ###############################################
        ## current
        # velocity
        frame, (h0, w0) = self.load_frame(index)
        # force 
        force = self.load_force(index)
        # pressure
        pressure = self.load_pressure(index)
        
        ## history
        # velocity
        frame_history = []
        for i in range(self.history_num):
            temp, _ = self.load_frame(history_idx_ls[i])
            if history_dt_m_ls[i] < 1: # update(revise) history (Interpolation will be done if dt_m < 1)
                frame_label = linear_interpolate([temp[0, 0, ...], frame[0, 0, ...]], 
                                                 dt=self.origin_dt, 
                                                 t_r=history_dt_m_ls[i])[None, None, ...]
            frame_history += [temp]
        
        ### merge input data (history + current)
        frames_in = np.concatenate(frame_history + [frame], axis=1) 
        
        ## label
        # velocity
        frame_label, _  = self.load_frame(label_idx)
        if label_dt_m < 1: # update(revise) label (Interpolation will be done if dt_m < 1)
            frame_label = linear_interpolate([frame[0, 0, ...], frame_label[0, 0, ...]], 
                                             dt=self.origin_dt, 
                                             t_r=label_dt_m)[None, None, ...]
        
        ### get label4operator (on frame0) ####################################
        # calculate the label for each operator(ANN module) of LaPON
        dfdx, dfdy, dfdt_xi = self.cal_label4operator(
            frame[0, ...], frame_label[0, ...], # shape: b=1, t=1, c, h, w
            dx=self.origin_dx, dy=self.origin_dx, dt=self.origin_dt * label_dt_m, 
            boundary_mode="periodic", boundary_value=0)
        
        ### Crop the solution domain for 1 frame ##############################
        # crop sub-solution domain on origin frame (final_crop_shape = fraction * preset_shape_in_DatasetClass)
        crop_box = [-1, -1]
        if self.crop_ratio != [-1, -1]:
            crop_box = self.set_domain((h0, w0), self.crop_ratio)
            frames_in   = self.crop_domain(frames_in, crop_box)
            frame_label = self.crop_domain(frame_label, crop_box)
            force       = self.crop_domain(force, crop_box)
            pressure    = self.crop_domain(pressure, crop_box)
            
            # label4operator
            dfdx, dfdy, dfdt_xi = (
                self.crop_domain(dfdx, crop_box),
                self.crop_domain(dfdy, crop_box),
                self.crop_domain(dfdt_xi, crop_box))
        
        ### Letterbox(resize & BC padding) ####################################
        # NOTE resize 时 分辨率放大时使用插值完成，分辨率减小时不能插值！而是直接取原值进行降采样！
        shape = self.frame_size  # preset shape
        frames_in, ratio, pad = letterbox(frames_in, shape, pad_mode=self.pad_mode, pad_value=self.pad_value, auto=False)
        frame_label, _, _     = letterbox(frame_label, shape, pad_mode=self.pad_mode, pad_value=self.pad_value, auto=False)
        force, _, _           = letterbox(force, shape, pad_mode=self.pad_mode, pad_value=self.pad_value, auto=False)
        pressure, _, _        = letterbox(pressure, shape, pad_mode=self.pad_mode, pad_value=self.pad_value, auto=False)
        # The loss is calculated without considering the padding boundary
        
        # label4operator
        dfdx, dfdy, dfdt_xi = (
            letterbox(dfdx, shape, pad_mode=self.pad_mode, pad_value=self.pad_value, auto=False)[0],
            letterbox(dfdy, shape, pad_mode=self.pad_mode, pad_value=self.pad_value, auto=False)[0],
            letterbox(dfdt_xi, shape, pad_mode=self.pad_mode, pad_value=self.pad_value, auto=False)[0]
            )
        dfdx, dfdy, dfdt_xi = (dfdx[np.newaxis, ...], 
                               dfdy[np.newaxis, ...], 
                               dfdt_xi[np.newaxis, ...])
        
        ### grid ##############################################################
        dx = self.origin_dx / ratio[0]
        grid = torch.full((1, self.history_num + 1, 2) + shape, dx) # dx & dy
        
        ### dt ################################################################
        dt = torch.Tensor([self.origin_dt * i for i in history_dt_m_ls] + [self.origin_dt * label_dt_m]).reshape(1, 1, -1)
        
        ### param #############################################################
        param = torch.Tensor([self.rho, self.diss, self.nu]).reshape(1, 1, -1)
        
        ### augment ###########################################################
        if self.augment:
            
            # Flip up-down
            if random.random() < self.flipud:
                frames_in   = frames_in[... , ::-1, :]
                force       = force[... , ::-1, :]
                pressure    = pressure[... , ::-1, :]
                frame_label = frame_label[... , ::-1, :]
                dfdx        = dfdx[... , ::-1, :]
                dfdy        = dfdy[... , ::-1, :]
                dfdt_xi     = dfdt_xi[... , ::-1, :]

            # Flip left-right
            if random.random() < self.fliplr:
                frames_in   = frames_in[... , :, ::-1]
                force       = force[... , :, ::-1]
                pressure    = pressure[... , :, ::-1]
                frame_label = frame_label[... , :, ::-1]
                dfdx        = dfdx[... , :, ::-1]
                dfdy        = dfdy[... , :, ::-1]
                dfdt_xi     = dfdt_xi[... , :, ::-1]

            # Cutouts
            frames_in = cutout(frames_in, p=self.cutout)
            force     = cutout(force, p=self.cutout)
            pressure  = cutout(pressure, p=self.cutout)
            
            # Add noise
            frames_in = add_noise(frames_in, level=self.noise)
            force     = add_noise(force, level=self.noise)
            pressure  = add_noise(pressure, level=self.noise)
        
        # NOTE The output sample tensor does not require the batch dimension here!!!
        return [torch.from_numpy(np.ascontiguousarray(frames_in[0,...])).float(), # shape: t, c, h, w
                grid[0,...].float(), # shape: t, c, h, w
                dt[0,...].float(), # shape: c, t
                param[0,...].float(), # shape: c, param
                torch.from_numpy(np.ascontiguousarray(force[0,...])).float(), # shape: c, h, w
                torch.from_numpy(np.ascontiguousarray(pressure[0,...])).float(), # shape: c, h, w
                torch.from_numpy(np.ascontiguousarray(frame_label[0, 0, ...])).float(), # shape: c, h, w
                torch.from_numpy(np.ascontiguousarray(dfdx[0, 0, ...])).float(), # shape: c, h, w
                torch.from_numpy(np.ascontiguousarray(dfdy[0, 0, ...])).float(), # shape: c, h, w
                torch.from_numpy(np.ascontiguousarray(dfdt_xi[0, 0, ...])).float(), # shape: c, h, w
                [self.files[index], index, self.files[label_idx], label_idx], # for train (get label for operator)
                crop_box, ratio, pad] # NOTE for train (get label for operator)
        # .astype("float32")
        
    def load_frame(self, idx, files=None):
        # Loads 1 frame(velocity) from dataset index 'idx', returns (frame0, original hw)
        
        # try:
        #     files = files
        # except:
        #     files = self.files
        files = self.files if files is None else files
        file = files[idx]
        
        # read
        with open(file, "rb") as fi:
            frame0 = cPickle.load(fi) # shape: z(1), y(h), x(w), u_xyz [or z, y(1), x, u_xyz or z, y, x(1), u_xyz]
        assert frame0 is not None, f'ERROR ❌ frame Not Found {frame0}'
        
        # intercept & unification: all condition transform to z(1), x(h), y(w), u_xy form
        if self.field_axis == "xy":
            frame0 = frame0[... , :2].transpose(0, 2, 1, 3)
        elif self.field_axis == "yz":
            frame0 = frame0[... , 1:].transpose(2, 1, 0, 3) # x(1), y(h), z(w), u_yz
        elif self.field_axis == "xz":
            frame0 = frame0[... , ::2].transpose(1, 2, 0, 3) # y(1), x(h), z(w), u_xz
        
        # convert 
        h0, w0 = frame0.shape[1:3]  # origin hw
        # shape: z(1), x(h), y(w), u_xy -> b, t, c(u_xy), h(x), w(y)
        frame0 = frame0.transpose(0, 3, 1, 2).reshape(1, 1, -1, h0, w0)
        
        return frame0, (h0, w0)  # frame0, hw_original

    def load_pressure(self, idx, files_p=None):
        # Loads 1 frame pressure for velocity
        
        # try:
        #     files_p = files_p
        # except:
        #     files_p = self.files_pressure
        files_p = self.files_pressure if files_p is None else files_p
        file = files_p[idx]
        
        # read
        with open(file, "rb") as fi:
            frame0 = cPickle.load(fi) # shape: z(1), y(h), x(w), p [or z, y(1), x, p or z, y, x(1), p]
        assert frame0 is not None, f'ERROR ❌ frame Not Found {frame0}'
        
        # unification: all condition transform to z(1), x(h), y(w), p form
        if self.field_axis == "xy":
            frame0 = frame0.transpose(0, 2, 1, 3)
        elif self.field_axis == "yz":
            frame0 = frame0.transpose(2, 1, 0, 3) # x(1), y(h), z(w), p
        elif self.field_axis == "xz":
            frame0 = frame0.transpose(1, 2, 0, 3) # y(1), x(h), z(w), p
        
        # convert 
        # shape: z(1), x(h), y(w), p -> b, c(p)(1), h(x), w(y)
        frame0 = frame0.transpose(0, 3, 1, 2)
        
        return frame0  # frame0
    
    def load_force(self, idx):
        # Loads 1 frame force for velocity
        
        file = self.files_force[idx]
        
        # read
        with open(file, "rb") as fi:
            frame0 = cPickle.load(fi) # b, c, h, w
        assert frame0 is not None, f'ERROR ❌ frame Not Found {frame0}'
        
        return frame0  # frame0
    
    def set_domain(self, shape, crop_ratio, rectangle=False):
        # Set the solution domain (rectangle) for 1 frame
        # crop sub-solution domain on origin frame (final_crop_shape = fraction * preset_shape_in_DatasetClass)
        h0, w0 = shape # origin shape
        ph, pw = self.frame_size # preset shape
        r1 = random.uniform(crop_ratio[0], crop_ratio[1]) # The size ratio between crop box and origin frame 
        if rectangle:
            r2 = random.uniform(crop_ratio[0], crop_ratio[1]) 
            h, w = min(max(int(ph * r1), 64), h0), min(max(int(pw * r2), 64), w0)
        else:
            h, w = min(max(int(ph * r1), 64), h0), min(max(int(pw * r1), 64), w0)
        i1 = random.randint(0, h0 - h)
        j1 = random.randint(0, w0 - w)
        crop_box = i1, j1, i1+h-1, j1+w-1 # i1, j1: left-top point; i2, j2: right-bottom point 
        return crop_box
    
    @staticmethod
    def crop_domain(frame, crop_box):
        # Crop the solution domain for 1 frame
        # crop sub-solution domain on origin frame (final_crop_shape = fraction * preset_shape_in_DatasetClass)
        i1, j1, i2, j2 = crop_box # i1, j1: left-top point; i2, j2: right-bottom point 
        frame = frame[..., i1:i2+1, j1:j2+1]
        return frame
    
    def assign_idx(self, idx, dt_multiple, history=False):
        # Assign history / label dt multiple for 1 frame
        if history: # history
            # if idx >= dt_multiple[1]:
            #     alternative = list(range(dt_multiple[0], dt_multiple[1] + 1))
            # else:
            #     alternative = list(range(dt_multiple[0], idx + 1))
            
            # alternative = list(range(dt_multiple[0], 
            #                          min(dt_multiple[1],
            #                              idx) + 1))
            # n = sorted(random.sample(alternative, self.history_num), reverse=True) # list
            
            dt_m = []
            length = idx
            for i in range(int(self.history_num)): # From near to far (from back to front) one by one
                dt_m += [ np.random.uniform(
                            dt_multiple[0], 
                            min(dt_multiple[1],
                                length)) ]
                length -= dt_m[-1] # The length of the remaining interval
            dt_m = dt_m[::-1] # From front to back
            n = np.cumsum(dt_m[::-1])[::-1]
            n = [int(np.around(i)) if i >= 1 else i for i in n] # Filter: Integer or decimal (0~1 & int)
            dt_m = - np.diff(list(n) + [0]) # update dt_m
            n = [i if i >= 1 else 1 for i in n] # (Interpolation will be done if n < 1)
            idx_hl = [idx - i for i in n]
        else: # label
            total = len(self.files)
            # if idx <= (total - dt_multiple[1] - 1):
            #     n = random.randint(dt_multiple[0], dt_multiple[1])
            # else:
            #     n = random.randint(dt_multiple[0], total - (idx+1))
            
            # n = random.randint(dt_multiple[0], 
            #                    min(dt_multiple[1],
            #                        total - (idx+1)))
            # NOTE random.randint: left closed and right closed; np.random.randint: left closed and right open; 
            
            dt_m = np.random.uniform(dt_multiple[0], 
                                     min(dt_multiple[1],
                                         total - (idx+1)))
            dt_m = int(np.around(dt_m)) if dt_m >= 1 else dt_m # Filter: Integer or decimal (0~1 & int)
            n = dt_m if dt_m >= 1 else 1 # (Interpolation will be done if dt_m < 1)
            idx_hl = idx + n # label index 
            
        return idx_hl, dt_m # history / label index, multiple number of dt_base
    
    def prepare_force(self, device="cuda"): # precompute force in __init__ stage
        if not torch.cuda.is_available():
            print("preparing force... But cuda is not available. Switch to CPU.")
            device = "cpu"
        
        files_dir = os.path.dirname(self.files[0])
        all_files = glob.glob(os.path.join(files_dir, '*'))
        all_files.sort()
        
        all_files_p = [
            (re.findall("(.*)_u_.*", file)[0] + 
                 "_p_" + 
                 re.findall(".*_u_(.*)", file)[0])
            for file in all_files
            ]
        
        for index, file in tqdm(enumerate(all_files[:-1]), desc="preparing force for whole dir"):
            file_force = (re.findall("(.*)_u_.*", file)[0] + 
                          "_force_" + 
                          re.findall(".*_u_(.*)", file)[0])
            
            # load
            frame, _ = self.load_frame(index, files=all_files) # load_frame
            frame_latter, _ = self.load_frame(index+1, files=all_files) # load_frame
            pressure = self.load_pressure(index, files_p=all_files_p) # load_pressure
            
            # convert
            frame = torch.from_numpy(frame).to(device)
            frame_latter = torch.from_numpy(frame_latter).to(device)
            pressure = torch.from_numpy(pressure).to(device)
            
            # calculate force (It can also be implemented: 3D field to 2D field)
            force = self.get_force(frame, pressure, frame_latter=frame_latter)
            
            # convert
            force = force.detach().cpu().numpy()
            
            # save
            with open(file_force, "wb") as fi:
                cPickle.dump(force, fi) 
    
    @staticmethod
    def get_force(frame, pressure, frame_latter, dx=ORIGIN_DX, dt=ORIGIN_DT, rho=RHO, nu=NU): # calculate 2D field force (It can also be implemented: 3D field to 2D field)
        #============
        fx, fy = frame[:, :, :1, ...], frame[:, :, 1:, ...]
        
        #============
        dfdt = (frame_latter - frame) / dt
        dfxdt, dfydt = dfdt[:, :, :1, ...], dfdt[:, :, 1:, ...]
        
        #============
        ### h(-y), w(x)
        # dfdx = torch.diff(frame, n=1, dim=-1, 
        #                   append=frame[... , :, -1:]) / dx
        # dfxdx, dfydx = dfdx[:, :, :1, ...], dfdx[:, :, 1:, ...]
        
        # dfdy = - torch.diff(frame, n=1, dim=-2, 
        #                     prepend=frame[... , -1:, :]) / dx # NOTE minus
        # dfxdy, dfydy = dfdy[:, :, :1, ...], dfdy[:, :, 1:, ...]
        
        ### h(x), w(y)
        dfdx, dfdy = spatial_derivative(frame, dx, "newman")
        dfxdx, dfydx = dfdx[:, :, :1, ...], dfdx[:, :, 1:, ...]
        dfxdy, dfydy = dfdy[:, :, :1, ...], dfdy[:, :, 1:, ...]
        
        #============
        ### h(-y), w(x)
        # d2fdx2 = torch.diff(dfdx, n=1, dim=-1, 
        #                     append=dfdx[... , :, -1:]) / dx**2
        # d2fxdx2, d2fydx2 = d2fdx2[:, :, :1, ...], d2fdx2[:, :, 1:, ...]
        
        # d2fdy2 = torch.diff(dfdy, n=1, dim=-2, 
        #                     prepend=dfdy[... , -1:, :]) / dx**2
        # d2fxdy2, d2fydy2 = d2fdy2[:, :, :1, ...], d2fdy2[:, :, 1:, ...]
        
        ### h(x), w(y)
        d2fdx2, d2fdxy = spatial_derivative(dfdx, dx, "newman")
        d2fdyx, d2fdy2 = spatial_derivative(dfdy, dx, "newman")
        d2fxdx2, d2fydx2 = d2fdx2[:, :, :1, ...], d2fdx2[:, :, 1:, ...]
        d2fxdy2, d2fydy2 = d2fdy2[:, :, :1, ...], d2fdy2[:, :, 1:, ...]
        
        #============
        ### h(-y), w(x)
        # dpdx = torch.diff(pressure, n=1, dim=-1, 
        #                   append=pressure[... , :, -1:]) / dx
        # dpdy = - torch.diff(pressure, n=1, dim=-2, 
        #                     prepend=pressure[... , -1:, :]) / dx # NOTE minus
        
        ### h(x), w(y)
        dpdx, dpdy = spatial_derivative(pressure, dx, "newman")
        
        #============
        force_x = (
            dfxdt 
            + (fx * dfxdx + fy * dfxdy)
            + 1 / rho
            * dpdx
            - nu
            * (d2fxdx2 + d2fxdy2)
        )
        force_y = (
            dfydt
            + (fx * dfydx + fy * dfydy)
            + 1 / rho
            * dpdy
            - nu
            * (d2fydx2 + d2fydy2)
        )
        force = torch.cat([force_x, force_y], dim=2)
        
        # convert 
        h, w = force.shape[-2:]
        return force.reshape(1, -1, h, w) # b, c, h, w
    
    #=========================================================================#
    
    @staticmethod
    def cal_label4operator(frame, frame_label, dx=ORIGIN_DX, dy=ORIGIN_DX, dt=ORIGIN_DT, 
                           boundary_mode="periodic", boundary_value=0, eps=1e-7): # 1e-5 1e-7
        # calculate the label (only numerical calculation)
        """
        input: np.array
        output: np.array
        
        input_shape: 
            x: batch_size, c=2, h, w
            dx / dy: a scalar
        output_shape(each): 
            batch_size, c=2, h, w
        """
    
        dfdx, dfdy = fd8(frame, dx, dy, 
                         boundary_mode=boundary_mode, boundary_value=boundary_value,
                         return_np=True)
        
        dfdt_xi = (frame_label - frame) / (dt + eps)
        
        return dfdx, dfdy, dfdt_xi 
    
    #=========================================================================#
    
    @staticmethod
    def check_files(files):
        ok = True # flag
        for file in files:
            if not os.path.exists(file):
                if ok:
                    ok = False
                    print("These files do not exist:")
                print(file)
        return ok
    
    @staticmethod
    def collate_fn_base(batch): # NOTE fn for batch (not for one sample) [merges a list of samples to form a mini-batch of Tensor(s)]
        # batch = [ sample1[feature1_Tensor, feature2_Tensor, ...], sample2[...], ... ]
        (frames_in, grid, dt, param, force, pressure, 
         frame_label, dfdx, dfdy, dfdt_xi,
         file, crop_box, ratio, pad) = zip(*batch) # NOTE transposed (shape: b, f -> f, b)

        # NOTE the last output of loader
        return [
            torch.stack(frames_in, 0), 
            torch.stack(grid, 0),
            torch.stack(dt, 0),
            torch.stack(param, 0),
            torch.stack(force, 0),
            torch.stack(pressure, 0),
            torch.stack(frame_label, 0),
            torch.stack(dfdx, 0),
            torch.stack(dfdy, 0),
            torch.stack(dfdt_xi, 0),
            file, 
            crop_box, ratio, pad] 
    
    @staticmethod
    def collate_fn(batch): 
        (frames_in, grid, dt, param, force, pressure, 
         frame_label, dfdx, dfdy, dfdt_xi,
         file, crop_box, ratio, pad) = zip(*batch) # transposed
        
        (frames_in, grid, dt, param, force, pressure, 
         frame_label, dfdx, dfdy, dfdt_xi,
         file, crop_box, ratio, pad) = [
            torch.stack(frames_in, 0), 
            torch.stack(grid, 0),
            torch.stack(dt, 0),
            torch.stack(param, 0),
            torch.stack(force, 0),
            torch.stack(pressure, 0),
            torch.stack(frame_label, 0),
            torch.stack(dfdx, 0),
            torch.stack(dfdy, 0),
            torch.stack(dfdt_xi, 0),
            file, 
            crop_box, ratio, pad] 
        
        ### do something...
        
        # the last output of loader
        return [frames_in, grid, dt, param, force, pressure, 
                frame_label, dfdx, dfdy, dfdt_xi,
                file, crop_box, ratio, pad]


def create_dataloader(path, # data root dir
                      frame_size=640, # preset frame shape; >= 128 if hyp.multi_scale == [0.5,1.5]
                      batch_size=64,
                      history_num=1, # 8
                      augment=False, # augmentation
                      hyp=None, # hyperparameters
                      rect=False, # Rectangular Training(rectangular batches)
                      stride=32, # stride for Rectangular Training
                      pad_mode="newman", # boundary condition
                      pad_value=0.0, # boundary condition
                      field_axis="xy", # The axis of the field slice, xy or xz or yz
                      workers=8, # 8 / 0
                      # prefetch_factor=2,
                      shuffle=True,
                      time_len=None, # Crop part of the time
                      **kwargs):
    dataset = LoadJHTDB(
                path=path, 
                frame_size=frame_size, 
                history_num=history_num, 
                augment=augment,  
                hyp=hyp, 
                rect=rect, 
                stride=int(stride),
                pad_mode=pad_mode, 
                pad_value=pad_value, 
                field_axis=field_axis, 
                time_len=time_len, # Crop part of the time
                )
    
    batch_size = min(batch_size, len(dataset))
    nd = torch.cuda.device_count()  # number of CUDA devices
    nw = min([os.cpu_count() // max(nd, 1), batch_size if batch_size > 1 else 0, workers])  # number of workers
    return DataLoader(dataset,
                batch_size=batch_size,
                shuffle=shuffle,
                num_workers=nw, # 8 / 0
                # prefetch_factor=prefetch_factor,
                drop_last=False,
                sampler=None,
                pin_memory=True,
                collate_fn=LoadJHTDB.collate_fn) # NOTE fn for batch (not for one sample) [merges a list of samples to form a mini-batch of Tensor(s)]

    
