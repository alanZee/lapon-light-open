# LaPON, GPL-3.0 license
# General utils of cfd (It can be used for data processing and model building)

import os, glob, random, re, sys
import _pickle as cPickle
from copy import deepcopy
from pathlib import Path
from tqdm import tqdm 
import logging
import platform

import numpy as np, pandas as pd, matplotlib.pyplot as plt

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
from utils.augmentations import spatial_derivative # base on class SpatialDerivative (ANN module)
from models.common import BoundaryPadding


def np_torch_trans(func):
    # transform np to torch for torch calculating, and return np or torch
    
    def wrapper(frame, *args, **kwargs):
        mandatory_use_cuda = kwargs.get('mandatory_use_cuda', False)
        return_np = kwargs.get('return_np', False)
        
        if not isinstance(frame, torch.Tensor):
            frame = torch.from_numpy(frame)
        if mandatory_use_cuda and torch.cuda.is_available():
            frame = frame.cuda()
        
        results = func(frame, *args, **kwargs)
        
        if return_np:
            if isinstance(results, (list, tuple)):
                return [i.detach().cpu().numpy() for i in results] 
            elif isinstance(results, np.ndarray):
                return results
            else:
                return results.detach().cpu().numpy()
        else:
            return results
    
    return wrapper


def np_torch_trans2(func):
    # transform np to torch for torch calculating, and return np or torch
    
    def wrapper(frame, grid, *args, **kwargs):
        mandatory_use_cuda = kwargs.get('mandatory_use_cuda', False)
        return_np = kwargs.get('return_np', False)
        
        if not isinstance(frame, torch.Tensor):
            frame = torch.from_numpy(frame)
        if not isinstance(grid, torch.Tensor):
            grid  = torch.from_numpy(grid)
        if mandatory_use_cuda and torch.cuda.is_available():
            frame = frame.cuda()
            grid  = grid.cuda()
        
        results = func(frame, grid, *args, **kwargs)
        
        if return_np:
            if isinstance(results, (list, tuple)):
                return [i.detach().cpu().numpy() for i in results] 
            else:
                return results.detach().cpu().numpy()
        else:
            return results
    
    return wrapper


###=====================================================================###

# Cubic Hermite Interpolation
def cubic_hermite_interpolate(
        f: [np.array, ...], # shape: c, h, w (len_ls = 4)
        dt: float = 2e-3,
        t_r: float = 0.25, # t_ratio: 0~1
        ) -> np.array: # shape: c, h, w
    """
    case:
        temp = np.random.rand(3, 64, 64)
        f = [temp + i**2 for i in range(4)]
        dt = 2e-3
        t_r = 0.25
        f_i = cubic_hermite_interpolate(f, dt, t_r=t_r)
        plt.plot([0, dt, (1+t_r)*dt, 2*dt, 3*dt],
                 [f[0][0,0,0], f[1][0,0,0], 
                  f_i[0,0,0],
                  f[2][0,0,0], f[3][0,0,0],
                  ])
    """
    # https://turbulence.pha.jhu.edu/docs/Database-functions.pdf
    t_f = np.arange(0, len(f) * dt, dt) # 0, dt, 2dt, 3dt
    t = (t_r + 1) * dt # 1*dt~2*dt
    
    a = f[1]
    b = (f[2] - f[0]) / (2*dt)
    c = (f[2] - 2 * f[1] + f[0]) / (2*dt**2)
    d = (-1 * f[0] + 3 * f[1] - 3 * f[2] + f[3]) / (2*dt**3)
    
    # f_i = (a + b * (t-t_f[1]) + c * (t-t_f[1])**2 + d * (t-t_f[1])**2 * (t-t_f[2]))
    f_i = (a + b * (t-t_f[1]) + (c + d * (t-t_f[2])) * (t-t_f[1])**2)
    
    return f_i

# Linear Interpolation
def linear_interpolate(
        f: [np.array, ...], # shape: c, h, w (len_ls = 2)
        dt: float = 2e-3,
        t_r: float = 0.25, # t_ratio: 0~1
        ) -> np.array: # shape: c, h, w
    """
    case:
        temp = np.random.rand(3, 64, 64)
        f = [temp + i**2 for i in range(2)]
        dt = 2e-3
        t_r = 0.25
        f_i = linear_interpolate(f, dt, t_r=t_r)
        plt.plot([0, t_r*dt, dt],
                 [f[0][0,0,0], f_i[0,0,0], f[1][0,0,0]])
    """
    t_f = np.arange(0, len(f) * dt, dt) # 0, dt
    t = t_r * dt # 0~1*dt
    
    a = f[0]
    b = (f[1] - f[0]) / dt
    
    f_i = (a + b * (t - t_f[0]))
    
    return f_i


###=====================================================================###

# 2D FD8(High-order precision derivatives): 8th-order centered finite differencing (on uniform grid; 8 points)
@np_torch_trans
def fd8(frame, dx, dy, boundary_mode="periodic", boundary_value=0, 
        mandatory_use_cuda=False, return_np=False):
    # https://turbulence.pha.jhu.edu/docs/Database-functions.pdf
    # With the edge replication of 4 data-points on each side, 
    # this is the highest-order finite difference available.
    """
    input: np.array / torch.Tensor
    output: tuple of np.array / torch.Tensor
    
    input_shape: 
        x: batch_size, c=2, h(x), w(y)
        dx / dy: a scalar
    output_shape(each): 
        batch_size, c=2, h(x), w(y)
    
    case:
        # f  = torch.arange(1, 10*2*64*64+1, dtype=torch.float32).reshape(10, 2, 64, 64).to("cuda")
        f  = torch.randn(10, 2, 64, 64).to("cuda")
        dx = dy = 0.006
        dfdx, dfdy = fd8(f, dx, dy, boundary_mode="periodic", boundary_value=0)
        print(dfdx.shape)
        plt.imshow(f[0,0,...].detach().cpu().numpy())
        plt.show()
        plt.imshow(dfdx[0,0,...].detach().cpu().numpy())
        plt.show()
        plt.imshow(dfdy[0,0,...].detach().cpu().numpy())
        plt.show()
    """
    
    bp = BoundaryPadding()
    if mandatory_use_cuda and torch.cuda.is_available():
        bp = bp.cuda()
        
    frame_pad_all = bp(frame, padding=(4, 4, 4, 4), mode=boundary_mode, value=boundary_value)
    
    idx_l, idx_h = 4, frame.shape[-1] + 8 - 4 # vailid idx low, high
    
    ### h(-y), w(x)
    # frame_pad = frame_pad_all[..., idx_l:idx_h, :]
    # dfdx = (
    #     4/(5*dx) * (frame_pad[..., :, idx_l+1:idx_h+1] - frame_pad[..., :, idx_l-1:idx_h-1])
    #     - 1/(5*dx) * (frame_pad[..., :, idx_l+2:idx_h+2] - frame_pad[..., :, idx_l-2:idx_h-2])
    #     + 4/(105*dx) * (frame_pad[..., :, idx_l+3:idx_h+3] - frame_pad[..., :, idx_l-3:idx_h-3])
    #     - 1/(280*dx) * (frame_pad[..., :, idx_l+4:idx_h+4] - frame_pad[..., :, idx_l-4:idx_h-4])
    #     )
    
    # frame_pad = frame_pad_all[..., :, idx_l:idx_h]
    # dfdy = (
    #     4/(5*dy) * (frame_pad[..., idx_l-1:idx_h-1, :] - frame_pad[..., idx_l+1:idx_h+1, :])
    #     - 1/(5*dy) * (frame_pad[..., idx_l-2:idx_h-2, :] - frame_pad[..., idx_l+2:idx_h+2, :])
    #     + 4/(105*dy) * (frame_pad[..., idx_l-3:idx_h-3, :] - frame_pad[..., idx_l+3:idx_h+3, :])
    #     - 1/(280*dy) * (frame_pad[..., idx_l-4:idx_h-4, :] - frame_pad[..., idx_l+4:idx_h+4, :])
    #     )

    ### h(x), w(y)
    frame_pad = frame_pad_all[..., :, idx_l:idx_h]
    dfdx = (
        4/(5*dx) * (frame_pad[..., idx_l+1:idx_h+1, :] - frame_pad[..., idx_l-1:idx_h-1, :])
        - 1/(5*dx) * (frame_pad[..., idx_l+2:idx_h+2, :] - frame_pad[..., idx_l-2:idx_h-2, :])
        + 4/(105*dx) * (frame_pad[..., idx_l+3:idx_h+3, :] - frame_pad[..., idx_l-3:idx_h-3, :])
        - 1/(280*dx) * (frame_pad[..., idx_l+4:idx_h+4, :] - frame_pad[..., idx_l-4:idx_h-4, :])
        )
    
    frame_pad = frame_pad_all[..., idx_l:idx_h, :]
    dfdy = (
        4/(5*dy) * (frame_pad[..., :, idx_l+1:idx_h+1] - frame_pad[..., :, idx_l-1:idx_h-1])
        - 1/(5*dy) * (frame_pad[..., :, idx_l+2:idx_h+2] - frame_pad[..., :, idx_l-2:idx_h-2])
        + 4/(105*dy) * (frame_pad[..., :, idx_l+3:idx_h+3] - frame_pad[..., :, idx_l-3:idx_h-3])
        - 1/(280*dy) * (frame_pad[..., :, idx_l+4:idx_h+4] - frame_pad[..., :, idx_l-4:idx_h-4])
        )
    
    return dfdx, dfdy # derivative


# calculate vorticity
@np_torch_trans
def get_vorticity(frame, dx, dy, boundary_mode="periodic", boundary_value=0, 
                  mandatory_use_cuda=False, return_np=False):
    # https://zh.wikipedia.org/wiki/%E6%B6%A1%E9%87%8F
    # ω = ∇ × u 
    """
    input: np.array / torch.Tensor
    output: list of np.array / torch.Tensor
    
    input_shape: 
        frame: batch_size, c=2, h, w
        dx & dy: a scalar
    output_shape: 
        batch_size, c=1, h, w
    
    case:
        f  = torch.randn(10, 2, 64, 64).to("cuda")
        dx = dy = 0.006
        omiga = get_vorticity(f, dx, dy, boundary_mode="periodic", boundary_value=0)
        print(omiga.shape)
        plt.imshow(omiga[0,0,...].detach().cpu().numpy())
        plt.show()
    """
    
    dfdx, dfdy = fd8(frame, dx, dy, boundary_mode=boundary_mode, boundary_value=boundary_value, 
                     return_np=return_np)
    dfxdx, dfydx = dfdx[:, :1, ...], dfdx[:, 1:2, ...]
    dfxdy, dfydy = dfdy[:, :1, ...], dfdy[:, 1:2, ...]
    
    omiga = dfydx - dfxdy # dvdx - dudy
    
    return omiga


###=====================================================================###

def get_energy_spectrum2d_(
        u, v, lx, ly,
        threshold=0.01,
        ) -> np.array:
    # https://www.nature.com/articles/srep35701
    # https://www.astronomy.ohio-state.edu/ryden.1/ast825/ch7.pdf
    # https://zhuanlan.zhihu.com/p/109080129
    # https://zhuanlan.zhihu.com/p/693460826
    # https://xiuqixi.github.io/Turbulence-Spectrum-Calculation/
    """
    计算2D速度场的径向动能谱 E(kappa)。
    
    输入参数：
    u, v - 速度场在 x, y 方向上的分量
    lx, ly - 速度场在 x, y 方向上的尺寸
    
    返回值：
    knyquist - 有效波数截止区间
    knorm - 正则化波数
    kappa_kr - 球形积分后的波数
    tke_kr - 球形积分后的能量
    
    case:
        lx, ly = 1.0, 1.0  # 速度场尺寸
        nx, ny = 128, 128  # 速度场分辨率
        u = np.random.rand(nx, ny)  # 随机生成速度场数据
        v = np.random.rand(nx, ny)
        
        knyquist, knorm, kappa_kr, tke_kr = get_energy_spectrum2d_(u, v, lx, ly)
        
        import matplotlib.pyplot as plt
        plt.plot(kappa_kr, tke_kr)
        plt.xlabel('Wavenumber $\kappa$')
        plt.ylabel('Energy spectrum &E(\kappa)&')
        plt.xscale('log')
        plt.yscale('log')
        plt.show()
    
    case2:
        import _pickle as cPickle
        with open('../TurboGenPY-master-working/vel_2d.pkl', 'rb') as fi: 
            data_uv = cPickle.load(fi) # 两个2D数组 
            u, v = data_uv['u'], data_uv['v']
        with open('../TurboGenPY-master-working/wnn_2d.pkl', 'rb') as fi:
            data_wnn = cPickle.load(fi) # 两个1D数组
            wn, tke_spec = data_wnn['wnn'], data_wnn['wnn_whichspec']
        lx, ly = [9 * 2.0 * np.pi / 100.0] * 2
             
        knyquist, knorm, kappa_kr, tke_kr = get_energy_spectrum2d_(u, v, lx, ly)
        
        import matplotlib.pyplot as plt
        plt.plot(kappa_kr, tke_kr, label='computation')
        plt.plot(wn, tke_spec, label='label')
        plt.axis([8, 10000, 1e-7, 1e-2]) # xlim, ylim
        plt.axvline(x=knyquist, linestyle='--', color='black')
        plt.xlabel('Wavenumber $\kappa$')
        plt.ylabel('Energy spectrum &E(\kappa)&')
        plt.xscale('log')
        plt.yscale('log')
        plt.show()
    """
    if u.shape != v.shape:
        raise ValueError("u and v need to have the same size!")
    
    nx, ny = u.shape
    nt = nx * ny
    
    # step 1 对速度场进行傅里叶变换
    uf = np.fft.fftshift(np.fft.fft2(u)) / nt
    vf = np.fft.fftshift(np.fft.fft2(v)) / nt
    
    # step 1.1 傅里叶变换后对应的波数空间 (wave number, kappa)
    kappa_x = np.arange(-nx//2, nx//2)
    kappa_y = np.arange(-ny//2, ny//2)
    kappa_x, kappa_y = np.meshgrid(kappa_x, kappa_y)
    # kr = np.sqrt(kappa_x**2 + kappa_y**2)
    
    # step 2 速度场能量谱 (kinetic energy spectrum of velocity-spectral)
    tkef = 0.5 * np.real(uf * np.conj(uf) + vf * np.conj(vf))
    
    # step 3 通过在波数空间的球面积分进行归一化 (surface integral at kappa space)
    tke_kr = np.zeros(max(nx, ny))
    for i in range(nx):
        for j in range(ny):
            kr = int(np.round(np.sqrt(kappa_x[i, j]**2 + kappa_y[i, j]**2)))
            tke_kr[kr] += tkef[i, j]
    
    k0x = 2.0 * np.pi / lx
    k0y = 2.0 * np.pi / ly
    knorm = (k0x + k0y) / 2.0
    knyquist = knorm * min(nx, ny) / 2
    kappa_kr = knorm * np.arange(len(tke_kr))
    tke_kr /= knorm
    
    return knyquist, knorm, kappa_kr, tke_kr


def get_energy_spectrum2d(
        u, v, lx, ly,
        smooth=False,
        threshold=0.01,
        ) -> np.array:
    # https://github.com/saadgroup/TurboGenPY
    """
    计算2D速度场的径向动能谱 E(kappa)。
    
    Given a velocity field u, v, w, this function computes the kinetic energy
    spectrum of that velocity field in spectral space. This procedure consists of the
    following steps:
    1. Compute the spectral representation of u, v, and w using a fast Fourier transform.
    This returns uf, vf, and wf (the f stands for Fourier)
    2. Compute the point-wise kinetic energy Ef (kx, ky, kz) = 1/2 * (uf, vf, wf)* conjugate(uf, vf, wf)
    3. For every wave number triplet (kx, ky, kz) we have a corresponding spectral kinetic energy
    Ef(kx, ky, kz). To extract a one dimensional spectrum, E(k), we integrate Ef(kx,ky,kz) over
    the surface of a sphere of radius k = sqrt(kx^2 + ky^2 + kz^2). In other words
    E(k) = sum( E(kx,ky,kz), for all (kx,ky,kz) such that k = sqrt(kx^2 + ky^2 + kz^2) ).

    Parameters:
    -----------
    u: 3D array
      The x-velocity component.
    v: 3D array
      The y-velocity component.
    w: 3D array
      The z-velocity component.
    lx: float
      The domain size in the x-direction.
    ly: float
      The domain size in the y-direction.
    lz: float
      The domain size in the z-direction.
    smooth: boolean
      A boolean to smooth the computed spectrum for nice visualization.
    
    返回值：
    knyquist - 有效波数截止区间
    knorm - 正则化波数
    kappa_kr - 球形积分后的波数
    tke_kr - 球形积分后的能量
    
    
    case:
        lx, ly = 1.0, 1.0  # 速度场尺寸
        nx, ny = 128, 128  # 速度场分辨率
        u = np.random.rand(nx, ny)  # 随机生成速度场数据
        v = np.random.rand(nx, ny)
        
        knyquist, wavenumber, tke_spectrum = get_energy_spectrum2d(u, v, lx, ly)
        
        import matplotlib.pyplot as plt
        plt.plot(wavenumber, tke_spectrum)
        plt.xlabel('Wavenumber $\kappa$')
        plt.ylabel('Energy spectrum &E(\kappa)&')
        plt.xscale('log')
        plt.yscale('log')
        plt.show()
    
    case2:
        import _pickle as cPickle
        with open('../TurboGenPY-master-working/vel_2d.pkl', 'rb') as fi: 
            data_uv = cPickle.load(fi) # 两个2D数组 
            u, v = data_uv['u'], data_uv['v']
        with open('../TurboGenPY-master-working/wnn_2d.pkl', 'rb') as fi:
            data_wnn = cPickle.load(fi) # 两个1D数组
            wn, tke_spec = data_wnn['wnn'], data_wnn['wnn_whichspec']
        lx, ly = [9 * 2.0 * np.pi / 100.0] * 2
             
        knyquist, wavenumber, tke_spectrum = get_energy_spectrum2d(u, v, lx, ly)
        
        import matplotlib.pyplot as plt
        plt.plot(wavenumber, tke_spectrum, label='computation')
        plt.plot(wn, tke_spec, label='label')
        plt.axis([8, 10000, 1e-7, 1e-2]) # xlim, ylim
        plt.axvline(x=knyquist, linestyle='--', color='black')
        plt.xlabel('Wavenumber $\kappa$')
        plt.ylabel('Energy spectrum &E(\kappa)&')
        plt.xscale('log')
        plt.yscale('log')
        plt.legend()
        plt.show()
    """
    
    nx = len(u[:, 0])
    ny = len(v[0, :])

    nt = nx * ny
    n = nx  # int(np.round(np.power(nt,1.0/3.0)))

    uh = np.fft.fftn(u) / nt
    vh = np.fft.fftn(v) / nt

    # tkeh = 0.5 * (uh * np.conj(uh) + vh * np.conj(vh))
    tkeh = 0.5 * np.real(uh * np.conj(uh) + vh * np.conj(vh))

    k0x = 2.0 * np.pi / lx
    k0y = 2.0 * np.pi / ly

    knorm = (k0x + k0y) / 2.0
    # print('knorm = ', knorm)

    kxmax = nx / 2
    kymax = ny / 2

    # dk = (knorm - kmax)/n
    # wn = knorm + 0.5 * dk + arange(0, nmodes) * dk

    wavenumber = knorm * np.arange(0, n) # wave_numbers

    tke_spectrum = np.zeros(len(wavenumber))

    for kx in range(-nx//2, nx//2-1):
        for ky in range(-ny//2, ny//2-1):
        	rk = np.sqrt(kx**2 + ky**2)
        	k = int(np.round(rk))
        	tke_spectrum[k] += tkeh[kx, ky]
    tke_spectrum = tke_spectrum / knorm

    #  tke_spectrum = tke_spectrum[1:]
    #  wavenumber = wavenumber[1:]
    if smooth:
        tkespecsmooth = movingaverage(tke_spectrum, 5)  # smooth the spectrum
        tkespecsmooth[0:4] = tke_spectrum[0:4]  # get the first 4 values from the original data
        tke_spectrum = tkespecsmooth

    knyquist = knorm * min(nx, ny) / 2

    return knyquist, wavenumber, tke_spectrum


def movingaverage(interval, window_size):
    window = np.ones(int(window_size)) / float(window_size)
    return np.convolve(interval, window, 'same')


###=====================================================================###

@np_torch_trans2
def get_rms_velocity(velocity, grid): 
    # https://turbulence.pha.jhu.edu/docs/README-isotropic.pdf
    """ 
    velocity & grid shape: 
        b, c(x,y), h, w (2D) or b, c(x,y,z), h, w, d (3D)
    output:
        E_point:
            b, 1, h, w (2D) or b, 1, h, w, d (3D)
        E_total & rms_velocity:
            b, 1, 1, 1 (2D) or b, 1, 1, 1, 1 (3D)
    
    case:
        E_point, E_total, rms_velocity = get_rms_velocity(torch.Tensor(5,2,64,64), torch.ones(5,2,64,64))
        print(E_point.shape, E_total, rms_velocity.shape)
        E_point, E_total, rms_velocity = get_rms_velocity(torch.Tensor(5,3,64,64,64), torch.ones(5,3,64,64,64))
        print(E_point.shape, E_total, rms_velocity.shape)
    """
    
    E_point = (0.5 * velocity**2).sum(dim=1, keepdim=True) # Energy density on every point
    
    # Total Kinetic Energy, TKE. (mean over filed)
    if len(velocity.shape) == 4: # 2D
        b, c, h, w = velocity.shape
        E_total = (
            (E_point * grid[:,0:1,...] * grid[:,1:2,...]).sum(dim=[-2, -1]) / 
            (grid[:,0:1,...].sum(dim=3, keepdim=True)[:,0:1,0,0]
             * grid[:,1:2,...].sum(dim=2, keepdim=True)[:,0:1,0,0]
             )
        ).reshape(b, 1, 1, 1)
        
        rms_velocity = (2 / 2 * E_total)**0.5 # Rms velocity
    else: # 3D
        b, c, h, w, d = velocity.shape
        E_total = (
            (E_point * grid[:,0:1,...] * grid[:,1:2,...] * grid[:,2:,...]).sum(dim=[-3, -2, -1]) / 
            (grid[:,0:1,...].sum(dim=3, keepdim=True)[:,0:1,0,0,0]
             * grid[:,1:2,...].sum(dim=2, keepdim=True)[:,0:1,0,0,0]
             * grid[:,2:,...].sum(dim=4, keepdim=True)[:,0:1,0,0,0]
             )
        ).reshape(b, 1, 1, 1, 1)
        
        rms_velocity = (2 / 3 * E_total)**0.5 # Rms velocity
        
    # rms_velocity = weight_mean_on_spatial((velocity**2).mean(dim=1, keepdim=True)**0.5) # Rms velocity
    return E_point, E_total, rms_velocity


def get_Taylor_micro_scale(rms_velocity, nu, epsilon): # Taylor Micro. Scale
    # https://turbulence.pha.jhu.edu/docs/README-isotropic.pdf
    Taylor_micro_scale = (15 * nu / epsilon)**0.5 * rms_velocity
    return Taylor_micro_scale


def get_Re(velocity, lenghth, nu): # Taylor-scale Reynolds: Re_lambda = get_Re(rms_velocity, Taylor_micro_scale, nu)
    Re = velocity * lenghth / nu
    return Re

# ----------------------------------------------------

def continuity3D(dudx, dvdy, dwdz): # continuity equation (3D field)
    return dudx + dvdy + dwdz # = 0

def pressure_poisson3D( # pressure poisson equation (3D field)
        d2pdx2, d2pdy2, d2pdz2, 
        dudx, dvdx, dwdx,
        dudy, dvdy, dwdy,
        dudz, dvdz, dwdz,
        rho=1e3, # rho_water: 1e3; rho_air: 1.29
        non_dimensional=True):

    if not non_dimensional:
        # the pressure poisson equation
        return (
            (d2pdx2 + d2pdy2 + d2pdz2)
            + rho 
            * (dudx * dudx + 2 * dudy * dvdx + dvdy * dvdy
                + 2 * dudz * dwdx + 2 * dvdz * dwdy + dwdz * dwdz)
            ) # = 0
    else:
        # the non-dimensional form pressure poisson equation
        return (
            (d2pdx2 + d2pdy2 + d2pdz2)
            + (dudx * dudx + 2 * dudy * dvdx + dvdy * dvdy
               + 2 * dudz * dwdx + 2 * dvdz * dwdy + dwdz * dwdz)
            ) # = 0

# ----------------------------------------------------

def trans3Dto2D_NS(
    force_x, force_y, 
    fz, dfxdz, dfydz, 
    d2fxdz2, d2fydz2, 
    Re_lambda=433, nu=0.000185):
    """ 
    case:
        force_x, force_y = LaPON.trans3Dto2D_NS(
            force_x, force_y, 
            fz, dfxdz, dfydz, 
            d2fxdz2, d2fydz2, 
            Re_lambda=433, nu=0.000185)
    """
    
    # the NS equation
    # force_x = (
    #     + force_x
    #     - fz * dfxdz
    #     + nu # 0.000185 # nu
    #     * d2fxdz2
    # )
    # force_y = (
    #     + force_y
    #     - fz * dfydz
    #     + nu # 0.000185 # nu
    #     * d2fydz2
    # )
    
    # the non-dimensional form NS equation
    force_x = (
        + force_x
        - fz * dfxdz
        + 1 / Re_lambda # 433 # Re_lambda
        * d2fxdz2
    )
    force_y = (
        + force_y
        - fz * dfydz
        + 1 / Re_lambda # 433 # Re_lambda
        * d2fydz2
    )
    
    return force_x, force_y

def trans3Dto2D_poisson(
    force, 
    d2pdz2, 
    dudz, dwdx, dvdz, dwdy, dwdz,
    rho=1e3):
    """
    case:
        force = LaPON.trans3Dto2D_poisson(
            force, 
            fz, dfxdz, dfydz, 
            d2fxdz2, d2fydz2, 
            rho=1e3)
    """
    
    # the pressure poisson equation
    # return (
    #     force
    #     - d2pdz2
    #     - rho 
    #     * (2 * dudz * dwdx + 2 * dvdz * dwdy + dwdz * dwdz)
    #     )
    
    # the non-dimensional form pressure poisson equation
    return (
        force
        - d2pdz2
        - (2 * dudz * dwdx + 2 * dvdz * dwdy + dwdz * dwdz)
        )
    

###=====================================================================###

### PDE (NSE)

# the NS equation (2D)
def nse2d(
        fx, fy,
        dfxdx, dfxdy,
        dfydx, dfydy,
        d2fxdx2, d2fxdy2,
        d2fydx2, d2fydy2,
        dpdx, dpdy,
        force_x, force_y,
        rho, nu
        ):
    dfxdt = (
        - (fx * dfxdx + fy * dfxdy)
        - 1 / rho
        * dpdx
        + nu
        * (d2fxdx2 + d2fxdy2)
        + force_x
    )
    dfydt = (
        - (fx * dfydx + fy * dfydy)
        - 1 / rho
        * dpdy
        + nu
        * (d2fydx2 + d2fydy2)
        + force_y
    )
    dfdt = torch.cat([dfxdt, dfydt], dim=1)
    return dfdt

# the non-dimensional form NS equation (2D)
def nse2d_nd(
        fx, fy,
        dfxdx, dfxdy,
        dfydx, dfydy,
        d2fxdx2, d2fxdy2,
        d2fydx2, d2fydy2,
        dpdx, dpdy,
        force_x, force_y,
        Re # i.e. Re_lambda
        ):
    dfxdt = (
        - (fx * dfxdx + fy * dfxdy)
        - dpdx
        + 1 / Re
        * (d2fxdx2 + d2fxdy2)
        + force_x
    )
    dfydt = (
        - (fx * dfydx + fy * dfydy)
        - dpdy
        + 1 / Re
        * (d2fydx2 + d2fydy2)
        + force_y
    )
    dfdt = torch.cat([dfxdt, dfydt], dim=1)
    return dfdt

# the velocity-pressure(VP) form non-dimensional NS equation (2D)
# TODO ...

# the vorticity-velocity(VV) form non-dimensional NS equation (2D)
# TODO ...


###=====================================================================###

