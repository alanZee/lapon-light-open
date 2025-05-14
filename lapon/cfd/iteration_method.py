# LaPON, GPL-3.0 license
# torch(GPU)-based Iteration method solver of Poisson equation 

'''
该求解器支持:
    self.mode_ls = ["jacobi", "gauss_seidel", "SOR", "exact"]
    
    self.boundary_mode_ls = [
        "dirichlet", # y=C
        "newman",    # for y'=0
        "periodic",  # y(0,t)=y(L,t)
        "symmetry"]  # y(-x,t)=y(x,t)
'''


import sys, os, glob, datetime, random, math, re
import _pickle as cPickle
from copy import deepcopy
from pathlib import Path
from tqdm import tqdm 

import numpy as np, pandas as pd, matplotlib.pyplot as plt

import torch
import torch.nn as nn

if __name__ == "__main__":
    sys.path.append(os.path.abspath("../"))
# else:
#     from pathlib import Path
#     FILE = Path(__file__).resolve()
#     ROOT = FILE.parents[0]  # LaPON root directory
#     if str(ROOT) not in sys.path:
#         sys.path.append(str(ROOT))  # add ROOT to PATH
#     ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative
try:
    from utils.augmentations import torch_resize, letterbox, cutout, spatial_derivative
except:
    print("WARNING ⚠️ def get_velocity_grad() : There are no dependent packages.")
    


#%%

class PoissonEqSolver(nn.Module): 
    """
    input shape:
        force: b, c(=1), h, w
    output shape:
        f: b, c(=1), h, w
    
    case0(plot all):
        solver = PoissonEqSolver(3, 4, 1, 1, mode="SOR", boundary_mode="dirichlet")
        for m in solver.mode_ls:
            for b in solver.boundary_mode_ls:
                solver = PoissonEqSolver(3, 4, 1, 2, mode=m, boundary_mode=b).to("cuda")
                b      = torch.zeros(1, 1, 4, 3).to("cuda")
                solver.plot_A()
                solver(b).shape
            
    case1:
        solver = PoissonEqSolver(32, 32, 1, 1, mode="SOR", boundary_mode="dirichlet").to("cuda")
        b      = torch.zeros(1, 1, 32, 32).to("cuda")
        solver.plot_A()
        solver(b).shape
    
    case2:
        solver = PoissonEqSolver(8, 8, 1, 2, mode="SOR", boundary_mode="dirichlet").to("cuda")
        b      = torch.zeros(1, 1, 8, 8).to("cuda")
        solver.plot_A()
        solver(b).shape
    
    case3:
        solver = PoissonEqSolver(3, 9, 1, 2, mode="SOR", boundary_mode="dirichlet").to("cuda")
        b      = torch.zeros(1, 1, 9, 3).to("cuda")
        solver.plot_A()
        solver(b).shape
        
    case4:
        solver = PoissonEqSolver(3, 4, 1, 2, mode="SOR", boundary_mode="dirichlet").to("cuda")
        b      = torch.zeros(1, 1, 4, 3).to("cuda")
        solver.plot_A()
        solver(b).shape
        
    case5:
        solver = PoissonEqSolver(4, 2, 1, 2, mode="SOR", boundary_mode="dirichlet").to("cuda")
        b      = torch.zeros(1, 1, 2, 4).to("cuda")
        solver.plot_A()
        solver(b).shape
        
    case6: (periodic)
        solver = PoissonEqSolver(4, 2, 1, 2, mode="SOR", boundary_mode="periodic").to("cuda")
        b      = torch.zeros(1, 1, 2, 4).to("cuda")
        solver.plot_A()
        solver(b).shape
    
    case7: (symmetry)
        solver = PoissonEqSolver(8, 4, 1, 2, mode="SOR", boundary_mode="symmetry").to("cuda")
        b      = torch.zeros(1, 1, 4, 8).to("cuda")
        solver.plot_A()
        solver(b).shape
        
    case8: (periodic)
        solver = PoissonEqSolver(18, 8, 1, 2, mode="SOR", boundary_mode="periodic").to("cuda")
        b      = torch.zeros(1, 1, 8, 18).to("cuda")
        solver.plot_A()
        solver(b).shape
        
    case9: (symmetry)
        solver = PoissonEqSolver(18, 8, 1, 2, mode="SOR", boundary_mode="symmetry").to("cuda")
        b      = torch.zeros(1, 1, 8, 18).to("cuda")
        solver.plot_A()
        solver(b, use_pbar=True).shape
        
    case10:
        solver = PoissonEqSolver(16, 32, 1, 2, mode="SOR", boundary_mode="symmetry").to("cuda")
        b      = torch.zeros(1, 1, 32, 16).to("cuda")
        b[:, :, :,0]=100
        y = solver(b, use_pbar=True)[0,0,...].detach().cpu()
        plt.imshow(y)
        
        solver = PoissonEqSolver(16, 32, 1, 2, mode="SOR", boundary_mode="periodic").to("cuda")
        b      = torch.zeros(1, 1, 32, 16).to("cuda")
        b[:, :, :,0]=100
        y = solver(b, use_pbar=True)[0,0,...].detach().cpu()
        plt.imshow(y)
        
        solver = PoissonEqSolver(16, 32, 1, 2, mode="SOR", boundary_mode="dirichlet").to("cuda")
        b      = torch.zeros(1, 1, 32, 16).to("cuda")
        b[:, :, :,0]=100
        y = solver(b, use_pbar=True)[0,0,...].detach().cpu()
        plt.imshow(y)
        
        solver = PoissonEqSolver(16, 32, 1, 2, mode="SOR", boundary_mode="newman").to("cuda")
        b      = torch.zeros(1, 1, 32, 16).to("cuda")
        b[:, :, :,0]=100
        y = solver(b, use_pbar=True)[0,0,...].detach().cpu()
        plt.imshow(y)
    
    case11:
        solver = PoissonEqSolver(100, 100, 0.02, 0.02, mode="SOR", boundary_mode="dirichlet").to("cuda")
        b      = torch.zeros(1, 1, 100, 100).to("cuda")
        b[:, :, 50, 23:27]=30
        b[:, :, 50, 75]=-120
        y = solver(b, use_pbar=True)[0,0,...].detach().cpu()
        plt.imshow(y)
        # 添加等高线
        contours = plt.contour(y, colors='black')
        # 添加等高线数字标签
        plt.clabel(contours, inline=True, fontsize=8, fmt='%1.3f')
    """
    def __init__(
            self, 
            nx, # num 
            ny,
            dx, 
            dy, 
            mode="SOR", 
            boundary_mode="dirichlet",
            **kwargs):
        super().__init__(**kwargs)
        self.nx   = nx
        self.ny   = ny
        self.dxr  = dxr = 1 / dx # reciprocal
        self.dyr  = dyr = 1 / dy
        self.mode = mode
        self.boundary_mode = boundary_mode = boundary_mode.lower()
        
        self.get_c() # for self.decouple()
        
        self.mode_ls = ["jacobi", "gauss_seidel", "SOR", "exact"]
        assert mode in self.mode_ls, f"mode={mode} is not supported, must be in {self.mode_ls}"
        
        self.boundary_mode_ls = [
            "dirichlet", # y=C
            "newman",    # for y'=0
            "periodic",  # y(0,t)=y(L,t)
            "symmetry"]  # y(-x,t)=y(x,t)
        assert boundary_mode in self.boundary_mode_ls, f'Received boundary_mode is {boundary_mode}, but it must be in {self.boundary_mode_ls}'
        
        # get laplace matrix
        self.A = self.get_laplace_matrix(nx, ny, dxr, dyr)
        
        # decomposition
        self.D, self.L, self.U = self.get_dlu(self.A)
        
        # np array -> torch Parameter
        self.A = nn.Parameter(
            torch.from_numpy(self.A),
            requires_grad=False)
        self.D = nn.Parameter(
            torch.from_numpy(self.D),
            requires_grad=False)
        self.L = nn.Parameter(
            torch.from_numpy(self.L),
            requires_grad=False)
        self.U = nn.Parameter(
            torch.from_numpy(self.U),
            requires_grad=False)
        
    def forward(self, force, init_solution=None,
                max_iter = 1000, tol = 1e-6,
                omiga = 1.8,
                boundary_value=0,
                use_pbar=False): 
        
        # 展开force得到方程 Poisson equation A @ y_flat = force_flat 的右端列向量
        #force_flat = force.flatten(order='F').reshape([-1,1]) # NOTE C: C风格(按行展开); F: Fortran风格(按列展开)
        force_flat = force.transpose(2, 3).reshape([force.shape[0], 1, -1, 1])
        
        if self.mode != "exact":
            if init_solution is None:
                init_solution_flat = torch.zeros(force.shape[0], 1, self.nx * self.ny, 1).to(self.A.device)
                # init_solution_flat = - force_flat
            else:
                init_solution_flat = init_solution.transpose(2, 3).reshape([force.shape[0], 1, -1, 1])
                
            B, F = self.getBF(force_flat, mode=self.mode, omiga=omiga)
            
            if use_pbar:
                pbar = tqdm(range(max_iter), desc='PE iter')
            else:
                pbar = range(max_iter)
                
            ### iteration
            y_flat_last = init_solution_flat + 0 # 类似于 deepcopy(init_solution_flat)
            y_flat = init_solution_flat + 0 
            temp = B + 0
            _ = torch.zeros([y_flat.shape[0], y_flat.shape[1], 
                             y_flat.shape[3], y_flat.shape[2]]).to(self.A.device)
            res = y_flat_last + 0
            for i in pbar:
                ### iter once
                # y_flat = B @ y_flat + F 
                temp @= y_flat # NOTE in-place 
                temp += F 
                y_flat[...] = temp # NOTE in-place 
                
                ### get and check res 
                res[...] = y_flat_last
                res -= y_flat
                if self.norm2(res) <= tol: 
                    break
                
                ### prepare for next iter
                y_flat_last[...] = y_flat 
                temp @= _ # revert temp shape
                temp[...] = B 
        else:
            # Exact solution 精确求解(maybe)
            try:
                y_flat = torch.linalg.solve(self.A, force_flat[:, 0, ...])[:, None, ...]
            except:
                y_flat = np.linalg.solve(self.A.cpu(), force_flat[:, 0, ...].cpu())[:, np.newaxis, ...]
                y_flat = torch.from_numpy(y_flat).to(self.A.device)

        y = y_flat.reshape([force.shape[0], force.shape[1], force.shape[3], force.shape[2]]).transpose(2, 3)
        
        y = self.decouple(y, boundary_value=boundary_value)
        
        return y
    
    @staticmethod
    def norm2(x): 
        ls = []
        for i in range(x.shape[0]):
            # try:
            #     ls.append(
            #         torch.linalg.norm(x[i,0,...].to("cuda"), ord=2) 
            #         )
            # except:
            #     ls.append(
            #         np.linalg.norm(x[i,0,...].cpu(), ord=2) 
            #         )
            ls.append(
                np.linalg.norm(x[i,0,...].cpu(), ord=2) 
                )
        return float(max(ls))
    
    def get_laplace_matrix(self, nx, ny, dxr, dyr): 
        N = nx * ny
        A = np.zeros([N, N], dtype="float32")
        
        # NOTE 2D filed; 1 padding; 5点星展开成矩阵(五点二阶精度二阶差分)
        
        if self.boundary_mode in [
            "dirichlet", # y=C
            "newman",    # for y'=0
            ]:
            # Dirichlet / Newman boundry condition; 每一行有 5(中间点) / 4(边界点) / 3(拐角点) 个 非零值
            # 展开成5对角矩阵
            for j in range(nx): 
                for i in range(ny):
                    baseline = j*ny + i
                    
                    A[baseline, baseline] = -2 * (dxr**2 + dyr**2) 
                    # np.fill_diagonal(A, 1)
                    
                    if j != nx-1:
                        A[baseline, baseline + ny] = dxr**2 
                        A[baseline + ny, baseline] = dxr**2 
                        
                    if i != ny-1: # 全体j都有
                        A[baseline, baseline + 1] = dyr**2 
                        A[baseline + 1, baseline] = dyr**2 
                        
        elif self.boundary_mode == "periodic": # y(0,t)=y(L,t)
            # Periodic boundry condition; 每一行有 5 个 非零值
            # 非5对角矩阵
            for j in range(nx): 
                for i in range(ny):
                    baseline = j*ny + i
                    
                    A[baseline, baseline] = -2 * (dxr**2 + dyr**2) 
                    
                    if j != nx-1:
                        A[baseline, baseline + ny] = dxr**2 
                        A[baseline + ny, baseline] = dxr**2 
                        
                    if i != ny-1: # 全体j都有
                        A[baseline, baseline + 1] = dyr**2 
                        A[baseline + 1, baseline] = dyr**2 
                        
                    # 边界点的处理(含拐角点)
                    
                    if j == 0:
                        A[baseline, (nx-1)*ny + i] = dxr**2 
                        A[(nx-1)*ny + i, baseline] = dxr**2 
                    
                    if i == 0: # 全体j都有
                        A[baseline, baseline + (ny-1)] = dyr**2 
                        A[baseline + (ny-1), baseline] = dyr**2 
                        
        elif self.boundary_mode == "symmetry": # y(-x,t)=y(x,t)
            # Symmetry boundry condition; 每一行有 5 个 非零值
            # 展开成5对角矩阵
            for j in range(nx): 
                for i in range(ny):
                    baseline = j*ny + i
                    
                    A[baseline, baseline] = -2 * (dxr**2 + dyr**2) 
                    
                    if j != nx-1:
                        A[baseline, baseline + ny] = dxr**2 
                        A[baseline + ny, baseline] = dxr**2 
                        
                    if i != ny-1: # 全体j都有
                        A[baseline, baseline + 1] = dyr**2 
                        A[baseline + 1, baseline] = dyr**2 
                    
                    # 边界点的处理(含拐角点)
                    
                    if j == 0:
                        A[baseline, baseline + ny] *= 2
                    if j == nx-1:
                        A[baseline, baseline - ny] *= 2
                    
                    if i == 0: # 全体j都有
                        A[baseline, baseline + 1] *= 2 
                    if i == ny-1: # 全体j都有
                        A[baseline, baseline - 1] *= 2 
                        
        return A
    
    @staticmethod
    def get_dlu(A): # decomposition: A = D + L + U
    
        # 将矩阵 A，可以将其分解为三个矩阵 A = D + L + U。其中 D 为对角矩阵，L 为下三角矩阵，U 为上三角矩阵。
        D = np.diag(np.diag(A, 0), 0) # 取对角矩阵 
        L = np.tril(A, -1) # 获取下三角矩阵 
        U = np.triu(A, 1) # 获取下三角矩阵
        
        return D, L, U
    
    def getBF(self, force_flat, mode="SOR", omiga=1.8):
        device = force_flat.device
        
        if mode == "jacobi":
            try:
                B = torch.linalg.solve(- self.D, self.L + self.U)
                F = torch.linalg.solve(self.D, force_flat)
                # 如果你的系数矩阵是三角矩阵（只有下三角或上三角有非零元素），你可以使用torch.triangular_solve来求解。这个函数比torch.linalg.solve更高效，因为它利用了矩阵的三角性质。
                # x, _ = torch.triangular_solve(b, A, upper=False)
            except:
                B = nn.Parameter(torch.from_numpy(
                        np.linalg.solve(- self.D.detach().cpu(), (self.L + self.U).detach().cpu())
                        ), requires_grad=False).to(device)
                F = nn.Parameter(torch.from_numpy(
                        np.linalg.solve(self.D.detach().cpu(), force_flat.detach().cpu())
                        ), requires_grad=False).to(device)
        elif mode == "gauss_seidel":
            try:
                B = torch.linalg.solve(- (self.D + self.L), self.U)
                F = torch.linalg.solve((self.D + self.L), force_flat)
            except:
                B = nn.Parameter(torch.from_numpy(
                        np.linalg.solve(- (self.D + self.L).detach().cpu(), self.U.detach().cpu())
                        ), requires_grad=False).to(device)
                F = nn.Parameter(torch.from_numpy(
                        np.linalg.solve((self.D + self.L).detach().cpu(), force_flat.detach().cpu())
                        ), requires_grad=False).to(device)
        elif mode == "SOR":
            try:
                B = torch.linalg.solve(self.D + omiga * self.L, (1-omiga) * self.D - omiga * self.U)
                F = torch.linalg.solve(self.D + omiga * self.L, omiga * force_flat)
            except:
                B = nn.Parameter(torch.from_numpy(
                        np.linalg.solve((self.D + omiga * self.L).detach().cpu(), ((1-omiga) * self.D - omiga * self.U).detach().cpu())
                        ), requires_grad=False).to(device)
                F = nn.Parameter(torch.from_numpy(
                        np.linalg.solve((self.D + omiga * self.L).detach().cpu(), (omiga * force_flat).detach().cpu())
                        ), requires_grad=False).to(device)
                # print(F)
        
        return B, F
    
    def plot_A(self, cmap=None):
        fig = plt.figure(figsize = (6, 6)) # ich:16, 9; resolution:2560, 1440; dpi:160, 160
        a = fig.add_subplot(111)
        a.imshow(self.A.detach().cpu(), cmap=cmap)
        a.set_axis_off()
        plt.savefig(f"laplace_matrix-{self.nx}x{self.ny}-{self.mode}-{self.boundary_mode}.png", dpi=200)
        plt.show()
        plt.close(fig)
        
    def decouple(self, y, boundary_value=0): 
        if self.boundary_mode == self.boundary_mode_ls[0]:
            y[0,:]  = y[0,:]  + self.cb * boundary_value 
            y[-1,:] = y[-1,:] + self.cb * boundary_value 
            y[:,0]  = y[:,0]  + self.ca * boundary_value 
            y[:,-1] = y[:,-1] + self.ca * boundary_value 
        elif self.boundary_mode == self.boundary_mode_ls[1]:
            temp = 2 * y[0,0], 2 * y[0,-1], 2 * y[-1,0], 2 * y[-1,-1]
            
            y[0,:]  = y[0,:]  / (1 + self.cb)
            y[-1,:] = y[-1,:] / (1 + self.ca)
            y[:,0]  = y[:,0]  / (1 + self.cb)
            y[:,-1] = y[:,-1] / (1 + self.ca) 
            
            y[0,0], y[0,-1], y[-1,0], y[-1,-1] = temp
        elif self.boundary_mode in self.boundary_mode_ls[2:]:
            pass 
        
        return y
    
    def get_c(self): # for self.decouple()
        self.ca = 0.5 * self.dxr**2 / (self.dxr**2 + self.dyr**2)
        self.cb = 0.5 * self.dyr**2 / (self.dxr**2 + self.dyr**2)
    
    @staticmethod
    def get_velocity_grad(frames, grid, boundary_mode="newman", boundary_value=0):
        # frames shape: b, (t,) c, h, w; grid shape: a scalar or a tensor shape as frames
        assert len(frames.shape) in [4, 5], f"len(frames.shape)={len(frames.shape)} must be in [4, 5]"
        
        try:
            dfdx, dfdy = spatial_derivative(frames, grid, boundary_mode, boundary_value)
            if len(frames.shape) == 5:
                dfxdx, dfydx = dfdx[:, :, :1, ...], dfdx[:, :, 1:, ...]
                dfxdy, dfydy = dfdy[:, :, :1, ...], dfdy[:, :, 1:, ...]
            else:
                dfxdx, dfydx = dfdx[:, :1, ...], dfdx[:, 1:, ...]
                dfxdy, dfydy = dfdy[:, :1, ...], dfdy[:, 1:, ...]
            return dfxdx, dfydx, dfxdy, dfydy 
        except:
            raise Exception("def get_velocity_grad() : There are no dependent packages.")
    
    @staticmethod
    def get_PPE_right_hand_side_item(dfxdx, dfydx, dfxdy, dfydy, rho=1.29, non_dimensional=True):
        # rho_water: 1e3; rho_air: 1.29
        """
        usage: 
            PoissonEqSolver.get_PPE_right_hand_side_item(
                *PoissonEqSolver.get_velocity_grad(frames, grid, 
                       boundary_mode="newman", boundary_value=0)
                , rho=1.29, non_dimensional=False)
        """
        if non_dimensional:
            return - (dfxdx**2 + 2 * dfxdy * dfydx + dfydy**2)
        else:
            return - rho * (dfxdx**2 + 2 * dfxdy * dfydx + dfydy**2)
    
