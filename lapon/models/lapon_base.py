# LaPON, GPL-3.0 license
# LaPON base ANN modules (for PDE)

"""
AI + DNS for large discrete scale: DNS 嵌入 AI 中，并基于离散精度施加硬约束
LaPON目前专注于含时偏微分方程初边值问题的智能求解（流场非稳态解）, 特点是可以在任意物理场上, 以较高精度在大时间步长上做时间演进
# 在有限差分的语境下，离散误差指的是Taylor公式相关的截断误差
"""

"""
Architecture:
    Backbone
        HistoricalDynamicExtractor; （ConvLSTM <- f，dx网格，dt；历史帧，历史帧的dt和dx网格；物性ρ和ν，工况L和v，系统参数Re）
        PDEModule[  - Rate of change  
            SpatialModule[  - Rate of change (Gradient)
                SpatialDerivative[ （数值微分；冻结参数 <- f，dx网格）（计算当前空间点的数值导数）（利用卷积代替空间差分，使用GPU实现并行计算）
                    SpatialDifferential], 
                SpatialOperator], （CrossAttention <- f ，dx网格；历史动态；物性ρ和ν，工况L和v，系统参数Re）（基于当前空间点的数值导数，计算该导数的精确值）
            PDE_part] （无量纲形式PDE）（离散的PDE；冻结参数 <- f，dt）（计算当前时刻的数值导数）
        TemporalOperator  - Variation 变化量 ÷ dt （CrossAttention <- f，dt；历史动态；物性ρ和ν，工况L和v，系统参数Re）（基于当前时刻的数值导数，计算中值的导数）
    Head （欧拉法；冻结参数 <- f；Variation ÷ dt）
        
需要自定义的部分：
    PDEModule的计算图
"""


import os, sys
from copy import deepcopy

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
from models.common import *


#%%

# 算子模块
class YoloBasedUnetBranchOperatorDemo(nn.Module):
    stride = 32
    """
    The maximum reduction factor of the CNN module to the input frame size.
    """
    
    """
    input_shape: (batch_size,) c, h, w
    output_shape: (batch_size,) c_out, h, w
    
    case:
        n = YoloBasedUnetBranchOperatorDemo(3).to("cuda")
        x = torch.Tensor(10,3,32,32).to("cuda")
        y = n(x)
        print(y.shape)
        plt.imshow(x[0,0,...].detach().cpu()), plt.show()
        plt.imshow(y[0,0,...].detach().cpu()), plt.show()
        plt.imshow(y.permute(0,1,3,2)[0,0,...].detach().cpu()), plt.show()
        plt.imshow(n(x.permute(0,1,3,2))[0,0,...].detach().cpu()), plt.show()
    """
    
    def __init__(
        self,
        c_in,
        c_out=1,
        use_isotropic_conv=True,
        depth_multiple=1.00, # 0.33, 1.33
        width_multiple=1.00, # 0.50, 1.25
        **kwargs
    ):
        super().__init__(**kwargs)
        self.use_isotropic_conv = use_isotropic_conv
        self.depth_multiple = depth_multiple
        self.width_multiple = width_multiple
        
        n_layer_base = 2 # module depth
        n_layer = max(round(n_layer_base * self.depth_multiple), 1) 
        
        c_latent_base = 1024 # num of latent layer channel 
        c_latent = int(c_latent_base * self.width_multiple)
        c_latent2 = int(c_latent // 2)
        c_latent3 = int(c_latent2 // 2)
        
        self.c_latent = c_latent
        
        self.conv1 = CBS(c_in, c_latent2, 6, 2, 2, 
                        use_isotropic_conv=self.use_isotropic_conv)
        self.c31 = nn.Sequential(*( # module
            C3(c_latent2, c_latent2,
               use_isotropic_conv=self.use_isotropic_conv)
            for _ in range(n_layer)))
        self.conv2 = CBS(c_latent2, c_latent, 3, 2,
                        use_isotropic_conv=self.use_isotropic_conv)
        self.c32 = nn.Sequential(*( # module
            C3(c_latent, c_latent, 
               use_isotropic_conv=self.use_isotropic_conv)
            for _ in range(n_layer)))
        self.c33 = nn.Sequential(*( # module
            C3(c_latent, c_latent, 
               use_isotropic_conv=self.use_isotropic_conv)
            for _ in range(n_layer)))
        self.sppf = SPPF(c_latent, c_latent, 
                         use_isotropic_conv=self.use_isotropic_conv)
        
        self.conv3 = CBS(c_latent, c_latent2, 1, 1, 
                        use_isotropic_conv=self.use_isotropic_conv)
        self.upsample1 = nn.Upsample(None, 2, 'nearest')
        self.cat1 = Concat(1)
        self.c34 = C3(c_latent2 + c_latent2, c_latent2, shortcut=False, 
                      use_isotropic_conv=self.use_isotropic_conv)
        self.c34_2 = nn.Sequential(*( # module
            C3(c_latent2, c_latent2, shortcut=False, 
               use_isotropic_conv=self.use_isotropic_conv)
            for _ in range(n_layer)))
        
        self.conv4 = CBS(c_latent2, c_latent3, 1, 1, 
                        use_isotropic_conv=self.use_isotropic_conv)
        self.upsample2 = nn.Upsample(None, 2, 'nearest')
        self.cat2 = Concat(1)
        self.c35 = C3(c_in + c_latent3, c_latent3, shortcut=False, 
                      use_isotropic_conv=self.use_isotropic_conv)
        self.c35_2 = nn.Sequential(*( # module
            C3(c_latent3, c_latent3, shortcut=False, 
               use_isotropic_conv=self.use_isotropic_conv)
            for _ in range(n_layer)))
        
        self.conv5 = CBS(c_latent3, c_out, 1, 1, 
                        use_isotropic_conv=self.use_isotropic_conv)

    def forward(self, x, training=None):
        x1 = self.c31(self.conv1(x))
        x2 = self.c33(self.c32(self.conv2(x1)))
        x3 = self.sppf(x2)
        x4 = self.c34_2(self.c34(self.cat1([x1, self.upsample1(self.conv3(x3))])))
        x5 = self.c35_2(self.c35(self.cat2([x, self.upsample2(self.conv4(x4))])))
        return self.conv5(x5)

    
#%%

# YoloBased Encoder
class YoloBasedEncoder(nn.Module):
    stride = 32
    """
    The maximum reduction factor of the CNN module to the input frame size.
    """
    
    """
    input_shape: (batch_size,) c, h, w
    output_shape: (batch_size,) c_latent, h_map, w_map
    
    case:
        n = YoloBasedEncoder(3).to("cuda")
        x = torch.Tensor(10,3,32,32).to("cuda")
        y = n(x)
        print(y.shape)
        plt.imshow(x[0,0,...].detach().cpu()), plt.show()
        plt.imshow(y[0,0,...].detach().cpu()), plt.show()
        plt.imshow(y.permute(0,1,3,2)[0,0,...].detach().cpu()), plt.show()
        plt.imshow(n(x.permute(0,1,3,2))[0,0,...].detach().cpu()), plt.show()
    """
    
    def __init__(
        self,
        c_in,
        use_isotropic_conv=True,
        depth_multiple=1.00, # 0.33, 1.33
        width_multiple=1.00, # 0.50, 1.25
        **kwargs
    ):
        super().__init__(**kwargs)
        self.use_isotropic_conv = use_isotropic_conv
        self.depth_multiple = depth_multiple
        self.width_multiple = width_multiple
        
        n_layer_base = 2 # module depth
        n_layer = max(round(n_layer_base * self.depth_multiple), 1) 
        
        c_latent_base = 1024 # num of latent layer channel 
        c_latent = int(c_latent_base * self.width_multiple)
        c_latent2 = int(c_latent // 2)
        c_latent3 = int(c_latent2 // 2)
        
        self.c_latent = c_latent
        
        self.conv1 = CBS(c_in, c_latent2, 6, 2, 2, 
                        use_isotropic_conv=self.use_isotropic_conv)
        self.c31 = nn.Sequential(*( # module
            C3(c_latent2, c_latent2,
               use_isotropic_conv=self.use_isotropic_conv)
            for _ in range(n_layer)))
        self.conv2 = CBS(c_latent2, c_latent, 3, 2,
                        use_isotropic_conv=self.use_isotropic_conv)
        self.c32 = nn.Sequential(*( # module
            C3(c_latent, c_latent, 
               use_isotropic_conv=self.use_isotropic_conv)
            for _ in range(n_layer)))
        self.c33 = nn.Sequential(*( # module
            C3(c_latent, c_latent, 
               use_isotropic_conv=self.use_isotropic_conv)
            for _ in range(n_layer)))
        self.sppf = SPPF(c_latent, c_latent, 
                         use_isotropic_conv=self.use_isotropic_conv)
        
    def forward(self, x, training=None):
        x1 = self.c31(self.conv1(x))
        x2 = self.c33(self.c32(self.conv2(x1)))
        x3 = self.sppf(x2)
        return x3
   

#%%

# 历史动态的提取
class HistoricalDynamicExtractor(nn.Module):
    """
    input_shape: 
        x:     (batch_size,) series_len, c, h, w
        grid:  (batch_size,) series_len, c, h, w
        dt:    (batch_size,) 1(chanel), series_len
        param: (batch_size,) 1(chanel), n_param
    output_shape: 
        (batch_size,) 1, self.c_out
        
    case1:
        hde = HistoricalDynamicExtractor((None, 8, 2, 64, 64), (None, 8, 2, 64, 64), (None, 1, 8), (None, 1, 5)).to("cuda")
        x     = torch.Tensor(10, 8, 2, 64, 64).to("cuda")
        grid  = torch.Tensor(10, 8, 2, 64, 64).to("cuda")
        dt    = torch.Tensor(10, 1, 8).to("cuda")
        param = torch.Tensor(10, 1, 5).to("cuda")
        y = hde(x, grid, dt, param)
        print(y.shape)
    
    case2:
        hde = HistoricalDynamicExtractor((None, 1, 2, 64, 64), (None, 1, 2, 64, 64), (None, 1, 1), (None, 1, 5)).to("cuda")
        x     = torch.Tensor(10, 1, 2, 64, 64).to("cuda")
        grid  = torch.Tensor(10, 1, 2, 64, 64).to("cuda")
        dt    = torch.Tensor(10, 1, 1).to("cuda")
        param = torch.Tensor(10, 1, 5).to("cuda")
        y = hde(x, grid, dt, param)
        print(y.shape)
    
    case3(single sample):
        # ValueError: Expected more than 1 value per channel when training, got input size torch.Size([1, 1024, 1, 1])
        
        hde = HistoricalDynamicExtractor((None, 1, 2, 64, 64), (None, 1, 2, 64, 64), (None, 1, 1), (None, 1, 3)).to("cuda")
        x     = torch.Tensor(1, 1, 2, 64, 64).to("cuda")
        grid  = torch.Tensor(1, 1, 2, 64, 64).to("cuda")
        dt    = torch.Tensor(1, 1, 1).to("cuda")
        param = torch.Tensor(1, 1, 3).to("cuda")
        y = hde(x, grid, dt, param)
        print(y.shape)
    """
    def __init__(
        self,
        shape_in_f,
        shape_in_grid,
        shape_in_dt,
        shape_in_param,
        c_out=1024,
        use_isotropic_conv=True,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.c_out = c_out
        self.use_isotropic_conv = use_isotropic_conv
        
        self.boundary = BoundaryPadding()
        self.boundary_padding = (3,3,3,3) # 各边需一致 (与第一层Cell的kernel_size匹配，实现同维卷积)
        
        C_LATENT = 256
        
        # B, T, C, H, W -> B, C_out, H_out, W_out
        self.convlstms = CConvLSTM(
            input_dim=shape_in_f[-3],
            hidden_dim=[64, 128, C_LATENT],
            kernel_size=[(7, 7), (5, 5), (5, 5)],
            num_layers=3, 
            stride=2,
            use_isotropic_conv=self.use_isotropic_conv)
        convlstms_out_size = self.convlstms.get_convlstms_out_size(shape_in_f[-1] + self.boundary_padding[0] * 2)
        
        v_shape = (shape_in_f[0], C_LATENT, convlstms_out_size, convlstms_out_size)
        
        self.spatial_attention = AdaptiveMultiHeadCrossAttention(
            shape_in_grid, 
            v_shape, v_shape,
            use_isotropic_conv=self.use_isotropic_conv)
        self.channel_attention = AdaptiveMultiHeadCrossAttention(
            (None, 1, shape_in_dt[-1] + shape_in_param[-1]), 
            v_shape, v_shape,
            use_isotropic_conv=self.use_isotropic_conv)
        
        self.conv = CBS(C_LATENT, c_out, 1, 2, 0,
                        use_isotropic_conv=self.use_isotropic_conv)
        self.c3 = C3(c_out, c_out,
                     use_isotropic_conv=self.use_isotropic_conv)
        
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
    def forward(self, x, grid, dt, param, 
                boundary_mode="periodic", boundary_value=0, training=None):
        params = torch.cat([dt, param], -1)
        
        # boundary condition
        x_ = torch.zeros(x.shape[0], x.shape[1], x.shape[2], 
                         x.shape[3] + self.boundary_padding[0] * 2, 
                         x.shape[4] + self.boundary_padding[0] * 2,
                         ).to(self.conv.conv.weight.device)
        grid_ = deepcopy(x_)
        for i in range(x.shape[1]):
            x_[:,i,...] = self.boundary(x[:,i,...], padding=self.boundary_padding, mode=boundary_mode, value=boundary_value) 
            grid_[:,i,...] = self.boundary(grid[:,i,...], padding=self.boundary_padding, mode=boundary_mode, value=boundary_value) 
        x = x_
        grid = grid_
        
        _, h_c_state_list = self.convlstms(x) 
        x = h_c_state_list[-1][0] # e.g torch.Size([3, 2, 2, 96, 96]) -> torch.Size([3, 256, 9, 9])
        
        x, _attention_weights1 = self.spatial_attention(grid, x, x)
        x, _attention_weights2 = self.channel_attention(params, x, x)
        
        x = self.c3(self.conv(x))
        
        y = self.max_pool(x)
        
        return y.view(-1, 1, self.c_out)


#%%

# 空间上的 数值微分 
class SpatialDifferential(nn.Module): 
    """
    input_shape: 
        x: (batch_size,) c, h, w
    output_shape(each): 
        (batch_size,) c, h, w
        
    case:
        sdi = SpatialDifferential().to("cuda")
        f = torch.arange(1, 10*2*64*64+1, dtype=torch.float32).reshape(10, 2, 64, 64).to("cuda")
        sdi(f, boundary_mode="periodic", boundary_value=0)[0].shape
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        self.boundary = BoundaryPadding()
        
    def forward(self, inputs, 
                boundary_mode="periodic", boundary_value=0):
        # 通过灵活改变shape的形式，实现conv在原通道维度上的广播运算
        inputs_shape = inputs.shape
        c = inputs.shape[1] # c=2: x and y
        # (n, c, h, w) -> (n * c, 1, h, w)
        inputs = inputs.reshape((inputs.shape[0] * c, 1, inputs.shape[2], inputs.shape[3]))

        # (n * c, 1, h, w) -> (n * c, filters, h, w) # filters=1
        
        ### h(x), w(y)
        df_1order_x = torch.diff(
            self.boundary(inputs, padding=(0,0,0,1), mode=boundary_mode, value=boundary_value) 
            , n=1, dim=-2) 
        df_1order_y = torch.diff(
            self.boundary(inputs, padding=(0,1,0,0), mode=boundary_mode, value=boundary_value) 
            , n=1, dim=-1)
        
        df_1order_x_ = torch.diff(
            self.boundary(inputs, padding=(0,0,1,0), mode=boundary_mode, value=boundary_value) 
            , n=1, dim=-2) 
        df_1order_y_ = torch.diff(
            self.boundary(inputs, padding=(1,0,0,0), mode=boundary_mode, value=boundary_value) 
            , n=1, dim=-1)
        
        df_2order_x = 0.5 * (df_1order_x + df_1order_x_)
        df_2order_y = 0.5 * (df_1order_y + df_1order_y_)

        # du_x, dv_x
        df_1order_x = df_1order_x.view(inputs_shape)
        # du_y, dv_y
        df_1order_y = df_1order_y.view(inputs_shape)
        # du_x, dv_x
        df_2order_x = df_2order_x.view(inputs_shape)
        # du_y, dv_y
        df_2order_y = df_2order_y.view(inputs_shape)
        
        return df_1order_x, df_1order_y, df_2order_x, df_2order_y


# 空间上的 数值导数 (output: gradient)
class SpatialDerivative(nn.Module): # 2D scalar or vector field
    """
    input_shape: 
        x: (batch_size,) c, h, w
        grid: a scalar or a Tensor (shape as x)
    output_shape(each): 
        (batch_size,) c, h, w
        
    case1: 2D scalar field
        sde = SpatialDerivative().to("cuda")
        f    = torch.arange(1, 10*1*64*64+1, dtype=torch.float32).reshape(10, 1, 64, 64).to("cuda")
        grid = torch.ones(10, 2, 64, 64).to("cuda")
        sde(f, grid, boundary_mode="periodic", boundary_value=0)[0].shape
    
    case2: 2D vector field
        sde = SpatialDerivative().to("cuda")
        f    = torch.arange(1, 10*2*64*64+1, dtype=torch.float32).reshape(10, 2, 64, 64).to("cuda")
        grid = torch.ones(10, 2, 64, 64).to("cuda")
        sde(f, grid, boundary_mode="periodic", boundary_value=0)[0].shape
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.spatial_differential = SpatialDifferential()
        
    def forward(self, f, grid,
                boundary_mode="periodic", boundary_value=0):
        df_1order_x, df_1order_y, df_2order_x, df_2order_y = self.spatial_differential(f,
                                                                boundary_mode=boundary_mode, 
                                                                boundary_value=boundary_value)
        grid_reciprocal = 1 / grid # 倒数
        
        if isinstance(grid, (int, float)):
            dx_r, dy_r = grid_reciprocal, grid_reciprocal
        else:
            dx_r = grid_reciprocal[:,:1,...]
            dy_r = grid_reciprocal[:,-1:,...]
        
        # du/dx, dv/dx
        dfdx_1order = df_1order_x * dx_r
        # du/dy, dv/dy
        dfdy_1order = df_1order_y * dy_r
        # du/dx, dv/dx
        dfdx_2order = df_2order_x * dx_r
        # du/dy, dv/dy
        dfdy_2order = df_2order_y * dy_r
        
        return dfdx_1order, dfdy_1order, dfdx_2order, dfdy_2order
    

#%%

# 空间上的 数值导数的 修正 - dfdxi
class SpatialOperator(nn.Module): 
    """
    input_shape: 
        dfdx(each):
            (batch_size,) c, h, w
        grid:
            (batch_size,) c, h, w
    output_shape(each): 
        (batch_size,) c, h, w
        
    case:
        so = SpatialOperator((None, 2, 64, 64), (None, 2, 64, 64), use_isotropic_conv=True).to("cuda")
        dfdx_low  = torch.Tensor(10, 2, 64, 64).to("cuda")
        dfdy_low  = torch.Tensor(10, 2, 64, 64).to("cuda")
        dfdx_high = torch.Tensor(10, 2, 64, 64).to("cuda")
        dfdy_high = torch.Tensor(10, 2, 64, 64).to("cuda")
        grid      = torch.ones(10, 2, 64, 64).to("cuda")
        dfdx_correct, dfdy_correct = so(dfdx_low, dfdy_low, dfdx_high, dfdy_high, grid)
        print(dfdx_correct.shape)
        plt.imshow(dfdx_high[0,0,...].detach().cpu()), plt.show()
        plt.imshow(dfdx_correct[0,0,...].detach().cpu()), plt.show()
        plt.imshow(dfdx_correct[0,0,...].detach().cpu() - dfdx_high[0,0,...].detach().cpu()), plt.show()
    """
    def __init__(
        self,
        shape_in_dfdxi,
        shape_in_grid,
        use_isotropic_conv = True,
        depth_multiple = 1.00, # 0.33, 1.33
        width_multiple = 1.00, # 0.50, 1.25
        Class_UnetBranchOperator = YoloBasedUnetBranchOperatorDemo,
        **kwargs):
        super().__init__(**kwargs)
        self.use_isotropic_conv = use_isotropic_conv
        self.depth_multiple = depth_multiple # 0.33, 1.33
        self.width_multiple = width_multiple # 0.50, 1.25
        assert shape_in_dfdxi == shape_in_grid, f"shape_in_dfdxi={shape_in_dfdxi}  & shape_in_grid={shape_in_grid} must be equal"
        
        self.branch_operator = Class_UnetBranchOperator(
            c_in=shape_in_dfdxi[1] * 2, c_out=shape_in_dfdxi[1],
            use_isotropic_conv=self.use_isotropic_conv,
            depth_multiple=self.depth_multiple,
            width_multiple=self.width_multiple)
        
        b, c, h, w = shape_in_dfdxi
        self.spatial_attention = AdaptiveMultiHeadCrossAttention(
            (b, 1, c, h, w), 
            (b, 1, c, h, w), (b, 1, c, h, w),
            use_isotropic_conv=self.use_isotropic_conv)
        
        self.bn_x = nn.BatchNorm2d(c)
        self.bn_y = nn.BatchNorm2d(c)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, dfdx_low, dfdy_low, dfdx_high, dfdy_high, grid,
                no_constrain=False): # remove hard condstrain of operator's output
        dfdx_low, dfdy_low, dfdx_high, dfdy_high, grid = dfdx_low.detach(), dfdy_low.detach(), dfdx_high.detach(), dfdy_high.detach(), grid.detach()

        dfdx_temp = self.branch_operator( # x_operator with x kernel
            torch.cat([dfdx_low, dfdx_high], dim=1)
            )
        dfdy_temp = self.branch_operator( # y_operator  with rotated x kernel (shared weight)
            torch.cat([dfdy_low, dfdy_high], dim=1)
            )
        
        dfdx_temp_, _ = self.spatial_attention(grid, dfdx_temp, dfdx_temp) # x_attention with x kernel
        dfdx_temp = dfdx_temp + dfdx_temp_
        dfdy_temp_, _ = self.spatial_attention(grid, dfdy_temp, dfdy_temp) # y_attention with rotated x kernel (shared weight)
        dfdy_temp = dfdy_temp + dfdy_temp_
        
        if not no_constrain:
            # Hard constraints
            _a = 2 * (dfdx_low - dfdx_high) + 1e-7 # abs(dfdx_low - dfdx_high)
            _b = 2 * (dfdy_low - dfdy_high) + 1e-7 # abs(dfdy_low - dfdy_high)
            dfdx_temp = self.sigmoid(self.bn_x(dfdx_temp) / _a)
            dfdy_temp = self.sigmoid(self.bn_y(dfdy_temp) / _b)
            dfdx_correct = (dfdx_temp - 0.5) * _a + dfdx_high
            dfdy_correct = (dfdy_temp - 0.5) * _b + dfdy_high
        else:
            dfdx_correct = dfdx_temp + dfdx_high
            dfdy_correct = dfdy_temp + dfdy_high
        return dfdx_correct, dfdy_correct


# SpatialOperator 的 装饰版本（在神经算子输入和输出端分别施加归一化和相应的反归一化，并在forward时额外return算子未反归一化的计算结果，便于单独训练算子）
class WrappedSpatialOperator(SpatialOperator):
    """
    case:
        wso = WrappedSpatialOperator((None, 2, 64, 64), (None, 2, 64, 64), use_isotropic_conv=True).to("cuda")
        dfdx_low  = torch.Tensor(10, 2, 64, 64).to("cuda")
        dfdy_low  = torch.Tensor(10, 2, 64, 64).to("cuda")
        dfdx_high = torch.Tensor(10, 2, 64, 64).to("cuda")
        dfdy_high = torch.Tensor(10, 2, 64, 64).to("cuda")
        grid      = torch.ones(10, 2, 64, 64).to("cuda")
        
        print(dfdx_high)
        
        dfdx_correct, dfdy_correct, dfdx_correct_operator_output, dfdy_correct_operator_output = wso(dfdx_low, dfdy_low, dfdx_high, dfdy_high, grid)
        
        print(dfdx_high)
        
        print(dfdx_correct == dfdx_correct_operator_output, dfdx_correct, dfdx_correct_operator_output)
    """
    def __init__(self, shape_in_dfdxi, *args, **kwargs):
        super().__init__(shape_in_dfdxi, *args, **kwargs)
        b, c, h, w = shape_in_dfdxi
        self.bni_dfdx = BatchNormAndInverse2d(num_features=c, eps=1e-05, momentum=0.1, affine=False)
        self.bni_dfdy = BatchNormAndInverse2d(num_features=c, eps=1e-05, momentum=0.1, affine=False)
        self.bni_grid = BatchNormAndInverse2d(num_features=c, eps=1e-05, momentum=0.1, affine=False)
        
    def forward(self, dfdx_low, dfdy_low, dfdx_high, dfdy_high, grid, 
                no_constrain=False, # remove hard condstrain of operator's output
                no_ni=False): # Disable the normalization and inverse normalization of the inputs and outputs of each ANN module
        
        if not no_ni:
            # norm 
            dfdx_low  = self.bni_dfdx(dfdx_low)
            dfdx_high = self.bni_dfdx(dfdx_high)
            dfdy_low  = self.bni_dfdy(dfdy_low)
            dfdy_high = self.bni_dfdy(dfdy_high)
            grid      = self.bni_grid(grid)
            
        # forward
        [
        dfdx_correct_operator_output, 
        dfdy_correct_operator_output
        ] = super().forward(dfdx_low, dfdy_low, dfdx_high, dfdy_high, grid,
                            no_constrain=no_constrain)
        
        if not no_ni:
            # inverse norm
            dfdx_correct = self.bni_dfdx.inverse(dfdx_correct_operator_output)
            dfdy_correct = self.bni_dfdy.inverse(dfdy_correct_operator_output)
            
            # restore the distribution of input data (maybe needn't?)
            dfdx_low  = self.bni_dfdx.inverse(dfdx_low)
            dfdx_high = self.bni_dfdx.inverse(dfdx_high)
            dfdy_low  = self.bni_dfdy.inverse(dfdy_low)
            dfdy_high = self.bni_dfdy.inverse(dfdy_high)
            grid      = self.bni_grid.inverse(grid)
        else:
            dfdx_correct, dfdy_correct = dfdx_correct_operator_output, dfdy_correct_operator_output
        
        # [dfdx_correct, dfdy_correct] for subsequent PDE solving; 
        # [dfdx_correct_operator_output, dfdy_correct_operator_output] for train this module separately
        return dfdx_correct, dfdy_correct, dfdx_correct_operator_output, dfdy_correct_operator_output


# 空间微分项相关的运算
class SpatialModule(nn.Module):
    """
    input_shape: 
        x: (batch_size,) c, h, w
        grid: (batch_size,) c, h, w
    output_shape(each): 
        (batch_size,) c, h, w
        
    case:
        spatialm = SpatialModule((None, 2, 64, 64), (None, 2, 64, 64), use_isotropic_conv=True).to("cuda")
        f = torch.arange(1, 10*2*64*64+1, dtype=torch.float32).reshape(10, 2, 64, 64).to("cuda")
        grid = torch.ones(10, 2, 64, 64).to("cuda")
        dfdx_correct, dfdy_correct, dfdx_correct_operator_output, dfdy_correct_operator_output = spatialm(f, grid, boundary_mode="periodic", boundary_value=0)
        print(dfdx_correct.shape)
        plt.imshow(f[0,0,...].detach().cpu()), plt.show()
        plt.imshow(spatialm.spatial_derivative(f, grid, boundary_mode="periodic", boundary_value=0)[0][0,0,...].detach().cpu()), plt.show()
        plt.imshow(spatialm.spatial_derivative(f, grid, boundary_mode="periodic", boundary_value=0)[2][0,0,...].detach().cpu()), plt.show()
        plt.imshow(dfdx_correct[0,0,...].detach().cpu()), plt.show()
        plt.imshow(spatialm.spatial_derivative(f, grid, boundary_mode="periodic", boundary_value=0)[2][0,0,...].detach().cpu() - 
                   dfdx_correct[0,0,...].detach().cpu()), plt.show()
    """
    def __init__(
            self,
            shape_in_f,
            shape_in_grid,
            use_isotropic_conv = True,
            depth_multiple = 1.00, # 0.33, 1.33
            width_multiple = 1.00, # 0.50, 1.25
            Class_UnetBranchOperator = YoloBasedUnetBranchOperatorDemo,
            **kwargs):
        super().__init__(**kwargs)
        self.use_isotropic_conv = use_isotropic_conv
        self.depth_multiple = depth_multiple # 0.33, 1.33
        self.width_multiple = width_multiple # 0.50, 1.25
        
        self.spatial_derivative = SpatialDerivative()
        self.spatial_operator   = WrappedSpatialOperator(shape_in_f, shape_in_grid,
            use_isotropic_conv=self.use_isotropic_conv, 
            depth_multiple=self.depth_multiple,
            width_multiple=self.width_multiple,
            Class_UnetBranchOperator=Class_UnetBranchOperator)
        
    def forward(self, f, grid, boundary_mode="periodic", boundary_value=0, only_numerical=False, 
                no_constrain=False, # remove hard condstrain of operator's output
                no_ni=False): # Disable the normalization and inverse normalization of the inputs and outputs of each ANN module
        
        dfdx_1order, dfdy_1order, dfdx_2order, dfdy_2order = self.spatial_derivative(
            f, grid, boundary_mode=boundary_mode, boundary_value=0)
        
        if only_numerical:
            return dfdx_2order, dfdy_2order, None, None
        else:
            dfdx_correct, dfdy_correct, dfdx_correct_operator_output, dfdy_correct_operator_output  = self.spatial_operator(
                dfdx_low = dfdx_1order, 
                dfdy_low = dfdy_1order, 
                dfdx_high = dfdx_2order, 
                dfdy_high = dfdy_2order,
                grid = grid,
                no_constrain = no_constrain,
                no_ni = no_ni)
            return dfdx_correct, dfdy_correct, dfdx_correct_operator_output, dfdy_correct_operator_output 


#%%

# 时间上的变化量 的估计 based on 拉格朗日中值定理（估计中间点的导数） - dfdt_xi # dfdt_ξ (temporal variation ÷ dt)
class TemporalOperator(nn.Module): 
    """
    input_shape: 
        dfdt:
            (batch_size,) c, h, w
        dt:
            (batch_size,) 1(chanel), 1(series_len)
        param:
            (batch_size,) 1(chanel), n_param
        history_dynamic:
            (batch_size,) 1, c_out
        dfdt_previous:
            (batch_size,) c, h, w
    output_shape: 
        (batch_size,) c, h, w
        
    case:
        to = TemporalOperator((None, 2, 64, 64), (None, 1, 5), (None, 1, 1024), use_isotropic_conv=True).to("cuda")
        dfdt            = torch.arange(1, 10*2*64*64+1, dtype=torch.float32).reshape(10, 2, 64, 64).to("cuda")
        dt              = torch.ones(10, 1, 1).to("cuda")
        param           = torch.ones(10, 1, 5).to("cuda")
        history_dynamic = torch.ones(10, 1, 1024).to("cuda")
        dfdt_previous   = dfdt - 1
        dfdt_xi = to(dfdt, dt, param, history_dynamic, dfdt_previous)
        print(dfdt_xi.shape)
        plt.imshow(dfdt[0,0,...].cpu()), plt.show()
        plt.imshow(dfdt_xi[0,0,...].detach().cpu()), plt.show()
        plt.imshow(dfdt_xi[0,0,...].detach().cpu() - dfdt[0,0,...].cpu()), plt.show()
    """
    def __init__(
        self,
        shape_in_dfdt,
        shape_in_param,
        shape_in_history_dynamic,
        use_isotropic_conv = True,
        depth_multiple = 1.00, # 0.33, 1.33
        width_multiple = 1.00, # 0.50, 1.25
        Class_UnetBranchOperator = YoloBasedUnetBranchOperatorDemo,
        **kwargs):
        super().__init__(**kwargs)
        self.use_isotropic_conv = use_isotropic_conv
        self.depth_multiple = depth_multiple # 0.33, 1.33
        self.width_multiple = width_multiple # 0.50, 1.25
        
        self.branch_operator = Class_UnetBranchOperator(
            c_in=shape_in_dfdt[1], c_out=shape_in_dfdt[1],
            use_isotropic_conv=self.use_isotropic_conv,
            depth_multiple=self.depth_multiple,
            width_multiple=self.width_multiple)
        
        b, c, h, w = shape_in_dfdt
        self.channel_attention_current = AdaptiveMultiHeadCrossAttention(
            (None, 1, 1 + shape_in_param[-1]), 
            (b, 1, c, h, w), (b, 1, c, h, w),
            use_isotropic_conv=self.use_isotropic_conv)
        self.channel_attention_history = AdaptiveMultiHeadCrossAttention(
            shape_in_history_dynamic, 
            (b, 1, c, h, w), (b, 1, c, h, w),
            use_isotropic_conv=self.use_isotropic_conv)
        
        self.bn = nn.BatchNorm2d(c)
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.01)
        
    def forward(self, dfdt, dt, param, history_dynamic, dfdt_previous,
                no_constrain=False): # remove hard condstrain of operator's output
        dfdt_xi_temp = self.branch_operator(dfdt)
        
        dfdt_xi_temp_, _ = self.channel_attention_current(
            torch.cat([dt, param], dim=-1), 
            dfdt_xi_temp, dfdt_xi_temp) 
        dfdt_xi_temp = dfdt_xi_temp + dfdt_xi_temp_
        
        dfdt_xi_temp_, _ = self.channel_attention_history(
            history_dynamic, 
            dfdt_xi_temp, dfdt_xi_temp) 
        dfdt_xi_temp = dfdt_xi_temp + dfdt_xi_temp_
        
        if not no_constrain:
            # Hard constraints
            dfdt_xi_temp = self.leaky_relu(self.bn(dfdt_xi_temp))
            sign = torch.sign(dfdt - dfdt_previous)
            dfdt_xi = dfdt + sign * dfdt_xi_temp
        else:
            dfdt_xi = dfdt + dfdt_xi_temp
        return dfdt_xi # dfdt_ξ (temporal variation ÷ dt)


# TemporalOperator 的 装饰版本（在神经算子输入和输出端分别施加归一化和相应的反归一化，并在forward时额外return算子未反归一化的计算结果，便于单独训练算子）
class WrappedTemporalOperator(TemporalOperator):
    """
    case:
        wto = WrappedTemporalOperator((None, 2, 64, 64), (None, 1, 5), (None, 1, 1024), use_isotropic_conv=True).to("cuda")
        dfdt            = torch.arange(1, 10*2*64*64+1, dtype=torch.float32).reshape(10, 2, 64, 64).to("cuda")
        dt              = torch.ones(10, 1, 1).to("cuda")
        param           = torch.ones(10, 1, 5).to("cuda")
        history_dynamic = torch.ones(10, 1, 1024).to("cuda")
        dfdt_previous   = dfdt - 1
        
        print(dfdt)
        
        dfdt_xi, dfdt_xi_operator_output = wto(dfdt, dt, param, history_dynamic, dfdt_previous)
        
        print(dfdt)
        
        print(dfdt_xi == dfdt_xi_operator_output, dfdt_xi, dfdt_xi_operator_output)
    """
    def __init__(self, shape_in_dfdt, *args, **kwargs):
        super().__init__(shape_in_dfdt, *args, **kwargs)
        b, c, h, w = shape_in_dfdt
        self.bni_dfdt = BatchNormAndInverse2d(num_features=c, eps=1e-05, momentum=0.1, affine=False)
        self.bni_dt   = BatchNormAndInverse1d(num_features=1, eps=1e-05, momentum=0.1, affine=False)
        
    def forward(self, dfdt, dt, param, history_dynamic, dfdt_previous, 
                no_constrain=False, # remove hard condstrain of operator's output
                no_ni=False): # Disable the normalization and inverse normalization of the inputs and outputs of each ANN module
        
        if not no_ni:
            # norm 
            dfdt          = self.bni_dfdt(dfdt)
            dfdt_previous = self.bni_dfdt(dfdt_previous)
            dt            = self.bni_dt(dt)
        
        # forward
        dfdt_xi_operator_output = super().forward(dfdt, dt, param, history_dynamic, dfdt_previous,
                                                  no_constrain=no_constrain)
        
        if not no_ni: 
            # inverse norm
            dfdt_xi = self.bni_dfdt.inverse(dfdt_xi_operator_output)
            
            # restore the distribution of input data (maybe needn't?)
            dfdt          = self.bni_dfdt.inverse(dfdt)
            dfdt_previous = self.bni_dfdt.inverse(dfdt_previous)
            dt            = self.bni_dt.inverse(dt)
        else:
            dfdt_xi = dfdt_xi_operator_output
        
        # dfdt_xi for subsequent PDE solving; dfdt_xi_operator_output for train this module separately
        return dfdt_xi, dfdt_xi_operator_output


#%%

# 欧拉法 数值积分 
class Head(nn.Module):
    """
    input_shape: 
        frame:   (batch_size,) c, h, w
        dfdt_xi: (batch_size,) c, h, w
        dt:      (batch_size,) 1(chanel), 1(series_len)
    output_shape: 
        (batch_size,) c, h, w
        
    case:
        head    = Head().to("cuda")
        frame   = torch.arange(1, 10*2*64*64+1, dtype=torch.float32).reshape(10, 2, 64, 64).to("cuda")
        dfdt_xi = 1 + frame
        dt      = torch.ones(10, 1, 1).to("cuda")
        next_frame = head(frame, dfdt_xi, dt)
        print(next_frame.shape)
        plt.imshow(frame[0,0,...].cpu()), plt.show()
        plt.imshow(next_frame[0,0,...].detach().cpu()), plt.show()
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
    def forward(self, frame, dfdt_xi, dt):
        # next_frame
        return frame + dt.reshape(dt.shape[0], 1, 1, 1) * dfdt_xi
    
