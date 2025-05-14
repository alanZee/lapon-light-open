# LaPON, GPL-3.0 license
# ML models

import os, sys

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
from models.common import *
from models.lapon_base import *
from cfd.iteration_method import PoissonEqSolver
import cfd.general as cfdg


#%%

# the YOLO model
class YOLO(nn.Module):
    """
    input_shape: 
        frames: (batch_size,) series_len(>=2), c(x,y), h, w
        grid:   (batch_size,) series_len(>=2), c(x,y), h, w
        dt:     (batch_size,) 1(chanel), series_len(>=2)
        param:  (batch_size,) 1(chanel), n_param
        # force: scaler or (batch_size,) c, h, w
        # pressure: (batch_size,) 1, h, w 
        # boundary_value: scaler or ...
    output_shape: 
        if self.out_mode == 'all': # output: all (Direct output (no inverse norm) of every neural operator (Neural operators are trained separately) etc.)
            [dfdx_operator_output, dfdy_operator_output, dfdt_xi_operator_output, 
            dfdx, dfdy, dfdt_xi, 
            next_frame, current_pressure]:
                (batch_size,) c, h, w
        if self.out_mode == 'normal': 
            next_frame:            
                (batch_size,) c(x,y), h, w
            current_pressure:
                (batch_size,) 1, h, w
        
    case:
        m  = YOLO((None, 2, 2, 64, 64), (None, 2, 2, 64, 64), (None, 1, 2), (None, 1, 3), use_isotropic_conv = True).to("cuda")
        frames   = torch.arange(1, 5*2*2*64*64+1, dtype=torch.float32).reshape(5, 2, 2, 64, 64).to("cuda")
        grid     = torch.ones(5, 2, 2, 64, 64).to("cuda")
        dt       = torch.ones(5, 1, 2).to("cuda")
        param    = torch.ones(5, 1, 3).to("cuda")
        force    = torch.ones(5, 2, 64, 64).to("cuda")
        pressure = torch.ones(5, 1, 64, 64).to("cuda")
        next_frame, current_pressure = m(frames, grid, dt, param,
                                         force=force,
                                         pressure=pressure,
                                         continuously_infer=False, only_numerical=False)
        print(next_frame.shape)
        plt.imshow(frames[0,-1,0,...].cpu().numpy()), plt.show()
        plt.imshow(next_frame[0,0,...].detach().cpu().numpy()), plt.show()
    """
    
    def __init__(
        self, 
        shape_in_frames,
        shape_in_grid, 
        shape_in_dt,
        shape_in_param, # For example: rho(density), nu(viscosity), epsilon(dissipation) (ρ, ν, ε)  for fluid 
        use_isotropic_conv = False, #True,
        stride = 32,
        depth_multiple = 1.00, # 0.33, 1.33
        width_multiple = 1.00, # 0.50, 1.25
        out_mode = 'normal', 
        no_ni = False, 
        **kwargs):
        super().__init__(**kwargs)
        
        self.stride = stride 
        """
        The maximum reduction factor of the CNN module to the input frame size, 
        the input data size must be a multiple of this value.
        """
        self.depth_multiple = depth_multiple # 0.33, 1.33
        self.width_multiple = width_multiple # 0.50, 1.25
        self.out_mode = out_mode
        self.no_ni = no_ni
        
        assert shape_in_frames[-3:] == shape_in_grid[-3:], "the shape of shape_in_frames and shape_in_grid must be equal"
        assert shape_in_frames[1] == shape_in_dt[-1], "the time series lenghth of shape_in_frames and shape_in_dt must be equal"
        t, c, h, w = shape_in_frames[-4:]
        shape_in_param = (shape_in_param[0], shape_in_param[1], shape_in_param[2]+2) # add rms_velocity, Re_lambda
        
        self.module = YoloBasedUnetBranchOperatorDemo(
            c_in=c*2 + 1, # frames, pressure, force
            c_out=c,
            use_isotropic_conv=use_isotropic_conv,
            depth_multiple=self.depth_multiple,
            width_multiple=self.width_multiple,
            )
        
        # BatchNorm2d And Inverse Norm (work on input & output of module)
        self.bni_f = BatchNormAndInverse2d(num_features=c, eps=1e-07, momentum=0.1, affine=False) # 2d
        self.bni_pf = BatchNormAndInverse2d(num_features=c + 1, eps=1e-07, momentum=0.1, affine=False) # 2d
        self.bni_dx = BatchNormAndInverse2d(num_features=c, eps=1e-07, momentum=0.1, affine=False) # 2d
        self.bni_dt = BatchNormAndInverse1d(num_features=1, eps=1e-07, momentum=0.1, affine=False) # 1d
        
    def forward(self, frames, grid, dt, param, # system_parameter
                force=0,
                pressure=None,
                boundary_mode="newman", 
                boundary_value=0, 
                boundary_mode_pressure="newman", 
                boundary_value_pressure=0, 
                continuously_infer=False, # save and use pressure_previous & dfdt_previous
                only_numerical=False, # only numerical mode (no AI operator nearly)
                eps_div=1e-07, # 1e-5(common) or 1e-7(min)
                *args, **kwargs): 
        self.boundary_mode = boundary_mode
        
        ### prepare data
        historical_frames, current_frame = frames[:, :-1, ...], frames[:, -1, ...]
        historical_grid, current_grid = grid[:, :-1, ...], grid[:, -1, ...]
        historical_dt, current_dt = dt[..., :-1], dt[..., -1:]
        
        rho, epsilon, nu = param[..., :1, None], param[..., 1:2, None], param[..., -1:, None]
        
        pf = torch.cat([
            pressure,
            force], dim=1) # n, c, h, w
        
        ### norm 
        if not self.no_ni:
            # norm frame
            current_frame = self.bni_f(current_frame)
            # norm pressure & force
            pf = self.bni_pf(pf)
            # norm grid
            current_grid = self.bni_dx(current_grid)
            # norm dt
            current_dt = self.bni_dt(current_dt)
        
        ### prepare inputs
        inputs = torch.cat([
            current_frame, 
            pf], dim=1) # n, c, h, w
        
        ### forward
        outputs = self.module(inputs)
        
        ### inverse norm 
        if not self.no_ni:
            # inverse norm outputs
            outputs = self.bni_f.inverse(outputs)
            
            # inverse norm frame
            current_frame = self.bni_f.inverse(current_frame)
            # inverse norm pressure & force
            pf = self.bni_pf.inverse(pf)
            # inverse norm grid
            current_grid = self.bni_dx.inverse(current_grid)
            # inverse norm dt
            current_dt = self.bni_dt.inverse(current_dt)
        
        # Output
        if self.out_mode == 'all':
            ### final output: next_frame and None
            return [None]*6 + [outputs, None]
        else:
            ### normal final output: next_frame, None
            return outputs, None
    

#%%
# the DeepONetCNN model
class DeepONetCNN(nn.Module):
    """
    input_shape: 
        frames: (batch_size,) series_len(>=2), c(x,y), h, w
        grid:   (batch_size,) series_len(>=2), c(x,y), h, w
        dt:     (batch_size,) 1(chanel), series_len(>=2)
        param:  (batch_size,) 1(chanel), n_param
        # force: scaler or (batch_size,) c, h, w
        # pressure: (batch_size,) 1, h, w 
        # boundary_value: scaler or ...
    output_shape: 
        if self.out_mode == 'all': # output: all (Direct output (no inverse norm) of every neural operator (Neural operators are trained separately) etc.)
            [dfdx_operator_output, dfdy_operator_output, dfdt_xi_operator_output, 
            dfdx, dfdy, dfdt_xi, 
            next_frame, current_pressure]:
                (batch_size,) c, h, w
        if self.out_mode == 'normal': 
            next_frame:            
                (batch_size,) c(x,y), h, w
            current_pressure:
                (batch_size,) 1, h, w
        
    case:
        m  = DeepONetCNN((None, 2, 2, 64, 64), (None, 2, 2, 64, 64), (None, 1, 2), (None, 1, 3), use_isotropic_conv = True).to("cuda")
        frames   = torch.arange(1, 5*2*2*64*64+1, dtype=torch.float32).reshape(5, 2, 2, 64, 64).to("cuda")
        grid     = torch.ones(5, 2, 2, 64, 64).to("cuda")
        dt       = torch.ones(5, 1, 2).to("cuda")
        param    = torch.ones(5, 1, 3).to("cuda")
        force    = torch.ones(5, 2, 64, 64).to("cuda")
        pressure = torch.ones(5, 1, 64, 64).to("cuda")
        next_frame, current_pressure = m(frames, grid, dt, param,
                                         force=force,
                                         pressure=pressure,
                                         continuously_infer=False, only_numerical=False)
        print(next_frame.shape)
        plt.imshow(frames[0,-1,0,...].cpu().numpy()), plt.show()
        plt.imshow(next_frame[0,0,...].detach().cpu().numpy()), plt.show()
    """
    
    def __init__(
        self, 
        shape_in_frames,
        shape_in_grid, 
        shape_in_dt,
        shape_in_param, # For example: rho(density), nu(viscosity), epsilon(dissipation) (ρ, ν, ε)  for fluid 
        use_isotropic_conv = False, #True,
        stride = 32,
        depth_multiple = 1.00, # 0.33, 1.33
        width_multiple = 1.00, # 0.50, 1.25
        out_mode = 'normal', 
        no_ni = False, 
        **kwargs):
        super().__init__(**kwargs)
        
        self.stride = stride 
        """
        The maximum reduction factor of the CNN module to the input frame size, 
        the input data size must be a multiple of this value.
        """
        self.depth_multiple = depth_multiple # 0.33, 1.33
        self.width_multiple = width_multiple # 0.50, 1.25
        self.out_mode = out_mode
        self.no_ni = no_ni
        
        assert shape_in_frames[-3:] == shape_in_grid[-3:], "the shape of shape_in_frames and shape_in_grid must be equal"
        assert shape_in_frames[1] == shape_in_dt[-1], "the time series lenghth of shape_in_frames and shape_in_dt must be equal"
        t, c, h, w = shape_in_frames[-4:]
        self.c = c # u&v..
        shape_in_param = (shape_in_param[0], shape_in_param[1], shape_in_param[2]+2) # add rms_velocity, Re_lambda
        
        self.cat = Concat(-1)
        
        ### branch
        
        self.branch = YoloBasedEncoder(
            c_in=c*2 + 1, # frames, pressure, force
            use_isotropic_conv=use_isotropic_conv,
            depth_multiple=self.depth_multiple,
            width_multiple=self.width_multiple,
            )
        self.c_latent = c_latent = self.branch.c_latent # i.e. 1024
        
        self.conv = CBS(c_latent, c_latent * c, k=1, s=1, p="valid",
                        use_isotropic_conv=use_isotropic_conv)
        
        # self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        ### trunck
        
        # self.act = nn.SiLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())
        self.act = nn.SiLU()
        
        self.trunck_weight1 = nn.Parameter(
            torch.Tensor(c + 1, c_latent), # c + 1: x, y, (z,) t
            requires_grad=True)
        nn.init.xavier_uniform_(self.trunck_weight1)
        self.trunck_bias1 = nn.Parameter(torch.zeros(c_latent), requires_grad=True)
        
        n_trunck_latent = 3
        self.trunck_weight_ls = nn.ParameterList([ nn.Parameter(
            torch.Tensor(c_latent, c_latent),
            requires_grad=True) for _ in range(n_trunck_latent) ])
        [nn.init.xavier_uniform_(i) for i in self.trunck_weight_ls]
        self.trunck_bias_ls = nn.ParameterList([ nn.Parameter(torch.zeros(c_latent), requires_grad=True)
                                for _ in range(n_trunck_latent) ])
        
        self.trunck_weight2 = nn.Parameter(
            torch.Tensor(c_latent, c_latent), 
            requires_grad=True)
        nn.init.xavier_uniform_(self.trunck_weight2)
        self.trunck_bias2 = nn.Parameter(torch.zeros(c_latent), requires_grad=True)
        
        ### BatchNorm2d And Inverse Norm (work on input & output of module)
        self.bni_f = BatchNormAndInverse2d(num_features=c, eps=1e-07, momentum=0.1, affine=False) # 2d
        self.bni_pf = BatchNormAndInverse2d(num_features=c + 1, eps=1e-07, momentum=0.1, affine=False) # 2d
        # self.bni_dx = BatchNormAndInverse2d(num_features=c, eps=1e-07, momentum=0.1, affine=False) # 2d
        # self.bni_dt = BatchNormAndInverse1d(num_features=1, eps=1e-07, momentum=0.1, affine=False) # 1d
        self.bni_query = BatchNormAndInverse1d(num_features=c + 1, eps=1e-07, momentum=0.1, affine=False) # 1d # c + 1: x, y, (z,) t
        
    def forward(self, frames, grid, dt, param, # system_parameter
                force=0,
                pressure=None,
                boundary_mode="newman", 
                boundary_value=0, 
                boundary_mode_pressure="newman", 
                boundary_value_pressure=0, 
                continuously_infer=False, # save and use pressure_previous & dfdt_previous
                only_numerical=False, # only numerical mode (no AI operator nearly)
                eps_div=1e-07, # 1e-5(common) or 1e-7(min)
                *args, **kwargs): 
        self.boundary_mode = boundary_mode
        
        ### prepare data
        historical_frames, current_frame = frames[:, :-1, ...], frames[:, -1, ...]
        historical_grid, current_grid = grid[:, :-1, ...], grid[:, -1, ...]
        historical_dt, current_dt = dt[..., :-1], dt[..., -1:]
        
        rho, epsilon, nu = param[..., :1, None], param[..., 1:2, None], param[..., -1:, None]
        
        pf = torch.cat([
            pressure,
            force], dim=1) # n, c, h, w
        
        ### norm 
        if not self.no_ni:
            # norm frame
            current_frame = self.bni_f(current_frame)
            # norm pressure & force
            pf = self.bni_pf(pf)
            # # norm grid
            # current_grid = self.bni_dx(current_grid)
            # # norm dt
            # current_dt = self.bni_dt(current_dt)
        
        ### prepare inputs
        inputs = torch.cat([
            current_frame, 
            pf], dim=1) # n, c, h, w
        
        ### prepare sampling_points(query location)
        n, c, h, w = current_grid.shape
        query_x = torch.cumsum(current_grid[:, 0, ...], dim=-2).view(n, h * w, 1) # n, c, h(x), w(y) -> n, h(x) * w(y), 1
        query_y = torch.cumsum(current_grid[:, 1, ...], dim=-1).view(n, h * w, 1) # n, c, h(x), w(y) -> n, h(x) * w(y), 1
        query_t = current_dt.repeat(1, h * w, 1) # n, c(1), series_len(1) -> n, h(x) * w(y), 1
        query = self.cat([query_x, query_y, query_t]) # n, h(x) * w(y), sampling_points(x, y, t)
        
        ### norm 
        if not self.no_ni:
            # query
            query = self.bni_query(query.transpose(-2, -1)).transpose(-2, -1)
        
        ### forward
        # branch
        temp = self.branch(inputs)
        temp_branch = self.max_pool(self.conv(temp)).view(-1, self.c_latent, self.c) # n, c_latent, c(u&v..)
        # trunck
        temp = self.act(query @ self.trunck_weight1 + self.trunck_bias1)
        for trunck_weight, trunck_bias in zip(self.trunck_weight_ls, self.trunck_bias_ls):
            temp = self.act(temp @ trunck_weight + trunck_bias)
        temp_trunck = temp @ self.trunck_weight2 + self.trunck_bias2 # n, sampling_points, c_latent
        # output
        outputs = temp_trunck @ temp_branch
        outputs = outputs.transpose(1, 2).view(n, c, h, w) # n, sampling_points, c(u&v..) -> n, c(u&v..), h(x), w(y)
        
        ### inverse norm 
        if not self.no_ni:
            # inverse norm outputs
            outputs = self.bni_f.inverse(outputs)
            
            # inverse norm frame
            current_frame = self.bni_f.inverse(current_frame)
            # inverse norm pressure & force
            pf = self.bni_pf.inverse(pf)
            # # inverse norm grid
            # current_grid = self.bni_dx.inverse(current_grid)
            # # inverse norm dt
            # current_dt = self.bni_dt.inverse(current_dt)
            
            # query
            query = self.bni_query.inverse(query.transpose(-2, -1)).transpose(-2, -1)
        
        # Output
        if self.out_mode == 'all':
            ### final output: next_frame and None
            return [None]*6 + [outputs, None]
        else:
            ### normal final output: next_frame, None
            return outputs, None

