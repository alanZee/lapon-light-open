# LaPON, GPL-3.0 license
# LaPON top-level encapsulation (for PDE - ns)

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

import cv2


#%%

# （自定义） PDE 一般形式 df/dt = N(f) 中的右边项(the right-hand side of PDE) 的 离散形式
class PDEModule_NS(nn.Module): 
    # 无量纲形式 NS 方程 (NSE)
    """
    input_shape: 
        f:     (batch_size,) c, h, w
        grid:  (batch_size,) c, h, w
        Re_lambda: scaler or (batch_size,) 1(chanel), 1
        # force: scaler or (batch_size,) c, h, w
        # pressure: (batch_size,) 1, h, w 
        # boundary_value: scaler or ...
    output_shape(each): 
        (batch_size,) c, h, w
        
    case1:
        pde_m     = PDEModule_NS((10, 2, 64, 64), (10, 2, 64, 64)).to("cuda")
        f         = torch.arange(1, 10*2*64*64+1, dtype=torch.float32).reshape(10, 2, 64, 64).to("cuda")
        grid      = torch.ones(10, 2, 64, 64).to("cuda")
        Re_lambda = 433
        # force = f
        dfdt, dfdx_operator_output, dfdy_operator_output = pde_m(f, grid, Re_lambda, only_numerical=False)
        dfdt.shape
    
    case2:
        pde_m     = PDEModule_NS((2, 2, 64, 64), (2, 2, 64, 64)).to("cuda")
        f         = torch.arange(1, 2*2*64*64+1, dtype=torch.float32).reshape(2, 2, 64, 64).to("cuda")
        grid      = torch.ones(2, 2, 64, 64).to("cuda")
        Re_lambda = 433
        # force = f
        dfdt, dfdx_operator_output, dfdy_operator_output = pde_m(f, grid, Re_lambda, only_numerical=False)
        dfdt.shape
    """
    
    def __init__(
            self, 
            shape_in_frame,
            shape_in_grid, 
            use_isotropic_conv = True,
            Class_UnetBranchOperator = YoloBasedUnetBranchOperatorDemo,
            stride = 32,
            depth_multiple = 1.00, # 0.33, 1.33
            width_multiple = 1.00, # 0.50, 1.25
            **kwargs):
        super().__init__(**kwargs)
        self.use_isotropic_conv = use_isotropic_conv
        
        self.stride = stride 
        """
        The maximum reduction factor of the CNN module to the input frame size, 
        the input data size must be a multiple of this value.
        """
        self.depth_multiple = depth_multiple # 0.33, 1.33
        self.width_multiple = width_multiple # 0.50, 1.25
        
        self.spatial_module = SpatialModule(
            shape_in_frame,
            shape_in_grid,
            use_isotropic_conv = use_isotropic_conv,
            depth_multiple=self.depth_multiple,
            width_multiple=self.width_multiple,
            Class_UnetBranchOperator = Class_UnetBranchOperator)
        
    def forward(self, f, grid, Re_lambda=433, # f: velocity
                force=0, # Externally given or not(zero defult)
                pressure=0, # Externally given
                boundary_mode="newman", 
                boundary_value=0,
                only_numerical=False,
                no_ni=False, # Disable the normalization and inverse normalization of the inputs and outputs of each ANN module
                no_constrain=False, # remove hard condstrain of operator's output
                ):
        
        ### prepare data
        fx, fy = f[:,:1,...], f[:,1:,...]
        if isinstance(force, (int, float)):
            # force_x, force_y = force, force
            force_x = force_y = force
        elif force.shape[-2] == force.shape[-1] == 1:
            force_x = force_y = force
        else:
            force_x, force_y = force[:,:1,...], force[:,1:,...]
        
        ### spatial term module
        # first derivative
        dfdx, dfdy, dfdx_operator_output, dfdy_operator_output = self.spatial_module(
            f, grid, boundary_mode=boundary_mode, boundary_value=boundary_value,
            only_numerical = only_numerical,
            no_ni = no_ni,
            no_constrain = no_constrain)
        dfxdx, dfydx, dfxdy, dfydy = dfdx[:,:1,...], dfdx[:,1:,...], dfdy[:,:1,...], dfdy[:,1:,...]
        # second derivative
        _dx, _dy, _, _ = self.spatial_module(
            torch.cat([dfxdx, dfydx], dim=1), 
            grid, boundary_mode=boundary_mode, boundary_value=boundary_value,
            only_numerical = only_numerical,
            no_ni = no_ni,
            no_constrain = no_constrain)
        d2fxdx2, d2fydx2, d2fxdxdy, d2fydxdy = _dx[:,:1,...], _dx[:,1:,...], _dy[:,:1,...], _dy[:,1:,...]
        _dx, _dy, _, _ = self.spatial_module(
            torch.cat([dfxdy, dfydy], dim=1), 
            grid, boundary_mode=boundary_mode, boundary_value=boundary_value,
            only_numerical = only_numerical,
            no_ni = no_ni,
            no_constrain = no_constrain)
        d2fxdxdy, d2fydxdy, d2fxdy2, d2fydy2 = _dx[:,:1,...], _dx[:,1:,...], _dy[:,:1,...], _dy[:,1:,...]
        
        if isinstance(pressure, (int, float)):
            dpdx, dpdy = 0, 0
        else:
            # first derivative
            dpdx, dpdy, _, _ = self.spatial_module(
                pressure, grid, boundary_mode=boundary_mode, boundary_value=boundary_value,
                only_numerical = True,
                no_ni = no_ni,
                no_constrain = no_constrain)
            # second derivative
            # _dx, _dy, _, _ = self.spatial_module(
            #     torch.cat([dpdx, dpdy], dim=1), 
            #     grid, boundary_mode=boundary_mode, boundary_value=boundary_value,
            #     only_numerical = True,
            #     no_ni = no_ni,
            #     no_constrain = no_constrain)
            # d2pdx2, d2pdxdy, d2pdxdy, d2pdy2 = _dx[:,:1,...], _dx[:,1:,...], _dy[:,:1,...], _dy[:,1:,...]
        
        
        ### PDE
        
        # the non-dimensional form NS equation (2D)
        dfdt = cfdg.nse2d_nd(
                fx, fy,
                dfxdx, dfxdy,
                dfydx, dfydy,
                d2fxdx2, d2fxdy2,
                d2fydx2, d2fydy2,
                dpdx, dpdy,
                force_x, force_y,
                Re=Re_lambda
                )
        
        return [dfdt, 
                dfdx_operator_output, dfdy_operator_output,
                dfdx, dfdy] 
    

#%%

# the LaPON model
class LaPON(nn.Module):
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
        
    case0:
        lapon  = LaPON((None, 2, 2, 64, 64), (None, 2, 2, 64, 64), (None, 1, 2), (None, 1, 3), use_isotropic_conv = True,
                       out_mode = 'all').to("cuda")
        frames = torch.arange(1, 5*2*2*64*64+1, dtype=torch.float32).reshape(5, 2, 2, 64, 64).to("cuda")
        grid   = torch.ones(5, 2, 2, 64, 64).to("cuda")
        dt     = torch.ones(5, 1, 2).to("cuda")
        param  = torch.ones(5, 1, 3).to("cuda")
        pressure = torch.ones(5, 1, 64, 64).to("cuda")
        lapon.init4infer_new()
        lapon.train()
        [dfdx_operator_output, dfdy_operator_output, dfdt_xi_operator_output, 
        dfdx, dfdy, dfdt_xi, 
        next_frame, current_pressure] = lapon(frames, grid, dt, param,
                                              pressure=pressure,
                                              continuously_infer=False, only_numerical=False)
        
        # test attention module
        size = 128 
        frames = torch.arange(1, 5*2*2*size*size+1, dtype=torch.float32).reshape(5, 2, 2, size, size).to("cuda")
        grid   = torch.ones(5, 2, 2, size, size).to("cuda")
        dt     = torch.ones(5, 1, 2).to("cuda")
        param  = torch.ones(5, 1, 3).to("cuda")
        pressure = torch.ones(5, 1, size, size).to("cuda")
        lapon.init4infer_new()
        lapon.train()
        [dfdx_operator_output, dfdy_operator_output, dfdt_xi_operator_output, 
        dfdx, dfdy, dfdt_xi, 
        next_frame, current_pressure] = lapon(frames, grid, dt, param,
                                              pressure=pressure,
                                              continuously_infer=False, only_numerical=False)
        
    case1:
        lapon  = LaPON((None, 2, 2, 64, 64), (None, 2, 2, 64, 64), (None, 1, 2), (None, 1, 3), use_isotropic_conv = True).to("cuda")
        frames = torch.arange(1, 5*2*2*64*64+1, dtype=torch.float32).reshape(5, 2, 2, 64, 64).to("cuda")
        grid   = torch.ones(5, 2, 2, 64, 64).to("cuda")
        dt     = torch.ones(5, 1, 2).to("cuda")
        param  = torch.ones(5, 1, 3).to("cuda")
        lapon.init4infer_new()
        next_frame, current_pressure = lapon(frames, grid, dt, param,
                                             continuously_infer=False, only_numerical=False)
        print(next_frame.shape)
        plt.imshow(current_pressure[0,-1,...].detach().cpu()), plt.show()
        plt.imshow(frames[0,-1,0,...].cpu()), plt.show()
        plt.imshow(next_frame[0,0,...].detach().cpu()), plt.show()
    
    case2:
        lapon  = LaPON((None, 2, 2, 64, 64), (None, 2, 2, 64, 64), (None, 1, 2), (None, 1, 3), use_isotropic_conv = True).to("cuda")
        frames = torch.arange(1, 1*2*2*64*64+1, dtype=torch.float32).reshape(1, 2, 2, 64, 64).to("cuda")
        grid   = torch.ones(1, 2, 2, 64, 64).to("cuda")
        dt     = torch.ones(1, 1, 2).to("cuda")
        param  = torch.ones(1, 1, 3).to("cuda")
        lapon.init4infer_new()
        next_frame, current_pressure = lapon(frames, grid, dt, param,
                                             continuously_infer=False, only_numerical=False)
        print(next_frame.shape)
        plt.imshow(current_pressure[0,-1,...].detach().cpu()), plt.show()
        plt.imshow(frames[0,-1,0,...].cpu()), plt.show()
        plt.imshow(next_frame[0,0,...].detach().cpu()), plt.show()
    
    case3:
        lapon  = LaPON((None, 2, 2, 64, 64), (None, 2, 2, 64, 64), (None, 1, 2), (None, 1, 3), use_isotropic_conv = True).to("cuda")
        frames = torch.arange(1, 1*2*2*64*64+1, dtype=torch.float32).reshape(1, 2, 2, 64, 64).to("cuda")
        grid   = torch.ones(1, 2, 2, 64, 64).to("cuda")
        dt     = 0.002 * torch.ones(1, 1, 2).to("cuda")
        param  = torch.ones(1, 1, 3).to("cuda")
        lapon.init4infer_new()
        next_frame, current_pressure = lapon(frames, grid, dt, param,
                                             continuously_infer=False, only_numerical=False)
        print(next_frame.shape)
        # plt.imshow(lapon.get_pressure(frames[:,-1,...], grid[:,-1,...])[0,0,...].detach().cpu()), plt.show()
        plt.imshow(current_pressure[0,-1,...].detach().cpu()), plt.show()
        plt.imshow(frames[0,-1,0,...].cpu()), plt.show()
        plt.imshow(next_frame[0,0,...].detach().cpu()), plt.show()

    #########################################################
    
    Args:
        frames:
            All frames of the physics field.
            The last frame is the current frame, 
            and the others are historical frames. 
            In order to calculate the second derivative of time in the TemporalOperator, 
            there must be at least one history frame.
        grid:
            The grid for each frame.
        dt:
            The time step from each frame to the next.
        param:
            The intrinsic physical properties of the physics field.
    
    #########################################################
    
    Architecture:
        HistoricalDynamicExtractor;
        PDEModule[
            SpatialModule[
                SpatialDerivative[
                    SpatialDifferential], 
                SpatialOperator], 
            PDE]
        TemporalOperator
        Head
    """
    
    def __init__(
        self, 
        shape_in_frames,
        shape_in_grid, 
        shape_in_dt,
        shape_in_param, # For example: rho(density), nu(viscosity), epsilon(dissipation) (ρ, ν, ε)  for fluid 
        is_ode = False, 
        use_isotropic_conv = True,
        # You can define your own of the following classes, 
        # the only thing you need to pay attention to 
        # is the matching of interfaces between modules.
        # Besides, the "param" may be used in the Class_PDEModule,
        # and different Class_PDEModule have different inputs.
        Class_UnetBranchOperator = YoloBasedUnetBranchOperatorDemo,
        Class_PDEModule = PDEModule_NS, 
        stride = 32,
        depth_multiple = 1.00, # 0.33, 1.33
        width_multiple = 1.00, # 0.50, 1.25
        out_mode = 'normal', # 'normal': next_frame, current_pressure; 'all': Direct output (no inverse norm) of every neural operator etc. (for Neural operators are trained separately)
        no_ni = False, # Disable the normalization and inverse normalization of the inputs and outputs of each ANN module
        operator_cutoff = False, # cut off gradient flow of operators (for Neural operators are trained separately)
        no_constrain=False, # remove hard condstrain of operator's output
        **kwargs):
        super().__init__(**kwargs)
        self.use_isotropic_conv = use_isotropic_conv
        assert out_mode in ['normal', 'all'], f"out_mode={out_mode} must be in ['normal', 'all']!"
        self.out_mode = out_mode
        self.no_ni = no_ni
        self.operator_cutoff = operator_cutoff
        self.no_constrain = no_constrain
        
        self.stride = stride 
        """
        The maximum reduction factor of the CNN module to the input frame size, 
        the input data size must be a multiple of this value.
        """
        
        self.depth_multiple = depth_multiple # 0.33, 1.33
        self.width_multiple = width_multiple # 0.50, 1.25
        
        c_out = int(1024 * self.width_multiple)
        
        assert shape_in_frames[-3:] == shape_in_grid[-3:], "the shape of shape_in_frames and shape_in_grid must be equal"
        assert shape_in_frames[1] == shape_in_dt[-1], "the time series lenghth of shape_in_frames and shape_in_dt must be equal"
        
        t, c, h, w = shape_in_frames[-4:]
        shape_in_param = (shape_in_param[0], shape_in_param[1], shape_in_param[2]+2) # add rms_velocity, Re_lambda
        
        # BatchNorm2d And Inverse Norm (work on input of hd_extractor module)
        self.bn_e_f  = nn.BatchNorm2d(num_features=c, eps=1e-07, momentum=0.1, affine=False)
        self.bn_e_dx = nn.BatchNorm2d(num_features=c, eps=1e-07, momentum=0.1, affine=False)
        self.bn_e_dt = nn.BatchNorm1d(num_features=1, eps=1e-07, momentum=0.1, affine=False)
        
        self.hd_extractor = HistoricalDynamicExtractor(
            (None, t-1, c, h, w),
            (None, t-1, c, h, w),
            (None, 1, t-1),
            shape_in_param,
            c_out = c_out,
            use_isotropic_conv = use_isotropic_conv)
        self.pde_module = Class_PDEModule(
            (None, c, h, w),
            (None, c, h, w),
            use_isotropic_conv = use_isotropic_conv,
            Class_UnetBranchOperator = Class_UnetBranchOperator,
            stride = self.stride,
            depth_multiple=self.depth_multiple,
            width_multiple=self.width_multiple)
        # self.t_operator = TemporalOperator(
        #     (None, c, h, w),
        #     shape_in_param,
        #     (None, 1, c_out),
        #     use_isotropic_conv = use_isotropic_conv,
        #     Class_UnetBranchOperator = Class_UnetBranchOperator)
        self.t_operator = WrappedTemporalOperator(
            (None, c, h, w),
            shape_in_param,
            (None, 1, c_out),
            use_isotropic_conv = use_isotropic_conv,
            depth_multiple=self.depth_multiple,
            width_multiple=self.width_multiple,
            Class_UnetBranchOperator = Class_UnetBranchOperator)
        self.head = Head()
        
        self.bp_ = BoundaryPadding()
        
        self.dfdt_previous = None
        self.pressure_previous = None
        
        print("NOTE !!! Auto init is ok. It's better to run the self.init4infer_new() method every time you infer a new field")
        
    # NOTE Each time an inference is done for a new flow field, it must be run manually
    def init4infer_new(self): 
        self.dfdt_previous = None
        self.pressure_previous = None
        try:
            self.pde_module.init4infer_new()
        except:
            pass

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
                filter_output=False, # The output is filtered to prevent numerical oscillations
                filter_kernel_size=3, # 3 7
                gauss_filter=False,
                sigma_gaussian_filter_kernel=0.5, # 0.5 1.5
                ):
        self.boundary_mode = boundary_mode
        if not continuously_infer:
            # self.dfdt_previous = None
            # self.pressure_previous = None
            self.init4infer_new()
        
        ### prepare data
        historical_frames, current_frame = frames[:, :-1, ...], frames[:, -1, ...]
        historical_grid, current_grid = grid[:, :-1, ...], grid[:, -1, ...]
        historical_dt, current_dt = dt[..., :-1], dt[..., -1:]
        
        rho, epsilon, nu = param[..., :1, None], param[..., 1:2, None], param[..., -1:, None]
        
        ### characteristic params
        E_point, E_total, rms_velocity = self.get_rms_velocity(current_frame, current_grid)
        Taylor_micro_scale = self.get_Taylor_micro_scale(rms_velocity, nu, epsilon)
        Re_lambda = self.get_Re(rms_velocity, Taylor_micro_scale, nu)
        characteristic_time = (Taylor_micro_scale / rms_velocity).reshape(-1, 1, 1)
        characteristic_force = (rms_velocity / characteristic_time.reshape(-1, 1, 1, 1))
        characteristic_pressure = (rho * # rho_water: 1e3; rho_air: 1.29
                                   rms_velocity**2
                                   ) 
        
        param = torch.cat([
            param, rms_velocity[...,0], Re_lambda[...,0] # 尽量特征之间线性无关
            ], dim=-1)
        
        ### nondimensionalization (non-dimensional)
        # historical_frames /= rms_velocity.reshape(-1, 1, 1, 1, 1) # 需要计算grad的部分不能进行in-place操作
        historical_frames = historical_frames / (rms_velocity.reshape(-1, 1, 1, 1, 1) + eps_div)
        current_frame     = current_frame / (rms_velocity + eps_div)
        historical_grid   = historical_grid / (Taylor_micro_scale.reshape(-1, 1, 1, 1, 1) + eps_div)
        current_grid      = current_grid / (Taylor_micro_scale + eps_div)
        historical_dt     = historical_dt / (characteristic_time + eps_div)
        current_dt        = current_dt / (characteristic_time + eps_div)
        force             = force / (characteristic_force + eps_div)
        if pressure is not None:
            pressure = pressure / (characteristic_pressure + eps_div)
        
        ################################################
        ################################################
        
        ### get current pressure
        if pressure is None:
            pressure = self.get_pressure(current_frame, current_grid, 
                                         boundary_mode_pressure=boundary_mode_pressure, 
                                         boundary_value_pressure=boundary_value_pressure,
                                         only_numerical=only_numerical)
        if continuously_infer:
            self.pressure_previous = pressure
        
        ### get dfdt_previous
        if not only_numerical: 
            if continuously_infer and self.dfdt_previous is not None:
                dfdt_previous = self.dfdt_previous
            else:
                dfdt_previous = self.get_dfdt_previous(
                    historical_frames[:, -1, ...], 
                    historical_grid[:, -1, ...], 
                    Re_lambda,
                    pressure_near=pressure).detach()
        
        ################################################
        
        ### calculate
        
        # norm & historical dynamic extract
        if not only_numerical: 
            if not self.no_ni:
                # norm frames
                t, c, h, w = historical_frames.shape[-4:]
                historical_frames = historical_frames.reshape(-1, c, h, w)
                historical_frames = self.bn_e_f(historical_frames).reshape(-1, t, c, h, w)
                # norm grid
                historical_grid = historical_grid.reshape(-1, c, h, w)
                historical_grid = self.bn_e_dx(historical_grid).reshape(-1, t, c, h, w)
                # norm dt
                historical_dt = self.bn_e_dt(historical_dt)
            # extract
            historical_dynamic = self.hd_extractor(
                historical_frames, 
                historical_grid, 
                historical_dt, 
                param,
                boundary_mode = boundary_mode, 
                boundary_value = boundary_value)
            
        # inputs of Customize Class_PDEModule
        [dfdt_curent, 
         dfdx_operator_output, dfdy_operator_output,
         dfdx, dfdy] = self.pde_module( # 空间微分解耦出来,避免重复计算
            current_frame, 
            current_grid, 
            Re_lambda,
            force = force,
            pressure = pressure,
            boundary_mode = boundary_mode, 
            boundary_value = boundary_value,
            only_numerical = only_numerical,
            no_ni = self.no_ni,
            no_constrain = self.no_constrain)

        # cut off gradient flow of operators (for Neural operators are trained separately)
        if self.operator_cutoff:
            dfdt_curent = dfdt_curent.detach()
        
        # dfdt_curent -> dfdt_ξ
        if not only_numerical:
            dfdt_xi, dfdt_xi_operator_output = self.t_operator(
                dfdt_curent,
                current_dt,
                param,
                historical_dynamic,
                dfdt_previous,
                no_ni = self.no_ni,
                no_constrain = self.no_constrain)
        else:
            dfdt_xi = dfdt_curent
            
        next_frame = self.head(
            current_frame, 
            dfdt_xi, 
            current_dt)
        
        ################################################
        
        ### record dfdt
        if not only_numerical: 
            if continuously_infer:
                # self.dfdt_previous = dfdt_curent
                self.dfdt_previous = dfdt_xi.detach() # dfdt_ξ
        
        ################################################
        ################################################
        
        ### inverse nondimensionalization (not non-dimensional)
        next_frame       = next_frame * (rms_velocity + eps_div) 
        current_pressure = pressure * (characteristic_pressure + eps_div)
        
        # restore the distribution of input data 
        historical_frames = historical_frames * (rms_velocity.reshape(-1, 1, 1, 1, 1) + eps_div)
        current_frame     = current_frame * (rms_velocity + eps_div)
        historical_grid   = historical_grid * (Taylor_micro_scale.reshape(-1, 1, 1, 1, 1) + eps_div)
        current_grid      = current_grid * (Taylor_micro_scale + eps_div)
        historical_dt     = historical_dt * (characteristic_time + eps_div)
        current_dt        = current_dt * (characteristic_time + eps_div)
        force             = force * (characteristic_force + eps_div)
        
        ################################################
        ################################################
        
        # print(dfdt_xi_operator_output == dfdt_xi, dfdt_xi_operator_output, dfdt_xi)
        
        # Box Filter of output
        # https://turbulence.pha.jhu.edu/docs/Database-functions.pdf
        if filter_output:
            dx = grid[0, 0, 0, 0, 0].item()
            n, c, h, w = next_frame.shape
            pad_ = int(filter_kernel_size//2)
            if not gauss_filter:
                filter_kernel = torch.full((1, 1, filter_kernel_size, filter_kernel_size), 1)
            else:
                gaussian_kernel_x = cv2.getGaussianKernel(filter_kernel_size, sigma_gaussian_filter_kernel)
                gaussian_kernel_y = gaussian_kernel_x.T
                # gaussian_kernel_2d
                filter_kernel = gaussian_kernel_x * gaussian_kernel_y
                filter_kernel = filter_kernel / filter_kernel.sum() * (filter_kernel_size**2)
                filter_kernel = torch.from_numpy(filter_kernel).view(1, 1, filter_kernel_size, filter_kernel_size)
            filter_kernel = nn.Parameter(
                        filter_kernel,
                        requires_grad=False).float().to(next_frame.device)
            # filter: 1/S * sum(f * ds)
            next_frame = ( 
                (1 / (filter_kernel_size * dx)**2) * 
                F.conv2d(
                    self.bp_(
                        (next_frame * 
                         (grid[:, -1, 0:1, ...] * grid[:, -1, 1:2, ...]) # ds = dx * dy
                         ),
                        padding=(pad_,pad_,pad_,pad_), mode="newman", value=0
                        ).view(n*c, 1, h + 2*pad_, w + 2*pad_),
                    filter_kernel,
                    bias = None, 
                    stride=(1, 1), 
                    padding=0,
                    groups=1
                    ).view(n, c, h, w)
                )
        
        # Output
        if self.out_mode == 'all':
            ### final output: Direct output (no inverse norm) of every neural operator etc. (for Neural operators are trained separately)
            return [dfdx_operator_output, dfdy_operator_output, dfdt_xi_operator_output, 
                    dfdx, dfdy, dfdt_xi, 
                    next_frame, current_pressure]
        else:
            ### normal final output: next_frame, current_pressure
            return next_frame, current_pressure
    
    ###=====================================================================###
    
    # Trial calculation (Rough calculation)
    def get_dfdt_previous(self, f, grid, Re_lambda, pressure_near=None):
        dfdt, _, _, _, _ = self.pde_module(f, grid, Re_lambda, 
                               force=0,
                               pressure=pressure_near if pressure_near is not None else self.get_pressure(f, grid),
                               boundary_mode=self.boundary_mode, 
                               boundary_value=0,
                               no_ni = self.no_ni,
                               no_constrain = self.no_constrain)
        return dfdt
    
    ###=====================================================================###
    
    def get_pressure( # 2D 
            self, 
            velocity,
            grid, 
            dwdx=0, dwdy=0, 
            dudz=0, dvdz=0, dwdz=0, 
            rho=1e3,# rho_water: 1e3; rho_air: 1.29
            boundary_mode_pressure="newman", 
            boundary_value_pressure=0,
            only_numerical=False): 
        dfdx, dfdy, _, _ = self.pde_module.spatial_module(
                velocity, grid, 
                boundary_mode=boundary_mode_pressure, boundary_value=boundary_value_pressure,
                only_numerical = only_numerical,
                no_ni = self.no_ni,
                no_constrain = self.no_constrain)
        dudx, dvdx, dudy, dvdy = dfdx[:,:1,...], dfdx[:,1:,...], dfdy[:,:1,...], dfdy[:,1:,...]
        
        right_hand_poisson = - self.pressure_poisson3D(
            0, 0, 0, 
            dudx, dvdx, dwdx, 
            dudy, dvdy, dwdy, 
            dudz, dvdz, dwdz, 
            rho)
        
        # pressure field init
        if self.pressure_previous is not None:
            pressure_init = self.pressure_previous
        else:
            pressure_init = torch.zeros(*right_hand_poisson.shape, device=right_hand_poisson.device)
        
        pressure = self._get_pressure(right_hand_poisson, 
                                      grid[0,0,0,0], grid[0,1,0,0],
                                      pressure_init,
                                      boundary_mode_pressure=boundary_mode_pressure, 
                                      boundary_value_pressure=boundary_value_pressure)
        return pressure
            
    def _get_pressure(self, right_hand_poisson, dx, dy, 
                      pressure_init=None, 
                      boundary_mode_pressure="newman", 
                      boundary_value_pressure=0): 
        
        # 2D field SOR iteration
        poissonEq_solver = PoissonEqSolver(
            nx = right_hand_poisson.shape[-1], 
            ny = right_hand_poisson.shape[-2],
            dx = dx, 
            dy = dy, 
            mode = "SOR", 
            boundary_mode = boundary_mode_pressure).to(right_hand_poisson.device)
        pressure = poissonEq_solver(
            force = right_hand_poisson, 
            init_solution = pressure_init,
            max_iter = 1000, tol = 1e-6,
            omiga = 1.8,
            boundary_value = boundary_value_pressure)
        del poissonEq_solver
        
        return pressure
    
    ###=====================================================================###
    
    @staticmethod
    def get_rms_velocity(velocity, grid): 
        E_point, E_total, rms_velocity = cfdg.get_rms_velocity(velocity, grid)
        return E_point, E_total, rms_velocity
    
    @staticmethod
    def get_Taylor_micro_scale(rms_velocity, nu, epsilon): # Taylor Micro. Scale
        Taylor_micro_scale = cfdg.get_Taylor_micro_scale(rms_velocity, nu, epsilon)
        return Taylor_micro_scale
    
    @staticmethod
    def get_Re(velocity, lenghth, nu): # Taylor-scale Reynolds: Re_lambda = get_Re(rms_velocity, Taylor_micro_scale, nu)
        Re = cfdg.get_Re(velocity, lenghth, nu)
        return Re
    
    ###=====================================================================###
    
    @staticmethod
    def continuity3D(dudx, dvdy, dwdz): # continuity equation (3D field)
        return cfdg.continuity3D(dudx, dvdy, dwdz)
    
    @staticmethod
    def pressure_poisson3D( # pressure poisson equation (3D field)
            d2pdx2, d2pdy2, d2pdz2, 
            dudx, dvdx, dwdx,
            dudy, dvdy, dwdy,
            dudz, dvdz, dwdz,
            rho=1e3, # rho_water: 1e3; rho_air: 1.29
            non_dimensional=True):
    
        return cfdg.pressure_poisson3D( 
                d2pdx2, d2pdy2, d2pdz2, 
                dudx, dvdx, dwdx,
                dudy, dvdy, dwdy,
                dudz, dvdz, dwdz,
                rho=rho,
                non_dimensional=non_dimensional)
    
    ###=====================================================================###
    
    @staticmethod
    def trans3Dto2D_NS(
        force_x, force_y, 
        fz, dfxdz, dfydz, 
        d2fxdz2, d2fydz2, 
        Re_lambda=433, nu=0.000185):
        
        return cfdg.trans3Dto2D_NS( # the non-dimensional form NS equation
            force_x, force_y, 
            fz, dfxdz, dfydz, 
            d2fxdz2, d2fydz2, 
            Re_lambda=Re_lambda, nu=nu)
    
    @staticmethod
    def trans3Dto2D_poisson(
        force, 
        d2pdz2, 
        dudz, dwdx, dvdz, dwdy, dwdz,
        rho=1e3):
        
        # the non-dimensional form pressure poisson equation
        return cfdg.trans3Dto2D_poisson(
            force, 
            d2pdz2, 
            dudz, dwdx, dvdz, dwdy, dwdz,
            rho=rho)
    
