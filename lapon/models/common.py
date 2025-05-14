# LaPON, GPL-3.0 license
# Common ANN modules

from copy import deepcopy
import numpy as np
import matplotlib.pyplot as plt
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F

get_conv_out_size = lambda x, k, s: int((x - k) // s + 1)


class BatchNormAndInverse2d(nn.Module):
    # BatchNorm2d And Inverse Norm (work on input & output of a ANN(model/module))
    """
    case:
        bni = BatchNormAndInverse2d(2)
        # x = torch.Tensor([ [[[1,1],[2,2]], [[1,1],[2,2]]] , [[[2,2],[4,4]], [[2,2],[4,4]]] , [[[3,3],[6,6]], [[3,3],[6,6]]] ])
        # x = x.transpose(1,-1)
        x = torch.arange(1, 10*2*64*64+1, dtype=torch.float32).reshape(10, 2, 64, 64)
        for i in range(100): # track sufficiently
            y = bni(x)
        print(y)
        
        print(bni.bn.running_mean.shape, bni.bn.running_var.shape)
    
        mean = bni.bn.running_mean 
        var = bni.bn.running_var
        print(mean, var)
    
        x_ = bni.inverse(y)
        print(x, x_)
    """
    def __init__(self, num_features, eps=1e-07, momentum=0.1, affine=False): # eps:1e-5(common) or 1e-7(min)
        super().__init__()
        self.bn = nn.BatchNorm2d(num_features, eps=eps, momentum=momentum, affine=affine)
        self.trans_shape = (1, -1, 1, 1) # n, c, h, w
    
    def forward(self, x):
        return self.bn(x) # forward and track the running mean and variance 
        # track(update): arg_new = (1 - momentum) * arg_old + momentum * arg_current
    
    def inverse(self, x):
        mean = self.bn.running_mean.reshape(self.trans_shape)
        var = self.bn.running_var.reshape(self.trans_shape)
        if not self.bn.affine:
            return x * (var + self.bn.eps)**0.5 + mean 
        else:
            weight = self.bn.weight.reshape(self.trans_shape)
            bias = self.bn.bias.reshape(self.trans_shape)
            return (x - bias) / (weight + self.bn.eps) * (var + self.bn.eps)**0.5 + mean
        

class BatchNormAndInverse1d(BatchNormAndInverse2d):
    def __init__(self, num_features, eps=1e-07, momentum=0.1, affine=False): # eps:1e-5(common) or 1e-7(min)
        nn.Module.__init__(self)
        self.bn = nn.BatchNorm1d(num_features, eps=eps, momentum=momentum, affine=affine)
        self.trans_shape = (1, -1, 1) # n, c, len


def autopad(k, p="SAME"):  # kernel, padding mode or padding
    k = k if isinstance(k, int) else k[0]
    
    if isinstance(p, int):
        return (p, p)
    elif isinstance(p, (list, tuple)):
        return p
    elif isinstance(p, str):
        if p in ("SAME", "same"):
            p = k // 2 if isinstance(k, int) else [x // 2 for x in k] 
        elif p in ("VALID", "valid"):
            p = 0
        else:
            raise ValueError(f'Received p = {p} is not acceptable, '
                             'it must be in ("SAME", "same", "VALID", "valid")') 
        return (p, p)
    else:
        raise ValueError(f'Received p = {p} is not acceptable.') 


class BoundaryPadding(nn.Module):
    # Applies padding for boundary conditions 
    def __init__(self):  
        super().__init__()
        self.mode_dict = {
            "dirichlet": "constant", # y=C
            "newman": "replicate", # for y'=0
            "periodic": "circular", # y(0,t)=y(L,t)
            "symmetry": "reflect" # y(-x,t)=y(x,t)
        } # ["dirichlet", "newman", "circular", "symmetry"]
    def forward(self, x, padding=(32,32,32,32), mode="periodic", value=0): # x shape: (n,c,h,w) # padding = left, right, top, bottom
        self.mode = mode.lower()
        assert self.mode in self.mode_dict.keys(), f'Received mode is {self.mode}, but it must be in {self.mode_dict.keys()}'
        return F.pad(x, padding, self.mode_dict[self.mode], value)


# 重要通用模块：各向同性卷积(天然满足旋转对称性)
class IsotropicConv2D(nn.Module):
    # 在笛卡尔坐标系(直角坐标系)下，基于曼哈顿距离(出租车距离)/欧式距离 和 共享权重，定义各向同性卷积核，实现各向同性卷积
    # 该卷积核的权值只与到中心center的 曼哈顿距离(出租车距离)/欧式距离 有关，与方向(只有 横向方向 和 纵向方向)无关
    # 该卷积相较于传统卷积，功能会受到一定限制/约束，但仍然可以实现常见的 如直接的边缘检测(梯度计算)：卷积核[1, -2, 1]
    # 各向同性的约束，参数量更小，也同样适用于CV领域
    """
    case1:
        ic = IsotropicConv2D(3, kernel_size=3, bias=True).to("cuda")
        print((ic(torch.Tensor(10,3,64,64).to("cuda"))).shape)
        ic.plot_kernel(channel_index=2, filter_index=10, save=True)
    case2:
        ic = IsotropicConv2D(3, kernel_size=3, bias=True).to("cuda")
        r = ic(torch.ones(10,3,64,64).to("cuda")).mean()
        r.backward()
        ic.kernel.grad
    """
    def __init__(self, in_channels, filters=1024, kernel_size=3, stride=1, padding=0, groups=1, 
                 bias=True, l2_rate=0.001, **kwargs):
        super().__init__(**kwargs)
        self.in_channels = in_channels
        self.filters = filters
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = autopad(kernel_size, padding)
        self.groups = groups
        self.bias = bias
        self.l2_rate = l2_rate
        
        if isinstance(self.stride, int):
            self.stride = (self.stride, self.stride)
        elif isinstance(self.stride, list):
            self.stride = tuple(self.stride)
            
        if isinstance(self.kernel_size, int):
            self.shape1 = self.shape2 = self.kernel_size
        elif isinstance(self.kernel_size, (list, tuple)):
            self.shape1, self.shape2 = self.kernel_size
        else:
            raise ValueError(
                f"Received: kernel_size={self.kernel_size}. The type is {type(self.kernel_size)}. "
                "Acceptable value types are only: (int, list, tuple) for IsotropicConv2D."
            )
        assert self.shape1 == self.shape2, f"Received: kernel_size={self.kernel_size}. Acceptable value is only: shape1 == shape2 for IsotropicConv2D."
        if self.groups != 1:
            raise ValueError(f"Received: groups={self.groups}. Acceptable value is only: 1 for IsotropicConv2D.")

        if self.shape1 != 1 and self.shape2 != 1:
            # 卷积核张量，形状为 [out_channels, in_channels, filter_height, filter_width]。
            self.weight = self.kernel = self._build_kernel(self.in_channels) 
        else:
            self.weight = self.kernel = nn.Parameter(torch.Tensor(self.filters, in_channels, 1, 1))
            nn.init.xavier_uniform_(self.weight)
        if self.bias:
            self.bias_param = nn.Parameter(torch.zeros(self.filters))

    def forward(self, inputs):
        return F.conv2d(inputs, self.kernel, 
                        bias = self.bias_param if self.bias else None, 
                        stride=self.stride, padding=self.padding)

    def plot_kernel(self, channel_index=0, filter_index=0, save=True) -> None:
        """
        use:
            ic = IsotropicConv2D(3, bias=True)
            ic.plot_kernel(channel_index=2, filter_index=10, save=True)
        """
        plt.imshow(self.kernel[filter_index, channel_index, :, :].detach().cpu().numpy(), cmap='viridis')
        ax = plt.gca()
        # 次要网格线
        ax.set_xticks(np.arange(-0.5, self.shape2, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, self.shape1, 1), minor=True)
        ax.grid(which='minor', color='white', linestyle='-', linewidth=1) # 设置网格线的样式和颜色
        # 主要网格线
        # ax.set_xticks(np.arange(-0.5, self.shape2, 3), minor=False)
        # ax.set_yticks(np.arange(-0.5, self.shape1, 1), minor=False)
        # plt.grid(which='major', color='white', linestyle='-', linewidth=1.5)
        # 隐藏坐标轴及其标签
        ax.tick_params(which='both', bottom=False, left=False, labelbottom=False, labelleft=False)
        if save:
            plt.savefig(f"isotropic_conv2d_kernel_{self.shape1}x{self.shape2}.png", dpi=240)
        plt.show()

    # @classmethod # def _build_kernel(cls, input_shape)
    def _build_kernel(self, in_channels):
        if self.shape1 == 3:
            kernel_c = nn.Parameter(torch.Tensor(self.filters, in_channels))
            kernel_d1 = nn.Parameter(torch.Tensor(self.filters, in_channels))
            kernel_d2 = nn.Parameter(torch.Tensor(self.filters, in_channels))
            nn.init.xavier_uniform_(kernel_c)
            nn.init.xavier_uniform_(kernel_d1)
            nn.init.xavier_uniform_(kernel_d2)

            # 构建各向同性卷积核 [[d2, d1, d2], [d1, c, d1], [d2, d1, d2]]
            # 卷积核张量，形状为 [out_channels, in_channels, filter_height, filter_width]。
            kernel = torch.zeros(self.filters, in_channels, 3, 3)
            kernel[:, :, 0, 0] = kernel_d2
            kernel[:, :, 0, 1] = kernel_d1
            kernel[:, :, 0, 2] = kernel_d2
            kernel[:, :, 1, 0] = kernel_d1
            kernel[:, :, 1, 1] = kernel_c
            kernel[:, :, 1, 2] = kernel_d1
            kernel[:, :, 2, 0] = kernel_d2
            kernel[:, :, 2, 1] = kernel_d1
            kernel[:, :, 2, 2] = kernel_d2
            
        else: # 通用方案
            center_point = [(self.shape1 - 1) / 2, (self.shape2 - 1) / 2]
            manhattan_distance2center = lambda i, j: abs(i - center_point[0]) + abs(j - center_point[1]) # 曼哈顿距离
            max_distance = manhattan_distance2center(self.shape1 - 1, self.shape2 - 1)

            if self.shape1 % 2: # 奇数存在一个中心格子，偶数无中心格子
                kernel_c = nn.Parameter(torch.Tensor(self.filters, in_channels))
                nn.init.xavier_uniform_(kernel_c)
            # 存在的格子的距离是：从1到max_distance
            kernel_d_all = [nn.Parameter(torch.Tensor(self.filters, in_channels)) for _ in range(int(max_distance))]
            for param in kernel_d_all:
                nn.init.xavier_uniform_(param)

            # 构建各向同性卷积核 
            # 卷积核张量，形状为 [out_channels, in_channels, filter_height, filter_width]。
            kernel = torch.zeros(self.filters, in_channels, self.shape1, self.shape2)
            if self.shape1 % 2:
                kernel[:, :, int(self.shape1 // 2 + 1), int(self.shape2 // 2 + 1)] = kernel_c
            for i in range(self.shape1):
                for j in range(self.shape2):
                    distance = manhattan_distance2center(i, j)
                    if distance != 0: # 排除奇数期刊下的中心点格子
                        kernel[:, :, i, j] = kernel_d_all[int(distance) - 1]

        kernel = nn.Parameter(kernel)
        
        return kernel


# 重要通用模块：通道共享权重Depthwise卷积
class SharedKernelConv2D(nn.Module):
    # Base module: 每个通道单独做卷积，且所有通道都用相同的卷积核做运算（共享卷积核）
    def __init__(self, filters=1, kernel_size=5, padding=2, bias=True, use_isotropic_conv=True):
        super().__init__()
        self.filters = filters
        self.kernel_size = (kernel_size, kernel_size) if isinstance(kernel_size, int) else kernel_size
        self.padding = autopad(kernel_size, padding)
        self.bias = bias
        self.use_isotropic_conv = use_isotropic_conv
        
        # 创建 共享卷积核
        if self.use_isotropic_conv:
            _ = IsotropicConv2D(1, self.filters, self.kernel_size, padding=self.padding)
            self.kernel = deepcopy(_._build_kernel(1))
            del _
        else:
            self.kernel = nn.Parameter(
                torch.Tensor(self.filters, 1, self.kernel_size[0], self.kernel_size[1]),
                requires_grad=True) # 相当于 tf 的trainable ，为 False 即 不跟新，该参视作常数，不对其求导但正常影响其他变量的求导
            nn.init.xavier_uniform_(self.kernel)

        if self.bias:
            self.bias_param = nn.Parameter(torch.Tensor(self.filters))
            nn.init.zeros_(self.bias_param)

    def forward(self, inputs):
        # 通过灵活改变shape的形式，实现conv在原通道维度上的广播运算
        
        c = inputs.shape[1]
        # Reshape inputs from (n, c, h, w) to (n * c, 1, h, w)
        inputs = inputs.view((inputs.shape[0] * c, 1, inputs.shape[2], inputs.shape[3]))

        # (n * c, 1, h, w) -> (n * c, filters, h, w)
        outputs = F.conv2d(inputs, self.kernel, 
                           bias = self.bias_param if self.bias else None, 
                           stride=(1, 1), padding=self.padding)
            
        # (n * c, filters, h, w) -> (n, c * filters, h, w)
        outputs = outputs.view((outputs.shape[0] // c, c * outputs.shape[1], outputs.shape[2], outputs.shape[3]))

        return outputs


# 重要通用模块：自适应多头交叉注意力机制（通用基础模块）
class AdaptiveMultiHeadCrossAttention(nn.Module):
    """
    多头 自适应 通道交叉注意力机制 和 空间交叉注意力机制 
    
    #################################################################
    case1:
        ca = AdaptiveMultiHeadCrossAttention((None,1,20), (None,1024,32,32), (None,1024,32,32)).to("cuda")
        z,s = ca(torch.ones((10,1,20)).to("cuda"), torch.ones((10,1024,32,32)).to("cuda"), torch.ones((10,1024,32,32)).to("cuda"))
        print(z.shape,s.shape)
        plt.plot(s[0,0,:,0,0].detach().cpu().numpy())
    case2:
        sa = AdaptiveMultiHeadCrossAttention((None,8,1,64,64), (None,1024,32,32), (None,1024,32,32)).to("cuda")
        z,s = sa(torch.ones((10,8,1,64,64)).to("cuda"), torch.ones((10,1024,32,32)).to("cuda"), torch.ones((10,1024,32,32)).to("cuda"))
        print(z.shape,s.shape)
        print(sa.q_linear_reshape.kernel.shape[-2], sa.q_linear_reshape.stride)
        plt.imshow(s[0,0,0,:,:].detach().cpu().numpy())
    sase3:
        sa = AdaptiveMultiHeadCrossAttention((None,8,2,128,128), (None,1024,32,32), (None,1024,32,32)).to("cuda")
        z,s = sa(torch.ones((10,8,2,128,128)).to("cuda"), torch.ones((10,1024,32,32)).to("cuda"), torch.ones((10,1024,32,32)).to("cuda"))
        print(z.shape,s.shape)
        print(sa.q_linear_reshape.kernel.shape[-2], sa.q_linear_reshape.stride)
        plt.imshow(s[0,0,0,:,:].detach().cpu().numpy())
    
    #################################################################
    z, attention_weights = multi_head_cross_attention([q_origin, k, v])
    q_origin是网络外部的输入或其简单处理后的特征(一维序列或图), 一般可以 k = v, 是卷积网络中的较深层的特征图conv_feature_maps(图)
    input:
        q_origin shape: [None, 1, series_len] or [None, series_len, channel, map_h, map_w] # channel推荐是1，虽然也可以不是1
        k, v shape: [None, channels, feature_map_h, feature_map_w]
    output:
        z shape: [None, channels, feature_map_h, feature_map_w]
        attention_weights shape: [num_head, None, channels, 1, 1] 
                                or [num_head, None, 1, feature_map_h, feature_map_w]
    
    主要流程可简化为:
        linear(concat(
            Attention(linear( nonlinear(q_origin) ), linear(k), linear(v))
            ))
        Attention: sim(q, k) -> softmax or sigmoid 归一化 -> weighting(v)
    
    """
    def __init__(self, q_origin_shape, k_shape, v_shape,
                 num_heads=8, linear_kernel_size=5, use_isotropic_conv=True, **kwargs):
        super().__init__(**kwargs)
        self.q_origin_shape, self.k_shape, self.v_shape = q_origin_shape, k_shape, v_shape
        assert len(self.q_origin_shape) in (3, 5), f"len(self.q_origin_shape)={len(self.q_origin_shape)} mast be 3 or 5"
        # dim_k = torch.Tensor((k_shape[-3],)) # for Attention
        
        self.num_heads = num_heads
        self.linear_kernel_size = linear_kernel_size
        self.use_isotropic_conv = use_isotropic_conv
        
        if self.use_isotropic_conv:
            self.Conv = IsotropicConv2D
        else:
            self.Conv = nn.Conv2d

        # channel attention
        if len(self.q_origin_shape) == 3: # (None, 1, series_len)
            self.q_linear_latent_size = self.q_origin_shape[-1] // 2 if self.q_origin_shape[-1] is not None and self.q_origin_shape[-1] > 16 else 8
            
            # (None, 1, series_len) @ (self.q_origin_shape[-1], self.q_linear_latent_size)
            # -> (None, 1, self.q_linear_latent_size) 
            self.q_linear = nn.Parameter(torch.Tensor(self.q_origin_shape[-1], self.q_linear_latent_size))
            nn.init.xavier_uniform_(self.q_linear)
            
            self.global_avgpool = nn.AdaptiveAvgPool2d((1, 1))
            
            # 扩展出多个 head 同时 进行相似度计算的第一阶段
            # (None, self.q_linear_latent_size + self.v_shape[-3]) @ (self.num_heads, self.q_linear_latent_size + self.v_shape[-3], self.v_shape[-3] // 2)
            # -> (self.num_heads, None, self.v_shape[-3] // 2)
            self.sim_linear1 = nn.Parameter(torch.Tensor(self.num_heads, self.q_linear_latent_size + self.v_shape[-3], self.v_shape[-3] // 2))
            nn.init.xavier_uniform_(self.sim_linear1)
            
            # 每个 head 里都做相同的相似度计算的第二阶段
            # (self.num_heads, None, self.v_shape[-3] // 2) @ (self.v_shape[-3] // 2, self.v_shape[-3]) 
            # -> (self.num_heads, None, self.v_shape[-3])
            self.sim_linear2 = nn.Parameter(torch.Tensor(self.v_shape[-3] // 2, self.v_shape[-3]))
            nn.init.xavier_uniform_(self.sim_linear2)
        
        # spatial attention
        elif len(self.q_origin_shape) == 5: # (None, series_len, c, map_h, map_w)
            # (None, channel, map_h, map_w) -> (None, channel // 2 + 1, map_h, map_w) 
            self.q_linear = self.Conv(self.q_origin_shape[-4] * self.q_origin_shape[-3], self.q_origin_shape[-3] // 2 + 1, kernel_size=5, padding='same', bias=False)
            
            _stride = int(np.round(self.q_origin_shape[-2] / self.k_shape[-2]))
            _kernel_size = self.q_origin_shape[-2] - (self.k_shape[-2]-1) * _stride
            if _kernel_size < _stride: # 如果 _kernel_size 还没有 _stride 大，那就会丢失信息，此时只能减小 _stride 以获得更大的 _kernel_size
                _stride -= 1
                _kernel_size = self.q_origin_shape[-2] - (self.k_shape[-2]-1) * _stride
            # (None, channel // 2 + 1, map_h, map_w)  -> (None, channel, k_h, k_w) 
            self.q_linear_reshape = self.Conv(self.q_origin_shape[-3] // 2 + 1, self.q_origin_shape[-3], _kernel_size, stride=_stride, padding=0, bias=False) 
            
            # 扩展出多个 head 同时 进行相似度计算的第一阶段
            # (None, channel_, map_h, map_w) -> (None, self.num_heads, map_h, map_w) 
            self.sim_linear1 = self.Conv(self.q_origin_shape[-3] + 2, self.num_heads, self.linear_kernel_size, padding='same', bias=False)
            # 每个 head 里都做相同的相似度计算的第二阶段
            # (None, self.num_heads, map_h, map_w) -> (None, self.num_heads, map_h, map_w) 
            self.sim_linear2 = SharedKernelConv2D(1, kernel_size=self.linear_kernel_size, padding='same', bias=False, 
                                                  use_isotropic_conv=self.use_isotropic_conv)
        
        # 扩展出多个 head 
        # (None, self.v_shape[-3], h, w) -> (None, self.v_shape[-3] * self.num_heads, h, w)
        self.v_linear = self.Conv(self.v_shape[-3], self.v_shape[-3] * self.num_heads, self.linear_kernel_size, padding='same', bias=False)
        # 合并 head
        self.z_linear = nn.Conv2d(self.v_shape[-3] * self.num_heads, self.v_shape[-3], kernel_size=1, groups=self.v_shape[-3], padding='same', bias=False)

    def forward(self, q_origin, k, v):
        # channel attention
        if len(self.q_origin_shape) == 3: # (None, 1, series_len)
            q = F.silu(
                torch.matmul(q_origin, self.q_linear)
                ).view((-1, self.q_linear_latent_size))
            
            k = self.global_avgpool(k).view((-1, self.k_shape[-3]))
            
            v = self.v_linear(v)
            
            # 计算q和k的相似度
            _ = torch.matmul(torch.cat([q, k], dim=-1), self.sim_linear1)
            attention_logits = torch.matmul(F.silu(_), self.sim_linear2)
            # 计算q和k的相似度分数
            attention_weights = torch.sigmoid(attention_logits).view(
                (self.num_heads, -1, self.v_shape[-3], 1, 1))
            # (self.num_heads, None, self.v_shape[-3], 1, 1) -> (None, self.v_shape[-3] * self.num_heads, 1, 1)
            attention_weights_ = attention_weights.permute((1, 2, 0, 3, 4)).reshape((-1, self.v_shape[-3] * self.num_heads, 1, 1))
            
        # spatial attention
        elif len(self.q_origin_shape) == 5: # (None, series_len, c, map_h, map_w)
            q_origin = q_origin.view(q_origin.shape[0], -1, q_origin.shape[-2], q_origin.shape[-1])
            q = F.silu(self.q_linear(q_origin))
            q = self.q_linear_reshape(q)
            
            k_mean   = torch.mean(k, dim=-3, keepdim=True)
            k_max, _ = torch.max(k, dim=-3, keepdim=True)
            k = torch.cat([k_mean, k_max], dim=-3)
            
            v = self.v_linear(v)
            
            # 计算q和k的相似度
            _ = self.sim_linear1(torch.cat([q, k], dim=-3))
            attention_logits = self.sim_linear2(F.silu(_))
            # 计算q和k的相似度分数
            attention_weights = torch.sigmoid(attention_logits)
            # (None, self.num_heads, map_h, map_w) -> (None, self.v_shape[-3] * self.num_heads, map_h, map_w)
            attention_weights_ = attention_weights.repeat(1, self.v_shape[-3], 1, 1) # <=> tf.tile()
            # (None, self.num_heads, map_h, map_w) -> (self.num_heads, None, 1, map_h, map_w) 
            # attention_weights = attention_weights.permute((1, 0, 2, 3)).reshape((self.num_heads, -1, 1, self.v_shape[-2], self.v_shape[-1]))
            attention_weights = attention_weights.permute((1, 0, 2, 3)).reshape((self.num_heads, -1, 1, attention_weights.shape[-2], attention_weights.shape[-1]))
            
        z = attention_weights_ * v
        z = self.z_linear(z)
        
        return z, attention_weights


#%% self-attention CBAM modules

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.f1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu = nn.ReLU()
        self.f2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.f2(self.relu(self.f1(self.avg_pool(x))))
        max_out = self.f2(self.relu(self.f1(self.max_pool(x))))
        out = self.sigmoid(avg_out + max_out)
        return out


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        assert kernel_size in (3, 7), "kernel size must be 3 or 7"
        padding = 3 if kernel_size == 7 else 1
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # 1*h*w
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        # 2*h*w
        x = self.conv(x)
        # 1*h*w
        return self.sigmoid(x)


class CBAM(nn.Module):
    def __init__(self, c1, c2, ratio=16, kernel_size=7):
        super().__init__()
        self.channel_attention = ChannelAttention(c1, ratio)
        self.spatial_attention = SpatialAttention(kernel_size)

    def forward(self, x):
        out = self.channel_attention(x) * x
        # c*h*w
        # c*h*w * 1*h*w
        out = self.spatial_attention(out) * out
        return out


#%% yolov5 base modules (see https://github.com/ultralytics/yolov5)

class CBS(nn.Module):
    # Standard convolution
    def __init__(self, c1, c2, k=1, s=1, p="same", g=1, act=True, use_isotropic_conv=True):  # ch_in, ch_out, kernel, stride, padding, groups
        super().__init__()
        self.use_isotropic_conv = use_isotropic_conv
        if self.use_isotropic_conv:
            self.Conv = IsotropicConv2D
        else:
            self.Conv = nn.Conv2d
            
        # if k == 1:
        #     self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False)
        # else:
        #     self.conv = self.Conv(c1, c2, k, s, autopad(k, p), groups=g, bias=False)
        if k == 1:
            self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=True)
        else:
            self.conv = self.Conv(c1, c2, k, s, autopad(k, p), groups=g, bias=True)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        return self.act(self.conv(x))


class Bottleneck(nn.Module):
    # Standard bottleneck
    def __init__(self, c1, c2, shortcut=True, g=1, e=0.5, use_isotropic_conv=True):  # ch_in, ch_out, shortcut, groups, expansion
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = CBS(c1, c_, 1, 1, use_isotropic_conv=use_isotropic_conv)
        self.cv2 = CBS(c_, c2, 3, 1, g=g, use_isotropic_conv=use_isotropic_conv)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class C3(nn.Module):
    # CSP Bottleneck with 3 CBSolutions
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5, use_isotropic_conv=True):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = CBS(c1, c_, 1, 1, use_isotropic_conv=use_isotropic_conv)
        self.cv2 = CBS(c1, c_, 1, 1, use_isotropic_conv=use_isotropic_conv)
        self.cv3 = CBS(2 * c_, c2, 1, use_isotropic_conv=use_isotropic_conv)  # optional act=FReLU(c2)
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, e=1.0, use_isotropic_conv=use_isotropic_conv) for _ in range(n)))

    def forward(self, x):
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), 1))
    

class SPPF(nn.Module):
    # Spatial Pyramid Pooling - Fast (SPPF) layer for YOLOv5 by Glenn Jocher
    def __init__(self, c1, c2, k=5, use_isotropic_conv=True):  # equivalent to SPP(k=(5, 9, 13))
        super().__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = CBS(c1, c_, 1, 1, use_isotropic_conv=use_isotropic_conv)
        self.cv2 = CBS(c_ * 4, c2, 1, 1, use_isotropic_conv=use_isotropic_conv)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

    def forward(self, x):
        x = self.cv1(x)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')  # suppress torch 1.9.0 max_pool2d() warning
            y1 = self.m(x)
            y2 = self.m(y1)
            return self.cv2(torch.cat((x, y1, y2, self.m(y2)), 1))


class Contract(nn.Module):
    # Contract width-height into channels, i.e. x(1,64,80,80) to x(1,256,40,40)
    def __init__(self, gain=2):
        super().__init__()
        self.gain = gain

    def forward(self, x):
        b, c, h, w = x.size()  # assert (h / s == 0) and (W / s == 0), 'Indivisible gain'
        s = self.gain
        x = x.view(b, c, h // s, s, w // s, s)  # x(1,64,40,2,40,2)
        x = x.permute(0, 3, 5, 1, 2, 4).contiguous()  # x(1,2,2,64,40,40)
        return x.view(b, c * s * s, h // s, w // s)  # x(1,256,40,40)
    

class Expand(nn.Module):
    # Expand channels into width-height, i.e. x(1,64,80,80) to x(1,16,160,160)
    def __init__(self, gain=2):
        super().__init__()
        self.gain = gain

    def forward(self, x):
        b, c, h, w = x.size()  # assert C / s ** 2 == 0, 'Indivisible gain'
        s = self.gain
        x = x.view(b, s, s, c // s ** 2, h, w)  # x(1,2,2,16,80,80)
        x = x.permute(0, 3, 4, 1, 5, 2).contiguous()  # x(1,16,80,2,80,2)
        return x.view(b, c // s ** 2, h * s, w * s)  # x(1,16,160,160)
    

class Concat(nn.Module):
    # Concatenate a list of tensors along dimension
    def __init__(self, dimension=1):
        super().__init__()
        self.d = dimension

    def forward(self, x):
        return torch.cat(x, self.d)


#%% Convolutional LSTM module in PyTorch (see https://github.com/ndrplz/ConvLSTM_pytorch)

class ConvLSTMCell(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size=3, bias=True, use_isotropic_conv=True):
        """
        Initialize ConvLSTM cell.

        Parameters
        ----------
        input_dim: int
            Number of channels of input tensor.
        hidden_dim: int
            Number of channels of hidden state.
        kernel_size: (int, int) etc.
            Size of the convolutional kernel.
        bias: bool
            Whether or not to add the bias.
            
        #######################################################################
        input shape: (b, c, h, w)
        
        case:
            x = torch.rand((32, 64, 128, 128)).to("cuda")
            cell = ConvLSTMCell(64,16,3,True, True).to("cuda")
            y, c = cell(x, cell.init_hidden(32, (128,128)))
            y.shape
        """
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.kernel_size = kernel_size
        self.padding = autopad(kernel_size, "same")
        self.bias = bias
        
        self.use_isotropic_conv = use_isotropic_conv
        if self.use_isotropic_conv:
            self.Conv = IsotropicConv2D
        else:
            self.Conv = nn.Conv2d
        
        self.conv = self.Conv(self.input_dim + self.hidden_dim,
                              4 * self.hidden_dim, 
                              self.kernel_size,
                              padding=self.padding,
                              bias=self.bias)

    def forward(self, input_tensor, cur_state):
        h_cur, c_cur = cur_state

        combined = torch.cat([input_tensor, h_cur], dim=1)  # concatenate along channel axis

        combined_conv = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)

        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)

        return h_next, c_next

    def init_hidden(self, batch_size, image_size):
        height, width = image_size
        return (torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device),
                torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device))


class CConvLSTM(nn.Module):
    # 在每层ConvLSTMCell前面加一层无bias和act的Conv，以实现多层ConvLSTM中map_shape的缩放
    """
    [Conv -> ConvLSTMCell] * num_layers
    shape (each layer):
        Conv 通常stride为3（所有层共用），padding为0，kernel_size为5或7，filters为in_c*4（首层除外，首层为64）
        ConvLSTMCell stride=1，padding="same"，in_c=上一层Conv的filters，out_c=上一层Conv的filters
        out shape: 
            spatial: (N-Conv_kernel_size) // Conv_stride + 1 
            channel: filters
    
    ###########################################################################
    
    Parameters:
        input_dim: Number of channels in input
        hidden_dim: Number of hidden channels
        kernel_size: Size of kernel in convolutions
        num_layers: Number of LSTM layers stacked on each other
        batch_first: Whether or not dimension 0 is the batch or not
        bias: Bias or no bias in Convolution
        return_all_layers: Return the list of computations for all layers
        Note: Will do same padding.

    Input:
        A tensor of size B, T, C, H, W or T, B, C, H, W
    Output:
        A tuple of two lists of length num_layers (or length 1 if return_all_layers is False).
            0 - layer_output_list is the list of lists of length T of each output
            1 - last_state_list is the list of last states
                    each element of the list is a tuple (h, c) for hidden state and memory
    Example1:
        x = torch.rand((32, 10, 64, 128, 128)).to("cuda")
        convlstms = CConvLSTM(64, 16, (3,3), 1, True, True, False, 1, use_isotropic_conv=False).to("cuda")
        _, h_c_state_list = convlstms(x)
        # y = _[-1][:,-1,...]
        y = h_c_state_list[-1][0]  # 0 for layer index, 0 for h index
        print(y.shape)
        out_size = convlstms.get_convlstms_out_size(128)
    
    Example2:
        x = torch.rand((32, 10, 3, 256, 256)).to("cuda")
        convlstms = CConvLSTM(
            input_dim=3,
            hidden_dim=[64, 256, 1024],
            kernel_size=[(7, 7), (5, 5), (5, 5)],
            num_layers=3, # lstm的层数，需要与len(hidden_dim)相等
            batch_first=True, # dimension 0位置是否是batch，是则True
            bias=True,
            return_all_layers=False, 
            stride=2, # 每层 ConvLSTM之前的Conv 共用的 stride
            use_isotropic_conv=True).to("cuda")
        _, h_c_state_list = convlstms(x)
        # y = _[-1][:,-1,...]
        y = h_c_state_list[-1][0]  # -1 for layer index, 0 for h index
        print(y.shape)
        out_size = convlstms.get_convlstms_out_size(256)
    
    Example3:
        x = torch.rand((1, 1, 2, 64, 64)).to("cuda")
        convlstms = CConvLSTM(
                   input_dim=2,
                   hidden_dim=[64, 256, 1024],
                   kernel_size=[(7, 7), (5, 5), (5, 5)],
                   num_layers=3, 
                   stride=2,
                   use_isotropic_conv=True).to("cuda")
        _, h_c_state_list = convlstms(x)
        # y = _[-1][:,-1,...]
        y = h_c_state_list[-1][0]  # 0 for layer index, 0 for h index
        print(y.shape)
        out_size = convlstms.get_convlstms_out_size(64)
    
    # recommend:
    for big map: hidden_dim=[64, 256, 1024], stride=3
    for small map: hidden_dim=[64, 128, 256], stride=2
    """
    def __init__(self, input_dim, hidden_dim=[64, 128, 256], 
                 kernel_size=[(7, 7), (5, 5), (5, 5)], num_layers=3,
                 batch_first=True, bias=True, return_all_layers=False, 
                 stride=2, use_isotropic_conv=True):
        super().__init__()

        self._check_kernel_size_consistency(kernel_size)

        # Make sure that both `kernel_size` and `hidden_dim` are lists having len == num_layers
        kernel_size = self._extend_for_multilayer(kernel_size, num_layers)
        hidden_dim = self._extend_for_multilayer(hidden_dim, num_layers)
        if not len(kernel_size) == len(hidden_dim) == num_layers:
            raise ValueError('Inconsistent list length.')

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.bias = bias
        self.return_all_layers = return_all_layers
        
        self.stride = stride
        self.use_isotropic_conv = use_isotropic_conv
        if self.use_isotropic_conv:
            self.Conv = IsotropicConv2D
        else:
            self.Conv = nn.Conv2d

        addition_conv_list = []
        cell_list = []
        for i in range(0, self.num_layers):
            cur_input_dim = self.input_dim if i == 0 else self.hidden_dim[i - 1]

            addition_conv_list.append(self.Conv(cur_input_dim,
                                                self.hidden_dim[i],
                                                self.kernel_size[i],
                                                self.stride,
                                                autopad(self.kernel_size[i], 0),
                                                bias=True))
            cell_list.append(ConvLSTMCell(input_dim=self.hidden_dim[i],
                                          hidden_dim=self.hidden_dim[i],
                                          kernel_size=self.kernel_size[i],
                                          bias=self.bias))

        self.addition_conv_list = nn.ModuleList(addition_conv_list)
        self.cell_list = nn.ModuleList(cell_list)

    def forward(self, input_tensor, hidden_state=None):
        """
        Parameters
        ----------
        input_tensor: 
            5-D Tensor either of shape (t, b, c, h, w) or (b, t, c, h, w)
        hidden_state: 
            None. todo implement stateful

        Returns
        -------
        last_state_list, layer_output
        """
        if not self.batch_first:
            # (t, b, c, h, w) -> (b, t, c, h, w)
            input_tensor = input_tensor.permute(1, 0, 2, 3, 4).contiguous() # 通常permute()之后需要此操作确保数据在内存中的连续性，当不确定原本是否连续的时候，这是一个保险的操作

        b, _, _, h, w = input_tensor.size()
        
        # Implement stateful ConvLSTM
        if hidden_state is not None:
            raise NotImplementedError("see code")
        else:
            # Since the init is done in forward. Can send image size here
            hidden_state = self._init_hidden(batch_size=b,
                                             image_size=(h, w))
            
        layer_output_list = []
        last_state_list = []

        seq_len = input_tensor.size(1)
        cur_layer_input = input_tensor

        for layer_idx in range(self.num_layers):
            
            h_state, c_state = hidden_state[layer_idx]
            output_inner = []
            for t in range(seq_len):
                _x_t = self.addition_conv_list[layer_idx](cur_layer_input[:, t, :, :, :])
                h_state, c_state = self.cell_list[layer_idx](input_tensor=_x_t,
                                                 cur_state=[h_state, c_state])
                output_inner.append(h_state)

            layer_output = torch.stack(output_inner, dim=1)
            cur_layer_input = layer_output

            layer_output_list.append(layer_output)
            last_state_list.append([h_state, c_state])

        if not self.return_all_layers:
            layer_output_list = layer_output_list[-1:]
            last_state_list = last_state_list[-1:]

        return layer_output_list, last_state_list

    def _init_hidden(self, batch_size, image_size):
        # 创建每一层的初始隐藏状态
        
        conv_h, conv_w = image_size
        shape_f = lambda l, k: int(np.floor((l - k) / self.stride + 1))
        init_states = []
        for i in range(self.num_layers):
            conv_h, conv_w = shape_f(conv_h, self.kernel_size[i][0]), shape_f(conv_w, self.kernel_size[i][1])
            init_states.append(self.cell_list[i].init_hidden(batch_size, (conv_h, conv_w)))
        return init_states

    @staticmethod
    def _check_kernel_size_consistency(kernel_size):
        if not (isinstance(kernel_size, tuple) or
                (isinstance(kernel_size, list) and all([isinstance(elem, tuple) for elem in kernel_size]))):
            raise ValueError('`kernel_size` must be tuple or list of tuples')

    @staticmethod
    def _extend_for_multilayer(param, num_layers):
        if not isinstance(param, list):
            param = [param] * num_layers
        return param
    
    def get_convlstms_out_size(self, l):
        m = l
        for i in range(self.num_layers):
             m = (m - self.kernel_size[i][0]) // self.stride + 1
        return m

