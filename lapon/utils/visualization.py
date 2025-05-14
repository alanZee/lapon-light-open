# LaPON, GPL-3.0 license
# Plotting utils


from typing import  Tuple
import os
import sys
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib import figure
# from matplotlib.image import imread
import numpy as np
# import seaborn as sn
from PIL import Image, ImageDraw, ImageFont
import _pickle as cPickle

if __name__ == "__main__":
    sys.path.append(os.path.abspath("../"))
# else:
#     from pathlib import Path
#     FILE = Path(__file__).resolve()
#     ROOT = FILE.parents[0]  # LaPON root directory
#     if str(ROOT) not in sys.path:
#         sys.path.append(str(ROOT))  # add ROOT to PATH
#     ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative
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


# Settings
RANK = int(os.getenv('RANK', -1))
FONTSIZE = 16 # 21 16 14 11
matplotlib.rc('font', **{'size': FONTSIZE})
SPINE_LINEWIDTH = 2 # 1
# matplotlib.use('Agg')  # for writing to files only (non-GUI backend)
config = {
    'font.family': 'Times New Roman',
    'font.sans-serif': 'Times New Roman',
    'mathtext.fontset': 'stix', # 设置数学公式字体为stix
    'axes.unicode_minus': False, # 解决负号无法显示的问题
}
plt.rcParams.update(config)
# plt.gcf().set_size_inches(7, 7*0.618) # NOTR 黄金比例 0.618
# plt.gcf().set_size_inches(27/(16**2+9**2)**0.5*16, 27/(16**2+9**2)**0.5*9) # 2560*1440 27英寸显示器
FIGSIZE = 7, 7*0.618
# 2560*1440 27英寸显示器dpi： 109 (其他推荐值: 240, 300)
DPI = 300

# cmap & colors
from cmcrameri import cm # Scientific colour maps (NATURE cmap 之一; 感知均匀的、有序的，对色觉缺陷和色盲友好，即使在黑白打印中也清晰可读)
CMAP = cm.managua # 选择一个colormap # berlin(like GOOGLE PNAS) roma managua # (中间白) roma vik broc; (两端白,中间暗) berlin lisbon managua; (其他经典) batlow batlowW lajolla devon nuuk hawaii lipari navia # batlowS, ...
# CMAP = cm.vik
CMAP2 = cm.vik
import seaborn
# CMAP2 = seaborn.cm.icefire # (GOOGLE PNAS)


class Colors:
    # Ultralytics color palette https://ultralytics.com/
    def __init__(self):
        # hex = matplotlib.colors.TABLEAU_COLORS.values()
        self.hexs = (
            'FF3838', 'FF9D97', 'FF701F', 'FFB21D', 'CFD231', '48F90A', '92CC17', '3DDB86', '1A9334', '00D4BB',
            '2C99A8', '00C2FF', '344593', '6473FF', '0018EC', '8438FF', '520085', 'CB38FF', 'FF95C8', 'FF37C7')
        self.palette = [self.hex2rgb(f'#{c}') for c in self.hexs]
        self.n = len(self.palette)

    def __call__(self, i, bgr=False):
        c = self.palette[int(i) % self.n]
        return (c[2], c[1], c[0]) if bgr else c

    @staticmethod
    def hex2rgb(h):  # rgb order (PIL)
        return tuple(int(h[1 + i:1 + i + 2], 16) for i in (0, 2, 4))

colors = Colors()  # create instance for 'from utils.visualization import colors'
# COLORS = list(matplotlib.colors.TABLEAU_COLORS.values())
COLORS = ['#' + i for i in colors.hexs] + ['#3C4A9E', '#55ABCF'] # 一堆渐变暖色 + 两个冷色


###############################################################################
###############################################################################
###############################################################################

# 热力图 or 等高线图 (单一样本, 单行图) (* plot_accuracy_heat_or_contour 简化版)
# for training, no auto save
def plot_heat(
        data: Tuple[np.array, ...] or np.array, # (len: idx, shape: h, w) or (shape: idx, h, w)
        nrows: int = 1,
        ncols: int = 3,
        top_title: Tuple[str, ...] = ('Input', 
                                      'Prediction',
                                      'Target'),
        left_title: str = 'Sample #1', 
        # fig_title: str = 'Vorticity',
        fig_label: str = '',
        title: str = 'Vorticity',
        contour: bool = True, # plot contour
        contourf: bool = False, # plot contourf when contour=True
        cmap: str or cm = None,
        use_cbar: bool = True,
        figsize: Tuple[float, ...] = None, # w, h # (7, 7 * 0.618)
        show_but_noreturn: bool = False
        ) -> (figure.Figure, np.array) or None: # fig, axes
    """
    case1:
        data = np.random.rand(3, 64, 64)
        plot_heat(data, show_but_noreturn=True)
    case2:
        data = np.random.rand(3, 64, 64)
        fig, axes = plot_heat(data)
    """
    cmap = (CMAP if contour else CMAP2) if cmap is None else cmap
    figsize = (7, 7 / ncols * nrows) if figsize is None else figsize
    fontsize = FONTSIZE
    
    data = np.array(data)
    # vmax = max(abs(data.min()), data.max())
    vmax = min(abs(data.min()), data.max()) * 0.64 # 0.9 0.64
    vmin = - vmax
    norm = Normalize(vmin=vmin, vmax=vmax) # unify the cmap & cbar of all subgraphs
    
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, 
                             # layout='tight'
                             layout='constrained'
                             )
    # plt.subplots_adjust(wspace=0.05, hspace=0.05) # 子图之间的间距(NOTE 与 layout 不兼容)
    for i, ax in enumerate(axes.ravel()):
        if not contour:
            im = ax.imshow(data[i],
                           interpolation='spline16', 
                           cmap=cmap, # CMAP 'viridis'，'jet', 'turbo', 'bwr', 'seismic', 'magma', 'plasma', 'RdYlGn';
                           # 'gray', 'binary', 'Greys', 'CMRmap', 'cool', 'hot', 'hot_r'
                           norm=norm,
                           ) 
        else:
            # 设置坐标(but 不显示, 仅用于绘图)
            # x = np.linspace(0, 2*np.pi, data.shape[-1])
            # y = np.linspace(0, 2*np.pi, data.shape[-2])
            x = np.arange(data.shape[-1])
            y = np.arange(data.shape[-2])
            X, Y = np.meshgrid(x, y)
            if not contourf:
                im = ax.contour(X, Y, 
                                data[i],
                                20, # 线数
                                linewidths=1.39,
                                cmap=cmap, # CMAP 'viridis'，'jet', 'turbo', 'bwr', 'seismic', 'magma', 'plasma', 'RdYlGn';
                                # 'gray', 'binary', 'Greys', 'CMRmap', 'cool', 'hot', 'hot_r'
                                ) 
            else:
                im = ax.contourf(X, Y, 
                                 data[i], 
                                 20, # 线数 
                                 cmap=cmap, # CMAP 'viridis'，'jet', 'turbo', 'bwr', 'seismic', 'magma', 'plasma'; 
                                 # 'gray', 'binary', 'Greys', 'CMRmap', 'cool', 'hot', 'hot_r' 
                                 ) 
        # ax.axis('off') # 隐藏坐标轴的刻度、标签和轴线
        ax.set_xticks([]) # 刻度
        ax.set_yticks([]) 
        for spine in ax.spines.values(): # ax.spines['bottom'], left, right, top
            spine.set_visible(True) # NOTE 与 ax.axis('off') 不兼容
            spine.set_color('black')
            spine.set_linewidth(0.5)
        
        if i == 0:
            ax.set_ylabel(left_title, 
                          fontsize=fontsize)
            # ax.text(- data.shape[-1] / 13, # 锚点坐标
            #         data.shape[-2] / 2 , 
            #         left_title[i], 
            #         va='center', ha='center', # 对齐方式
            #         fontsize=fontsize, rotation=90)
        ax.set_title(top_title[i], fontsize=fontsize)
        
    axes[0].text(
        - data[0].shape[-1] / 13, 
        - data[0].shape[-1] / 13, 
        fig_label, 
        va='center', ha='center', 
        fontsize=fontsize + 0.5,
        # transform=ax.transAxes, # 转而让文本的位置相对于轴的尺寸来确定，而不再是数据坐标
        )
    
    if use_cbar:
        cbar_ax = fig.add_axes([1.005, 0.075 - 0.015, 0.02, 0.85])  # 颜色条的位置和大小
        cbar = fig.colorbar(
            im, 
            ax=axes, # cbar 应用于所有子图
            cax=cbar_ax,
            orientation='vertical',
            extend='both', spacing='proportional', # 两头尖尖的形状延伸
            )
        cbar.set_label(title, rotation=270, labelpad=9.5, fontsize=fontsize + 0.8)
    
    if not show_but_noreturn:
        return fig, axes
    else:
        plt.show()
        plt.close()


# 热力图 or 等高线图 (单一样本, 单列图) (* plot_accuracy_heat_or_contour 简化版)
def plot_heat_or_contour_vertical(
        data: Tuple[np.array, ...] or np.array, # (len: idx, shape: h, w) or (shape: idx, h, w)
        # nrows: int = 3,
        # ncols: int = 4,
        sub_titles: Tuple[str, ...] = (
            'Original flow field', 
            'Add noise', 
            'Locally missing'
            ),
        # fig_title: str = 'Vorticity',
        fig_label: str = '(e)',
        title: str = 'Velocity component',
        contour: bool = False, # plot contour
        contourf: bool = False, # plot contourf when contour=True
        cmap: str or cm = None,
        use_cbar: bool = False,
        save_dir: Path = Path('runs/test/exp'),
        figname: str = "stability_noise_cutout_vis.png",
        figsize: Tuple[float, ...] = None, # w, h # (7, 7 * 0.618)
        show_but_nosave: bool = False
        ) -> None:
    """
    case1:
        data = np.random.rand(3, 64, 64)
        plot_heat_or_contour_vertical(data, show_but_nosave=True, contour=True)
        plot_heat_or_contour_vertical(data, show_but_nosave=True)
        
    case2:
        data = np.random.rand(3, 64, 64)
        plot_heat_or_contour_vertical(data, save_dir=Path('./'), contour=True)
        plot_heat_or_contour_vertical(data, save_dir=Path('./'))
    """
    cmap = (CMAP if contour else cm.roma) if cmap is None else cmap
    filename = save_dir / figname
    nrows = 3
    ncols = 1
    # figsize = (7, 7 / ncols * nrows) if figsize is None else figsize
    figsize = (7 / 2.2, 7) if figsize is None else figsize
    fontsize = FONTSIZE
    
    if not show_but_nosave:
        with open(filename.with_suffix('.pkl'), 'wb') as fi: 
            cPickle.dump(data, fi) # save data
    
    data = np.array(data)
    # vmax = max(abs(data.min()), data.max())
    vmax = min(abs(data.min()), data.max()) * 0.64 # 0.9 0.64
    vmin = - vmax
    norm = Normalize(vmin=vmin, vmax=vmax) # unify the cmap & cbar of all subgraphs
    
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, 
                             # layout='tight'
                             layout='constrained'
                             )
    # plt.subplots_adjust(wspace=0.05, hspace=0.05) # 子图之间的间距(NOTE 与 layout 不兼容)
    for i, ax in enumerate(axes.ravel()):
        if not contour:
            im = ax.imshow(data[i],
                           interpolation='spline16', 
                           cmap=cmap, # CMAP 'viridis'，'jet', 'turbo', 'bwr', 'seismic', 'magma', 'plasma', 'RdYlGn';
                           # 'gray', 'binary', 'Greys', 'CMRmap', 'cool', 'hot', 'hot_r'
                           norm=norm,
                           ) 
        else:
            # 设置坐标(but 不显示, 仅用于绘图)
            # x = np.linspace(0, 2*np.pi, data.shape[-1])
            # y = np.linspace(0, 2*np.pi, data.shape[-2])
            x = np.arange(data.shape[-1])
            y = np.arange(data.shape[-2])
            X, Y = np.meshgrid(x, y)
            if not contourf:
                im = ax.contour(X, Y, 
                                data[i],
                                20, # 线数
                                linewidths=1.39,
                                cmap=cmap, # CMAP 'viridis'，'jet', 'turbo', 'bwr', 'seismic', 'magma', 'plasma', 'RdYlGn';
                                # 'gray', 'binary', 'Greys', 'CMRmap', 'cool', 'hot', 'hot_r'
                                ) 
            else:
                im = ax.contourf(X, Y, 
                                 data[i], 
                                 20, # 线数 
                                 cmap=cmap, # CMAP 'viridis'，'jet', 'turbo', 'bwr', 'seismic', 'magma', 'plasma'; 
                                 # 'gray', 'binary', 'Greys', 'CMRmap', 'cool', 'hot', 'hot_r' 
                                 ) 
        # ax.axis('off') # 隐藏坐标轴的刻度、标签和轴线
        ax.set_xticks([]) # 刻度
        ax.set_yticks([]) 
        for spine in ax.spines.values(): # ax.spines['bottom'], left, right, top
            spine.set_visible(True) # NOTE 与 ax.axis('off') 不兼容
            spine.set_color('black')
            spine.set_linewidth(0.5)
            
        ax.set_title(sub_titles[i], 
                      fontsize=fontsize + 1)
        # ax.text(- data.shape[-1] / 13, # 锚点坐标
        #         data.shape[-2] / 2 , 
        #         left_title[i], 
        #         va='center', ha='center', # 对齐方式
        #         fontsize=fontsize + 1, rotation=90)
        
    if not contour:
        axes[0].text(
            - data.shape[-1] / 13, 
            - data.shape[-1] / 13, 
            fig_label, 
            va='center', ha='center', 
            fontsize=fontsize + 0.5 + 1,
            # transform=ax.transAxes, # 转而让文本的位置相对于轴的尺寸来确定，而不再是数据坐标
            )
    else:
        axes[0].text(
            - data.shape[-1] / 13, 
            data.shape[-2] + data.shape[-1] / 13, 
            fig_label, 
            va='center', ha='center', 
            fontsize=fontsize + 0.5 + 1,
            # transform=ax.transAxes, # 转而让文本的位置相对于轴的尺寸来确定，而不再是数据坐标
            )
    
    if use_cbar:
        cbar_ax = fig.add_axes([1.005, 0.075 - 0.015, 0.02, 0.85])  # 颜色条的位置和大小
        cbar = fig.colorbar(
            im, 
            ax=axes, # cbar 应用于所有子图
            cax=cbar_ax,
            orientation='vertical',
            extend='both', spacing='proportional', # 两头尖尖的形状延伸
            )
        cbar.set_label(title, rotation=270, labelpad=9.5, fontsize=fontsize + 0.8 + 1)
    
    if not show_but_nosave:
        fig.savefig(filename, dpi=DPI, 
                    bbox_inches='tight', # 裁剪图像周围的空白区域 (NOTE 会使图像小于声明的figsize)
                    # bbox_inches=matplotlib.transforms.Bbox(
                    #     [[figsize[0]*-0.02, figsize[1]*-0.03], [figsize[0]*1.11, figsize[1]*1.02]]
                    #     ),
                    ) 
    else:
        plt.show()
    plt.close()
    
    
# Simple barh 单张横向条形图 (* plot_comparisonML_comprehensiveness 简化版)
def plot_simple_barh(
        data: Tuple or np.array, # shape: model
        xlabel: str = 'Vorticity correlation',
        cluster_label: Tuple[str, ...] = (
            'DS 512$\\times$512',
            'DS 256$\\times$256',
            'DS 128$\\times$128',
            'DS 64$\\times$64',
            'LaPON 64$\\times$64'),
        xlim_origin: Tuple[float, ...] = [0., 1.], # None
        xvline: float = 0.95, # None
        yhline: float = None, 
        xscale_log: bool = False, # 对数轴 (与xlim等冲突, 不可同时使用)
        fig_label: str = '(b)', # ''
        # legend: bool = True,
        save_dir: Path = Path('runs/test/exp'),
        figname: str = "acc_vorticity_correlation_barh.png",
        figsize: Tuple[float, ...] = None, # w, h # (7, 7 * 0.618) # (7, 7 / 0.8)
        show_but_nosave: bool = False
        ) -> None:
    """
    case1:
        # data = np.random.rand(5)
        data = np.arange(5)
        data = data / data.max()
        plot_simple_barh(data, show_but_nosave=True)
        
    case2:
        data = np.random.rand(5)
        plot_simple_barh(data, save_dir=Path('./'))
    """
    filename = save_dir / figname
    figsize = (7, 7 * 0.618) if figsize is None else figsize
    fontsize = FONTSIZE + 3.3 
    colors = COLORS[:len(cluster_label)-1] + [COLORS[-1]]
    
    if not show_but_nosave:
        with open(filename.with_suffix('.pkl'), 'wb') as fi: 
            cPickle.dump(data, fi) # save data
    
    fig, ax = plt.subplots(1, 1, figsize=figsize, 
                           # layout='tight'
                           # layout='constrained'
                           )
    # plt.subplots_adjust(wspace=0.05, hspace=0.05) # 子图之间的间距(NOTE 与 layout 不兼容)
    
    # data = np.array(data)
    indices = np.arange(len(cluster_label))[::-1] # 从上往下画
    
    ax.barh(
        indices, 
        data, 
        # xerr, 
        # left, # 条形图的起始x值（堆叠条形图时使用）
        # height=0.357,
        linewidth=2, # bar边缘线宽(貌似与edgecolor参数控制的不是同一个边缘线)
        color=colors, # bar颜色 # cm.batlowS, COLORS[i], COLORS[-1]
        alpha=0.8,
        edgecolor=colors, # bar边缘颜色
        # align, # 条形图的对齐方式，可以是'center'、'edge'或'zero'
        # label=bar_label[0],
        )
    
    if not xscale_log and (xlim_origin is not None):
        x_spare = 0.03 * (xlim_origin[1] - xlim_origin[0])
        xlim = (xlim_origin[0] - x_spare, xlim_origin[1] + x_spare)
        ax.set_xlim(xlim)
    
        if xlim_origin == [0., 1.]:
            # ax.set_xticks(np.linspace(0., 1., 6)) 
            ax.set_xticks(np.linspace(0., 1., 6), 
                          # ['', 0.2, 0.4, 0.6, 0.8, ''], 
                          [0.0, 0.2, 0.4, 0.6, 0.8, 1.0], 
                          fontsize = fontsize) 
    else:
        ax.tick_params(axis='x', labelsize=fontsize)
        
    ax.set_xlabel(xlabel, fontsize = fontsize)
    
    # ax.set_yticks(indices)
    # ax.set_yticklabels(cluster_label)
    ax.set_yticks(indices, cluster_label, 
                  fontsize = fontsize - 0.5)
    
    for spine in ax.spines.values(): # ax.spines['bottom'], left, right, top
        spine.set_visible(True) # NOTE 与 ax.axis('off') 不兼容
        spine.set_color('black')
        spine.set_linewidth(SPINE_LINEWIDTH)
        
    ax.grid(axis='y', # x, y, both
            color='gray', linestyle='--', linewidth=0.5, alpha=0.7)
    ax.axvline(x=0, color='gray', linestyle='-', linewidth=SPINE_LINEWIDTH - 0.5, alpha=1)
    
    if xvline is not None:
        ax.axvline(x=xvline, color='gray', linestyle='-', linewidth=0.8, alpha=1)
    if yhline is not None:
        ax.axhline(y=yhline, color='gray', linestyle='-', linewidth=0.8, alpha=1)
    
    # 线性刻度 -> 对数刻度(更清晰展示跨越多个数量级的数据的变化趋势) (NOTE 与ax.set_xticks()和ax.set_xlim()不兼容)
    if xscale_log:
        # ax.set_xscale('log')
        ax.set_xscale('log')
    
    # if legend:
    #     # fig.legend()
    #     # 合并图例
    #     handles, labels = ax.get_legend_handles_labels()
    #     handles2, labels2 = ax2.get_legend_handles_labels()
    #     axes[-1, -1].legend(
    #         handles + handles2, labels + labels2,
    #         # loc='lower left',
    #         # fontsize=fontsize,
    #         # frameon=False, # 边框显示
    #         framealpha=0.6, # 背景透明度 
    #         # ncol=1, 
    #         )
    
    lim = ax.axis() # xlim, ylim
    anchor_y_spare = 1 / 18
    ax.text(
        # lim[0] - anchor_y_spare * (lim[1] - lim[0]), 
        # lim[3] + anchor_y_spare * (lim[3] - lim[2]), 
        - anchor_y_spare / figsize[0] * figsize[1] - 0.18, 
        1.0 + anchor_y_spare, 
        fig_label, 
        va='center', ha='center', 
        fontsize=fontsize + 0.5,
        transform=ax.transAxes, # 转而让文本的位置相对于轴的尺寸来确定，而不再是数据坐标
        )
    
    if not show_but_nosave:
        fig.savefig(filename, dpi=DPI, 
                    bbox_inches='tight', # 裁剪图像周围的空白区域 (NOTE 会使图像小于声明的figsize)
                    ) 
    else:
        plt.show()
    plt.close()
    
    
###############################################################################
###############################################################################
###############################################################################

# accuracy (accuracy, stability 等共用) 可视化: 热力图 or 等高线图 
def plot_accuracy_heat_or_contour(
        data: Tuple[np.array, ...] or np.array, # (len: idx, shape: h, w) or (shape: idx, h, w)
        # nrows: int = 3,
        # ncols: int = 4,
        top_title: Tuple[str, ...] = (
            "$\\Delta t$ = 2$\\times10^{-3}$ s",  
            "$\\Delta t$ = 4$\\times10^{-3}$ s",
            "$\\Delta t$ = 8$\\times10^{-3}$ s",
            "$\\Delta t$ = 1.6$\\times10^{-2}$ s"
            # "No. of time step = 0",
            # "No. of time step = 10",
            # "No. of time step = 21",
            # "No. of time step = 100"
            ),
        left_title: Tuple[str, ...] = (
            "Target",
            # "Target\n(DS 1024 $\\times$ 1024, \n$\\Delta t$ = 2$\\times10^{-4}$ s)", # 'DS 1024 × 1024\n($\Delta t$ = 2×$10^{-4}$ s)'
            "LaPON 64 $\\times$ 64", 
            "DS 64 $\\times$ 64"),
        # fig_title: str = 'Vorticity',
        fig_label: str = '(a)',
        title: str = 'Vorticity',
        contour: bool = True, # plot contour
        contourf: bool = False, # plot contourf when contour=True
        cmap: str or cm = None,
        use_cbar: bool = True,
        save_dir: Path = Path('runs/test/exp'),
        figname: str = "accuracy_heat.png",
        figsize: Tuple[float, ...] = None, # w, h # (7, 7 * 0.618)
        show_but_nosave: bool = False
        ) -> None:
    """
    case1:
        data = np.random.rand(12, 64, 64)
        plot_accuracy_heat_or_contour(data, show_but_nosave=True, contour=False)
        
    case2:
        data = np.random.rand(12, 64, 64)
        plot_accuracy_heat_or_contour(data, save_dir=Path('./'), contour=False)
    
    case3:
        x = np.linspace(-3.0, 3.0, 64)
        y = np.linspace(-3.0, 3.0, 64)
        X, Y = np.meshgrid(x, y)
        Z = np.exp(-(X**2 + Y**2)) - 0.5
        data = np.repeat(Z[np.newaxis, ...], 12, axis=0)
        
        plot_accuracy_heat_or_contour(data, show_but_nosave=True)
        plot_accuracy_heat_or_contour(data, show_but_nosave=True, contourf=True)
        
    case4:
        x = np.linspace(-3.0, 3.0, 64)
        y = np.linspace(-3.0, 3.0, 64)
        X, Y = np.meshgrid(x, y)
        Z = np.exp(-(X**2 + Y**2)) - 0.5
        data = np.repeat(Z[np.newaxis, ...], 12, axis=0)
        
        plot_accuracy_heat_or_contour(data, save_dir=Path('./'))
        plot_accuracy_heat_or_contour(data, save_dir=Path('./'), contourf=True)
    """
    cmap = (CMAP if contour else CMAP2) if cmap is None else cmap
    figname = figname.replace('heat', 'contour') if contour else figname
    figname = figname.replace('contour', 'contourf') if contour and contourf else figname
    filename = save_dir / figname
    nrows = len(left_title)
    ncols = len(top_title)
    figsize = (7, 7 / ncols * nrows) if figsize is None else figsize
    fontsize = FONTSIZE
    
    if not show_but_nosave:
        with open(filename.with_suffix('.pkl'), 'wb') as fi: 
            cPickle.dump(data, fi) # save data
    
    data = np.array(data)
    # vmax = max(abs(data.min()), data.max())
    vmax = min(abs(data.min()), data.max()) * 0.64 # 0.9 0.64
    vmin = - vmax
    norm = Normalize(vmin=vmin, vmax=vmax) # unify the cmap & cbar of all subgraphs
    
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, 
                             # layout='tight'
                             layout='constrained'
                             )
    # plt.subplots_adjust(wspace=0.05, hspace=0.05) # 子图之间的间距(NOTE 与 layout 不兼容)
    for i, ax in enumerate(axes.ravel()):
        if not contour:
            im = ax.imshow(data[i],
                           interpolation='spline16', 
                           cmap=cmap, # CMAP 'viridis'，'jet', 'turbo', 'bwr', 'seismic', 'magma', 'plasma', 'RdYlGn';
                           # 'gray', 'binary', 'Greys', 'CMRmap', 'cool', 'hot', 'hot_r'
                           norm=norm,
                           ) 
        else:
            # 设置坐标(but 不显示, 仅用于绘图)
            # x = np.linspace(0, 2*np.pi, data.shape[-1])
            # y = np.linspace(0, 2*np.pi, data.shape[-2])
            x = np.arange(data.shape[-1])
            y = np.arange(data.shape[-2])
            X, Y = np.meshgrid(x, y)
            if not contourf:
                im = ax.contour(X, Y, 
                                data[i],
                                20, # 线数
                                linewidths=1.39,
                                cmap=cmap, # CMAP 'viridis'，'jet', 'turbo', 'bwr', 'seismic', 'magma', 'plasma', 'RdYlGn';
                                # 'gray', 'binary', 'Greys', 'CMRmap', 'cool', 'hot', 'hot_r'
                                ) 
            else:
                im = ax.contourf(X, Y, 
                                 data[i], 
                                 20, # 线数 
                                 cmap=cmap, # CMAP 'viridis'，'jet', 'turbo', 'bwr', 'seismic', 'magma', 'plasma'; 
                                 # 'gray', 'binary', 'Greys', 'CMRmap', 'cool', 'hot', 'hot_r' 
                                 ) 
        # ax.axis('off') # 隐藏坐标轴的刻度、标签和轴线
        ax.set_xticks([]) # 刻度
        ax.set_yticks([]) 
        for spine in ax.spines.values(): # ax.spines['bottom'], left, right, top
            spine.set_visible(True) # NOTE 与 ax.axis('off') 不兼容
            spine.set_color('black')
            spine.set_linewidth(0.5)
        
    for i in range(nrows):
        for j in range(ncols):
            ax = axes[i, j]
            if i == 0:
                ax.set_title(top_title[j], fontsize=fontsize)
            if j == 0:
                ax.set_ylabel(left_title[i], 
                              fontsize=fontsize)
                # ax.text(- data.shape[-1] / 13, # 锚点坐标
                #         data.shape[-2] / 2 , 
                #         left_title[i], 
                #         va='center', ha='center', # 对齐方式
                #         fontsize=fontsize, rotation=90)
    
    if not contour:
        axes[0, 0].text(- data.shape[-1] / 13, 
                        - data.shape[-1] / 13, 
                        fig_label, 
                        va='center', ha='center', 
                        fontsize=fontsize + 0.5,
                        # transform=ax.transAxes, # 转而让文本的位置相对于轴的尺寸来确定，而不再是数据坐标
                        )
    else:
        axes[0, 0].text(- data.shape[-1] / 13, 
                        data.shape[-2] + data.shape[-1] / 13, 
                        fig_label, 
                        va='center', ha='center', 
                        fontsize=fontsize + 0.5,
                        # transform=ax.transAxes, # 转而让文本的位置相对于轴的尺寸来确定，而不再是数据坐标
                        )
    
    if use_cbar:
        # fig.subplots_adjust(
        #     # bottom=0.05, 
        #     # top=0.95,
        #     # left=0.067, 
        #     right=0.9, # 调整子图之间的间距，为colorbar预留空间
        #     wspace=0.05/4, 
        #     hspace=0.05/figsize[0]*figsize[1],
        #     )
        # cbar_ax = fig.add_axes([0.91, 0.5 - 0.8/2, 0.02, 0.8])  # 颜色条的位置和大小
        cbar_ax = fig.add_axes([1.005, 0.075 - 0.015, 0.02, 0.85])  # 颜色条的位置和大小
        cbar = fig.colorbar(
            im, 
            ax=axes, # cbar 应用于所有子图
            cax=cbar_ax,
            orientation='vertical',
            extend='both', spacing='proportional', # 两头尖尖的形状延伸
            )
        cbar.set_label(title, rotation=270, labelpad=9.5, fontsize=fontsize + 0.8)
    
    if not show_but_nosave:
        fig.savefig(filename, dpi=DPI, 
                    # bbox_inches='tight', # 裁剪图像周围的空白区域 (NOTE 会使图像小于声明的figsize)
                    bbox_inches=matplotlib.transforms.Bbox(
                        [[figsize[0]*-0.02, figsize[1]*-0.03], [figsize[0]*1.11, figsize[1]*1.02]]
                        ),
                    ) 
    else:
        plt.show()
    plt.close()


# accuracy 可视化: 相关系数图(单张折线图)
def plot_accuracy_correlation(
        x: list or np.array, # The size of time step interval Δt(Every data is consistent); shape: len
        data: np.array, # shape: idx, len
        line_label: Tuple[str, ...] = (
            'DS 512 × 512',
            'DS 256 × 256',
            'DS 128 × 128',
            'DS 64 × 64', 
            'LaPON 64 × 64'),
        line_colors_part: str = None,
        ylim_origin: Tuple[float, ...] = [0., 1.], # None
        xvline: float = (2e-3) * 3, # None # (Kolmogorov time of JHTDB 0.0424s, Large eddy turnover time of JHTDB 1.99s, Lyapunov time (流体混沌震荡: 2s from wiki))
        yhline: float = 0.95, # correlation 1: 完全相关; 0.95: 很强的相关性; 0.8: 强相关; 0.6: 中等相关; 0: 无关联
        xscale_log: bool = False, # 对数轴 (与xlim等冲突, 不可同时使用)
        yscale_log: bool = False, 
        xlabel: str = 'The size of $\Delta t$ (×$10^{-3}$ s)', # NOTE italic: $t$, $\mathit{\Delta}$; Δ:  $\Delta$ (LaTeX)
        ylabel: str = 'Vorticity correlation', 
        fig_label: str = '(b)',
        legend: bool = True,
        save_dir: Path = Path('runs/test/exp'),
        figname: str = "accuracy_correlation.png",
        figsize: Tuple[float, ...] = None, # w, h # (7, 7 * 0.618) # (7, 7 / 0.8)
        show_but_nosave: bool = False
        ) -> None:
    """
    case1:
        x = np.linspace(0, np.pi, 4 * 10)
        y = 0.5 * np.cos(x) + 0.5
        data = np.repeat(y[np.newaxis, ...], 5, axis=0) + np.random.rand(5,40)/10
        plot_accuracy_correlation(x, data, show_but_nosave=True)
        plot_accuracy_correlation(x, data, show_but_nosave=True, xscale_log=True, yscale_log=True)
        
    case2:
        x = np.linspace(0, np.pi, 4 * 10)
        y = 0.5 * np.cos(x) + 0.5
        data = np.repeat(y[np.newaxis, ...], 5, axis=0) + np.random.rand(5,40)/10
        plot_accuracy_correlation(x, data, save_dir=Path('./'))
    """
    filename = save_dir / figname
    figsize = (7, 7 * 0.618) if figsize is None else figsize
    aspect_ratio_r = figsize # w, h 
    fontsize = FONTSIZE
    line_colors_part = COLORS[:(data.shape[0]-1)] if line_colors_part is None else line_colors_part
    
    if not show_but_nosave:
        with open(filename.with_suffix('.pkl'), 'wb') as fi: 
            cPickle.dump([x, data], fi) # save data
    
    fig, ax = plt.subplots(1, 1, figsize=aspect_ratio_r, 
                           # layout='tight'
                           # layout='constrained'
                           )
    # plt.subplots_adjust(wspace=0.05, hspace=0.05) # 子图之间的间距(NOTE 与 layout 不兼容)
    
    for i, y in enumerate(data):
        if i == data.shape[0]-1:
            im = ax.plot(
                x,
                y,
                linestyle='--',
                color=COLORS[-1], # cm.batlowS, COLORS[i], COLORS[-1]
                # alpha=0.6,
                linewidth=1.8,
                # marker, markersize,
                label=line_label[i],
                ) 
        else:
            im = ax.plot(
                x,
                y,
                # linestyle='--',
                color=line_colors_part[i], # cm.batlowS, COLORS[i], COLORS[-1]
                # alpha=0.6,
                linewidth=1.8,
                # marker, markersize,
                label=line_label[i],
                ) 
            # ax2 = ax.twinx() # 共享y轴但有不同x轴的图
    
    ax.set_xlabel(xlabel, fontsize=fontsize) 
    ax.set_ylabel(ylabel, fontsize=fontsize)
    
    if not xscale_log and not yscale_log:
        if ylim_origin is not None:
            y_spare = 0.03 * (ylim_origin[1] - ylim_origin[0])
            ylim = (ylim_origin[0] - y_spare, ylim_origin[1] + y_spare)
            # ax.set_ylim(ylim)
            
            x_spare = (y_spare / aspect_ratio_r[0] * aspect_ratio_r[1]) * (max(x) - min(x))
            xlim = (min(x) - x_spare, max(x) + x_spare)
            # ax.set_xlim(xlim)
            
            # lim = ax.axis() # NOTE xlim, ylim
            ax.axis([*xlim, *ylim]) # xlim, ylim (不与 ax.set_xscale('log') 等冲突)
            
            if list(ylim_origin) == [0., 1.]:
                # ax.set_xticks([]) 
                ax.set_yticks(np.linspace(0., 1., 6),
                              [0.0, 0.2, 0.4, 0.6, 0.8, 1.0], 
                              fontsize = fontsize) 
    else:
        ax.tick_params(axis='y', labelsize=fontsize)
    ax.tick_params(axis='x', labelsize=fontsize)
    
    for spine in ax.spines.values(): # ax.spines['bottom'], left, right, top
        spine.set_visible(True) # NOTE 与 ax.axis('off') 不兼容
        spine.set_color('black')
        spine.set_linewidth(SPINE_LINEWIDTH)
    
    if legend:
        ax.legend(
            loc='lower left',
            # fontsize=fontsize,
            frameon=False, # 边框显示
            framealpha=0, # 背景透明度 
            # ncol=1, 
            )
    
    anchor_y_spare = 1 / 18
    ax.text(
        - anchor_y_spare / aspect_ratio_r[0] * aspect_ratio_r[1], 
        1.0 + anchor_y_spare, 
        fig_label, 
        va='center', ha='center', 
        fontsize=fontsize + 0.5,
        transform=ax.transAxes, # 转而让文本的位置相对于轴的尺寸来确定，而不再是数据坐标
        )
    
    # ax.grid(axis='both', # x, y, both
    #         color='gray', linestyle='--', linewidth=0.5, alpha=0.7)
    
    if xvline is not None:
        ax.axvline(x=xvline, color='gray', linestyle='-', linewidth=0.8, alpha=1)
    if yhline is not None:
        ax.axhline(y=yhline, color='gray', linestyle='-', linewidth=0.8, alpha=1)
    
    # 线性刻度 -> 对数刻度(更清晰展示跨越多个数量级的数据的变化趋势) (NOTE 与ax.set_xticks()和ax.set_xlim()不兼容)
    if xscale_log:
        ax.set_xscale('log') 
    if yscale_log:
        ax.set_yscale('log')
    
    if not show_but_nosave:
        fig.savefig(filename, dpi=DPI, 
                    bbox_inches='tight', # 裁剪图像周围的空白区域 (NOTE 会使图像小于声明的figsize)
                    ) 
    else:
        plt.show()
    plt.close()


# accuracy 可视化: 能量谱图(单张折线图)(*基本同plot_accuracy_correlation)
def plot_accuracy_spectrum(
        x: Tuple[np.array or Tuple, ...], # Wavenumber; len: model; shape: k_len
        data: Tuple[np.array or Tuple, ...], # len: model; shape: k_len
        line_label: Tuple[str, ...] = (
            'DS 512 × 512',
            'DS 256 × 256',
            'DS 128 × 128',
            'DS 64 × 64', 
            'LaPON 64 × 64',
            'Target'),
        xlim: Tuple[float, ...] = None, # None [8, 10000]
        ylim: Tuple[float, ...] = None, # None [1e-7, 1e-2]
        xvline: float = None, # None
        yhline: float = None, 
        xscale_log: bool = True, # 对数轴 (与xlim等冲突, 不可同时使用)
        yscale_log: bool = True, 
        xlabel: str = 'Wavenumber $\kappa$ (1/m)', 
        ylabel: str = 'Energy spectrum $E(\kappa)$ $(m^3/s^2)$', 
        fig_label: str = '(c)',
        legend: bool = True,
        save_dir: Path = Path('runs/test/exp'),
        figname: str = "accuracy_spectrum.png",
        figsize: Tuple[float, ...] = None, # w, h # (7, 7 * 0.618) # (7, 7 / 0.8)
        show_but_nosave: bool = False
        ) -> None:
    """
    case1:
        x = np.linspace(0, np.pi, 4 * 10)
        y = 0.5 * np.cos(x) + 0.5
        data = np.repeat(y[np.newaxis, ...], 6, axis=0) + np.random.rand(6,40)/10
        x = [x]*6
        data = [i for i in data]
        plot_accuracy_spectrum(x, data, show_but_nosave=True)
        
    case2:
        x = np.linspace(0, np.pi, 4 * 10)
        y = 0.5 * np.cos(x) + 0.5
        data = np.repeat(y[np.newaxis, ...], 6, axis=0) + np.random.rand(6,40)/10
        x = [x]*6
        data = [i for i in data]
        plot_accuracy_spectrum(x, data, save_dir=Path('./'))
    """
    filename = save_dir / figname
    figsize = (7, 7 * 0.618) if figsize is None else figsize
    aspect_ratio_r = figsize # w, h 
    fontsize = FONTSIZE
    
    if not show_but_nosave:
        with open(filename.with_suffix('.pkl'), 'wb') as fi: 
            cPickle.dump([x, data, xvline], fi) # save data
    
    fig, ax = plt.subplots(1, 1, figsize=aspect_ratio_r, 
                           # layout='tight'
                           # layout='constrained'
                           )
    # plt.subplots_adjust(wspace=0.05, hspace=0.05) # 子图之间的间距(NOTE 与 layout 不兼容)
    
    for i, y in enumerate(data):
        if i == len(data)-1:
            im = ax.plot(
                x[i],
                y,
                # linestyle='--',
                color=COLORS[-2], # cm.batlowS, COLORS[i], COLORS[-1]
                alpha=0.6,
                linewidth=2.1,
                # marker, markersize,
                label=line_label[i],
                ) 
        elif i == len(data)-2:
            im = ax.plot(
                x[i],
                y,
                linestyle='--',
                color=COLORS[-1], # cm.batlowS, COLORS[i], COLORS[-1]
                # alpha=0.6,
                linewidth=1.8,
                # marker, markersize,
                label=line_label[i],
                ) 
        else:
            im = ax.plot(
                x[i],
                y,
                # linestyle='--',
                color=COLORS[i], # cm.batlowS, COLORS[i], COLORS[-1]
                # alpha=0.6,
                linewidth=1.8,
                # marker, markersize,
                label=line_label[i],
                ) 
            # ax2 = ax.twinx() # 共享y轴但有不同x轴的图
    
    ax.set_xlabel(xlabel) 
    ax.set_ylabel(ylabel)
    
    if not xscale_log and not yscale_log:
        # ax.set_xticks([]) 
        ax.set_yticks(np.linspace(0., 1., 6)) 
    
    if xlim is not None and ylim is not None: 
        # ax.set_ylim(ylim)
        # ax.set_xlim(xlim)
        # xlim, ylim = ax.axis()
        ax.axis([*xlim, *ylim]) # xlim, ylim (不与 ax.set_xscale('log') 等冲突)
    
    for spine in ax.spines.values(): # ax.spines['bottom'], left, right, top
        spine.set_visible(True) # NOTE 与 ax.axis('off') 不兼容
        spine.set_color('black')
        spine.set_linewidth(SPINE_LINEWIDTH)
    
    if legend:
        ax.legend(
            loc='lower left',
            # fontsize=fontsize,
            frameon=False, # 边框显示
            framealpha=0, # 背景透明度 
            # ncol=1, 
            )
    
    anchor_y_spare = 1 / 18
    ax.text(
        - anchor_y_spare / aspect_ratio_r[0] * aspect_ratio_r[1], 
        1.0 + anchor_y_spare, 
        fig_label, 
        va='center', ha='center', 
        fontsize=fontsize + 0.5,
        transform=ax.transAxes, # 转而让文本的位置相对于轴的尺寸来确定，而不再是数据坐标
        )
    
    # ax.grid(axis='both', # x, y, both
    #         color='gray', linestyle='--', linewidth=0.5, alpha=0.7)
    
    if xvline is not None:
        ax.axvline(x=xvline, color='gray', linestyle='-', linewidth=0.8, alpha=1)
    if yhline is not None:
        ax.axhline(y=yhline, color='gray', linestyle='-', linewidth=0.8, alpha=1)
    
    # 线性刻度 -> 对数刻度(更清晰展示跨越多个数量级的数据的变化趋势) (NOTE 与ax.set_xticks()和ax.set_xlim()不兼容)
    if xscale_log:
        ax.set_xscale('log') 
    if yscale_log:
        ax.set_yscale('log')
    
    if not show_but_nosave:
        fig.savefig(filename, dpi=DPI, 
                    bbox_inches='tight', # 裁剪图像周围的空白区域 (NOTE 会使图像小于声明的figsize)
                    ) 
    else:
        plt.show()
    plt.close()
    

###############################################################################
###############################################################################
###############################################################################

# stability 可视化: 相关系数图 over time (多张 统计图-误差棒图errorbar)
def plot_stability_correlation(
        x1: Tuple[np.array, ...], # Simulation time(Corresponds to x2); len: dt; shape: model(share x), time-step
        x2: Tuple[np.array, ...], # No. of time step; len: dt; shape: model(share x), time-step
        mean: Tuple[np.array, ...], # len: dt; shape: model, time-step
        std: Tuple[np.array, ...], # len: dt; shape: model, time-step
        nrows: int = 2,
        ncols: int = 2,
        line_label: Tuple[str, ...] = (
            'DS 512 × 512',
            'DS 256 × 256',
            'DS 128 × 128',
            'DS 64 × 64', 
            'LaPON 64 × 64'),
        x1label: str = 'Simulation time (×$10^{-3}$ s)',
        x2label: str = 'No. of time step',
        ylabel: str = 'Vorticity correlation',
        subfig_label: Tuple[str, ...] = (
            '$\Delta t$ = 2×$10^{-3}$ s', # NOTE italic: $t$, $\mathit{\Delta}$; Δ:  $\Delta$ (LaTeX)
            '$\Delta t$ = 4×$10^{-3}$ s',
            '$\Delta t$ = 8×$10^{-3}$ s',
            '$\Delta t$ = 1.6×$10^{-2}$ s',),
        ylim_origin: Tuple[float, ...] = (0., 1.), # None
        xvline: float = (1.99 + 0.0424) * 3, # None # (Kolmogorov time of JHTDB 0.0424s, Large eddy turnover time of JHTDB 1.99s, Lyapunov time (流体混沌震荡: 2s from wiki))
        yhline: float = 0.95, # correlation 1: 完全相关; 0.95: 很强的相关性; 0.8: 强相关; 0.6: 中等相关; 0: 无关联
        yscale_log: bool = False, # 对数轴 (与xlim等冲突, 不可同时使用)
        fig_label: str = '(b)',
        legend: bool = True,
        save_dir: Path = Path('runs/test/exp'),
        figname: str = "stability_correlation.png",
        figsize: Tuple[float, ...] = None, # w, h # (7, 7 * 0.618) # (7, 7 / 0.8)
        show_but_nosave: bool = False
        ) -> None:
    """
    case1:
        x = np.linspace(0, np.pi, 7)
        y = 0.5 * np.cos(x) + 0.5
        mean = np.repeat(y[np.newaxis, ...], 5, axis=0) + np.random.rand(5,7)/10
        mean = [mean]*4
        std = [np.random.rand(5,7)/10]*4
        x1 = [np.repeat(x[np.newaxis, ...], 5, axis=0)]*4
        x2 = [np.repeat(np.linspace(0,6,7)[np.newaxis, ...], 5, axis=0)]*4
        
        plot_stability_correlation(x1, x2, mean, std, show_but_nosave=True)
        plot_stability_correlation(x1, x2, mean, std, show_but_nosave=True, nrows=4, ncols=1)
        
    case2:
        x = np.linspace(0, np.pi, 7)
        y = 0.5 * np.cos(x) + 0.5
        mean = np.repeat(y[np.newaxis, ...], 5, axis=0) + np.random.rand(5,7)/10
        mean = [mean]*4
        std = [np.random.rand(5,7)/10]*4
        x1 = [np.repeat(x[np.newaxis, ...], 5, axis=0)]*4
        x2 = [np.repeat(np.linspace(0,6,7)[np.newaxis, ...], 5, axis=0)]*4
        
        plot_stability_correlation(x1, x2, mean, std, save_dir=Path('./'))
        plot_stability_correlation(x1, x2, mean, std, save_dir=Path('./'), nrows=4, ncols=1)
    """
    filename = save_dir / figname
    figsize = (7 * 1.618, 7) if figsize is None else figsize 
    subfig_aspect_ratio_r = (1, figsize[1] / figsize[0] * ncols / nrows) # scaled w, h of subfig # (1, 1 / 0.8) (1, 0.618)
    """
    (recommended) First determine subfig_aspect_ratio_r[1], then calculate figsize.
    # figsize_h = figsize_w / ncols * nrows * subfig_aspect_ratio_r[1] # (figsize_w * 0.618)
    # figsize_w = figsize_h * ncols / nrows / subfig_aspect_ratio_r[1] # (figsize_h * 1.618)
    # subfig_aspect_ratio_r[1] = figsize_h / figsize_w * ncols / nrows # 0.618
    """
    fontsize = FONTSIZE
    
    if not show_but_nosave:
        with open(filename.with_suffix('.pkl'), 'wb') as fi: 
            cPickle.dump([x1, x2, mean, std], fi) # save data
    
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, 
                             # layout='tight'
                             layout='constrained'
                             )
    # plt.subplots_adjust(wspace=0.05, hspace=0.05) # 子图之间的间距(NOTE 与 layout 不兼容)
    
    for i in range(len(mean)):
        ax2 = axes.ravel()[i]
        ax = ax2.twiny() # 共享y轴但有不同x轴的图 (NOTE ax只有x轴, 没有自己的y轴, 故ax.set_ylabel等是无效的)
        for j in range(mean[i].shape[0]):
            if j == mean[i].shape[0]-1:
                im = ax.errorbar( # 误差棒图
                    x=x1[i][j], y=mean[i][j], 
                    # xerr, # (N,) 或 (2, N) 的 array
                    yerr=std[i][j], 
                    fmt='--o', # 数据点标记(与折线) 样式 # 'none'(不绘制数据点标记), 'o', '-o', '--o', '.', ... 
                    ms=3, # 数据点标记 大小
                    mfc=COLORS[-1], # 数据点标记 颜色 # cm.batlowS, COLORS[j], COLORS[-1]
                    # mec, # 数据点标记 边缘颜色
                    color=COLORS[-1], # 折线 颜色 # cm.batlowS, COLORS[j], COLORS[-1]
                    ecolor=COLORS[-1], # 误差棒线条 颜色 # cm.batlowS, COLORS[j], COLORS[-1]
                    elinewidth=1.25, # 误差棒线条 宽度(粗细)
                    capsize=3, # 误差棒线条 末端横杠长度(/箭头宽度)
                    # capthick=1, # 误差棒线条 末端横杠宽度(粗细)
                    # lolims, uplims, # 误差棒线条 上下界 (超过界限将由箭头符号表示)
                    # xlolims, xuplims, 
                    # alpha=0.6, 
                    label=line_label[j],
                    ) 
            else:
                im = ax.errorbar( # 误差棒图
                    x=x1[i][j], y=mean[i][j], 
                    # xerr, # (N,) 或 (2, N) 的 array
                    yerr=std[i][j], 
                    fmt='-o', # 数据点标记(与折线) 样式 # 'none'(不绘制数据点标记), 'o', '-o', '--o', '.', ... 
                    ms=3, # 数据点标记 大小
                    mfc=COLORS[j], # 数据点标记 颜色 # cm.batlowS, COLORS[j], COLORS[-1]
                    # mec, # 数据点标记 边缘颜色
                    color=COLORS[j], # 折线 颜色 # cm.batlowS, COLORS[j], COLORS[-1]
                    ecolor=COLORS[j], # 误差棒线条 颜色 # cm.batlowS, COLORS[j], COLORS[-1]
                    elinewidth=1.25, # 误差棒线条 宽度(粗细)
                    capsize=3, # 误差棒线条 末端横杠长度(/箭头宽度)
                    # capthick=1, # 误差棒线条 末端横杠宽度(粗细)
                    # lolims, uplims, # 误差棒线条 上下界 (超过界限将由箭头符号表示)
                    # xlolims, xuplims, 
                    # alpha=0.6, 
                    label=line_label[j],
                    ) 
        
        ax.set_xticks(x1[i][0])
        
        # NOTE 为ax2按ax的x1设置刻度，然后再按ax2自己的x2设置刻度标签，以此实现两个轴的相互对应
        ax2.set_xticks(x1[i][0])
        ax2.set_xticklabels(x2[i][0])
        
        if not yscale_log:
            ax2.set_yticks(np.linspace(0., 1., 6)) # ax.set_yticks 无效
        
        if not yscale_log:
            if ylim_origin is not None:
                y_spare = 0.03 * (ylim_origin[1] - ylim_origin[0])
                ylim = (ylim_origin[0] - y_spare, ylim_origin[1] + y_spare)
                ax2.set_ylim(ylim)
                
                x_spare = (y_spare / nrows * ncols) * (x1[i].max() - x1[i].min())
                xlim = (x1[i].min() - x_spare, x1[i].max() + x_spare)
                ax.set_xlim(xlim)
                ax2.set_xlim(xlim)
                
                # xlim, ylim = ax.axis()
                # ax.axis([*xlim, *ylim]) # xlim, ylim (不与 ax.set_xscale('log') 等冲突)
            
        # ax.set_xlabel(x1label) 
        # ax2.set_xlabel(x2label) 
        # ax2.set_ylabel(ylabel)
        
        for spine in ax.spines.values(): # ax.spines['bottom'], left, right, top
            spine.set_visible(True) # NOTE 与 ax.axis('off') 不兼容
            spine.set_color('black')
            spine.set_linewidth(SPINE_LINEWIDTH)
    
        ax.grid(axis='x', # x, y, both
                color='gray', linestyle='--', linewidth=0.5, alpha=0.7)
        
        if xvline is not None:
            ax.axvline(x=xvline, color='gray', linestyle='-', linewidth=0.8, alpha=1)
        if yhline is not None:
            ax.axhline(y=yhline, color='gray', linestyle='-', linewidth=0.8, alpha=1)
        
        # 线性刻度 -> 对数刻度(更清晰展示跨越多个数量级的数据的变化趋势) (NOTE 与ax.set_xticks()和ax.set_xlim()不兼容)
        if yscale_log:
            ax.set_yscale('log')
        
        if legend:
            if i == len(mean)-1:
                ax.legend(
                    loc='lower left',
                    # bbox_to_anchor=(1, 1),
                    # fontsize=fontsize,
                    frameon=False, # 边框显示
                    framealpha=0, # 背景透明度 
                    # ncol=2, 
                    # shadow=True, 
                    )
        
        text = ax.text(
            1 - 0.025, # 1 - 0.06
            1 - 0.08, # 1 - 0.13
            subfig_label[i], 
            va='top', ha='right', 
            fontsize=fontsize + 0.5,
            transform=ax.transAxes, # 转而让文本的位置相对于轴的尺寸来确定，而不再是数据坐标
            bbox=dict( # 设置背景和边框
                facecolor='white', 
                edgecolor='black', 
                boxstyle="round,pad=0.3", 
                alpha=0.45,), 
            )
    
    anchor_y_spare = 5e-3
    anchor_x = - anchor_y_spare * ncols / nrows / subfig_aspect_ratio_r[1]
    anchor_y = 1.0 + anchor_y_spare + 2e-2
    fig.text(
        anchor_x,  
        anchor_y, 
        fig_label, 
        va='center', ha='center', 
        fontsize=fontsize + 1.5,
        )
    
    fig.text(
        anchor_x,  
        0.5, 
        ylabel, 
        va='center', ha='center', 
        fontsize=fontsize + 1.5,
        rotation=90,
        )
    
    fig.text(
        0.5,  
        anchor_y,
        x1label, 
        va='center', ha='center', 
        fontsize=fontsize + 1.5,
        )
    
    fig.text(
        0.5,  
        - anchor_y_spare - 2e-2,
        x2label, 
        va='center', ha='center', 
        fontsize=fontsize + 1.5,
        )
    
    if not show_but_nosave:
        fig.savefig(filename, dpi=DPI, 
                    bbox_inches='tight', # 裁剪图像周围的空白区域 (NOTE 会使图像小于声明的figsize)
                    ) 
    else:
        plt.show()
    plt.close()


# stability 可视化: 抗噪性图(单张折线图)(*同plot_accuracy_correlation)
def plot_stability_noise_immunity(
        x: np.array, # The noise level(Every data is consistent); shape: len
        data: np.array, # shape: idx, len
        line_label: Tuple[str, ...] = (
            'DS 512 × 512',
            'DS 256 × 256',
            'DS 128 × 128',
            'DS 64 × 64', 
            'LaPON 64 × 64'),
        ylim_origin: Tuple[float, ...] = (0., 1.), # None
        xvline: float = None, # None 
        yhline: float = 0.95, # correlation 1: 完全相关; 0.95: 很强的相关性; 0.8: 强相关; 0.6: 中等相关; 0: 无关联
        xscale_log: bool = False, # 对数轴 (与xlim等冲突, 不可同时使用)
        yscale_log: bool = False, 
        xlabel: str = 'Noise (%)', 
        ylabel: str = 'Vorticity correlation', 
        fig_label: str = '(c)',
        legend: bool = True,
        save_dir: Path = Path('runs/test/exp'),
        figname: str = "stability_noise_immunity.png",
        figsize: Tuple[float, ...] = None, # w, h # (7, 7 * 0.618) # (7, 7 / 0.8)
        show_but_nosave: bool = False
        ) -> None:
    """
    case1:
        x = np.linspace(0, np.pi, 4 * 10)
        y = 0.5 * np.cos(x) + 0.5
        data = np.repeat(y[np.newaxis, ...], 5, axis=0) + np.random.rand(5,40)/10
        plot_stability_noise_immunity(x, data, show_but_nosave=True)
        plot_stability_noise_immunity(x, data, show_but_nosave=True, xscale_log=True, yscale_log=True)
        
    case2:
        x = np.linspace(0, np.pi, 4 * 10)
        y = 0.5 * np.cos(x) + 0.5
        data = np.repeat(y[np.newaxis, ...], 5, axis=0) + np.random.rand(5,40)/10
        plot_stability_noise_immunity(x, data, save_dir=Path('./'))
    """
    filename = save_dir / figname  
    figsize = (7, 7 * 0.618) if figsize is None else figsize
    aspect_ratio_r = figsize # w, h 
    fontsize = FONTSIZE
    
    if not show_but_nosave:
        with open(filename.with_suffix('.pkl'), 'wb') as fi: 
            cPickle.dump([x, data], fi) # save data
    
    fig, ax = plt.subplots(1, 1, figsize=aspect_ratio_r, 
                           # layout='tight'
                           # layout='constrained'
                           )
    # plt.subplots_adjust(wspace=0.05, hspace=0.05) # 子图之间的间距(NOTE 与 layout 不兼容)
    
    for i, y in enumerate(data):
        if i == data.shape[0]-1:
            im = ax.plot(
                x,
                y,
                linestyle='--',
                color=COLORS[-1], # cm.batlowS, COLORS[i], COLORS[-1]
                # alpha=0.6,
                linewidth=1.8,
                # marker, markersize,
                label=line_label[i],
                ) 
        else:
            im = ax.plot(
                x,
                y,
                # linestyle='--',
                color=COLORS[i], # cm.batlowS, COLORS[i], COLORS[-1]
                # alpha=0.6,
                linewidth=1.8,
                # marker, markersize,
                label=line_label[i],
                ) 
            # ax2 = ax.twinx() # 共享y轴但有不同x轴的图
    
    ax.set_xlabel(xlabel) 
    ax.set_ylabel(ylabel)
    
    if not xscale_log:
        ax.set_xticks(x) 
    if not yscale_log:
        ax.set_yticks(np.linspace(0., 1., 6)) 
    
    if not yscale_log:
        if ylim_origin is not None:
            y_spare = 0.03 * (ylim_origin[1] - ylim_origin[0])
            ylim = (ylim_origin[0] - y_spare, ylim_origin[1] + y_spare)
            ax.set_ylim(ylim)
            
            if not xscale_log: 
                x_spare = (y_spare / aspect_ratio_r[0] * aspect_ratio_r[1]) * (x.max() - x.min())
                xlim = (x.min() - x_spare, x.max() + x_spare)
                ax.set_xlim(xlim)
                
                # xlim, ylim = ax.axis()
                # ax.axis([*xlim, *ylim]) # xlim, ylim (不与 ax.set_xscale('log') 等冲突)
    
    for spine in ax.spines.values(): # ax.spines['bottom'], left, right, top
        spine.set_visible(True) # NOTE 与 ax.axis('off') 不兼容
        spine.set_color('black')
        spine.set_linewidth(SPINE_LINEWIDTH)
    
    if legend:
        ax.legend(
            loc='lower left',
            # fontsize=fontsize,
            frameon=False, # 边框显示
            framealpha=0, # 背景透明度 
            # ncol=1, 
            )
    
    anchor_y_spare = 1 / 18
    ax.text(
        - anchor_y_spare / aspect_ratio_r[0] * aspect_ratio_r[1], 
        1.0 + anchor_y_spare, 
        fig_label, 
        va='center', ha='center', 
        fontsize=fontsize + 0.5,
        transform=ax.transAxes, # 转而让文本的位置相对于轴的尺寸来确定，而不再是数据坐标
        )
    
    # grid
    # ax.grid(axis='x', # x, y, both
    #         color='gray', linestyle='--', linewidth=0.5, alpha=0.7)
    for i in x:
        ax.axvline(x=i, color='gray', linestyle='--', linewidth=0.5, alpha=0.7)
    
    if xvline is not None:
        ax.axvline(x=xvline, color='gray', linestyle='-', linewidth=0.8, alpha=1)
    if yhline is not None:
        ax.axhline(y=yhline, color='gray', linestyle='-', linewidth=0.8, alpha=1)
        
    # 线性刻度 -> 对数刻度(更清晰展示跨越多个数量级的数据的变化趋势) (NOTE 与ax.set_xticks()和ax.set_xlim()不兼容)
    if xscale_log:
        ax.set_xscale('log') 
    if yscale_log:
        ax.set_yscale('log')
    
    if not show_but_nosave:
        fig.savefig(filename, dpi=DPI, 
                    bbox_inches='tight', # 裁剪图像周围的空白区域 (NOTE 会使图像小于声明的figsize)
                    ) 
    else:
        plt.show()
    plt.close()
    

# stability 可视化: 数据缺失抗性图(单张条形图)(*类似于 plot_comparisonML_normal)
def plot_stability_data_missing_resistance(
        var1: np.array or Tuple, # (no cutout) shape: model
        var2: np.array or Tuple, # (cutout) shape: model
        cluster_label: Tuple[str, ...] = (
            'DS 64 $\\times$ 64',
            'LaPON 64 $\\times$ 64'),
        bar_label: Tuple[str, ...] = (
            'No data missing',
            'Data missing'),
        ylim_origin: Tuple[float, ...] = (0., 1.), # None
        xvline: float = None, # None
        yhline: float = 0.95, 
        yscale_log: bool = False, # 对数轴 (与xlim等冲突, 不可同时使用)
        ylabel: str = 'Vorticity correlation', 
        fig_label: str = '(d)',
        legend: bool = True,
        save_dir: Path = Path('runs/test/exp'),
        figname: str = "stability_data_missing_resistance.png",
        figsize: Tuple[float, ...] = None, # w, h # (7, 7 * 0.618) # (7, 7 / 0.8)
        show_but_nosave: bool = False
        ) -> None:
    """
    case1:
        var1 = [0.1, 0.2, ]
        var2 = [0.2, 0.3, ]
        plot_stability_data_missing_resistance(var1, var2, show_but_nosave=True)
        
    case2:
        var1 = [0.1, 0.2, ]
        var2 = [0.2, 0.3, ]
        plot_stability_data_missing_resistance(var1, var2, save_dir=Path('./'))
    """
    filename = save_dir / figname
    figsize = (7 * 0.618, 7) if figsize is None else figsize
    aspect_ratio_r = figsize # w, h 
    fontsize = FONTSIZE
    
    if not show_but_nosave:
        with open(filename.with_suffix('.pkl'), 'wb') as fi: 
            cPickle.dump([var1, var2], fi) # save data
    
    fig, ax = plt.subplots(1, 1, figsize=aspect_ratio_r, 
                           # layout='tight'
                           # layout='constrained'
                           )
    # plt.subplots_adjust(wspace=0.05, hspace=0.05) # 子图之间的间距(NOTE 与 layout 不兼容)
    
    # ax2 = ax.twinx() # 共享x轴但有不同y轴的图
    
    x = np.arange(var1.shape[0] if isinstance(var1, np.ndarray) else len(var1))
    ax.bar(
        x - 0.2, 
        var1, 
        # yerr=untrained_std, 
        # bottom, # 条形图的起始y值（堆叠条形图时使用）
        width=0.357, 
        linewidth=2, # bar边缘线宽(貌似与edgecolor参数控制的不是同一个边缘线)
        color=COLORS[0], # bar颜色 # cm.batlowS, COLORS[i], COLORS[-1]
        alpha=0.6,
        edgecolor=COLORS[0], # bar边缘颜色
        error_kw=dict(
            elinewidth=2, # 误差棒粗细
            ecolor=COLORS[0], # 误差棒颜色
            capsize=5, # 误差棒末端横杠的长短
            capthick=1.5 # 误差棒末端横杠的线宽
            ), 
        # align, # 条形图的对齐方式，可以是'center'、'edge'或'zero'
        # orientation, # 'vertical'或'horizontal'
        label=bar_label[0]
        )
    ax.bar(
        x + 0.2, 
        var2, 
        # yerr=trained_std, 
        # bottom, # 条形图的起始y值（堆叠条形图时使用）
        width=0.357, 
        linewidth=2, # bar边缘线宽(貌似与edgecolor参数控制的不是同一个边缘线)
        color=COLORS[-1], # bar颜色 # cm.batlowS, COLORS[i], COLORS[-1]
        alpha=0.6,
        edgecolor=COLORS[-1], # bar边缘颜色
        error_kw=dict(
            elinewidth=2, # 误差棒粗细
            ecolor=COLORS[-1], # 误差棒颜色
            capsize=5, # 误差棒末端横杠的长短
            capthick=1.5 # 误差棒末端横杠的线宽
            ), 
        # align, # 条形图的对齐方式，可以是'center'、'edge'或'zero'
        # orientation, # 'vertical'或'horizontal'
        label=bar_label[1]
        )
    
    ax.set_xticks(x)
    ax.set_xticklabels(cluster_label)

    ax.set_ylabel(
        ylabel, 
        # color=COLORS[0]
        )
    # ax2.set_ylabel(
    #     y2label, 
    #     # color=COLORS[-1]
    #     )
    
    if not yscale_log:
        # ax.set_xticks([]) 
        ax.set_yticks(np.linspace(0., 1., 6)) 
        # ax2.set_yticks(np.linspace(0., 1., 6)) 
    
    if not yscale_log:
        if ylim_origin is not None:
            y_spare = 0.03 * (ylim_origin[1] - ylim_origin[0])
            ylim = (ylim_origin[0] - y_spare, ylim_origin[1] + y_spare)
            ax.set_ylim(ylim)
            # ax2.set_ylim(ylim)
            
            ax.axhline(y=0, color='gray', linestyle='-', linewidth=SPINE_LINEWIDTH - 0.3, alpha=1)
    
    for spine in ax.spines.values(): # ax.spines['bottom'], left, right, top
        spine.set_visible(True) # NOTE 与 ax.axis('off') 不兼容
        spine.set_color('black')
        spine.set_linewidth(SPINE_LINEWIDTH)
    
    if legend:
        # fig.legend()
        # 合并图例
        handles, labels = ax.get_legend_handles_labels()
        # handles2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(
            # handles + handles2, labels + labels2,
            handles, labels,
            # loc='lower left',
            # fontsize=fontsize,
            # frameon=False, # 边框显示
            framealpha=0.6, # 背景透明度 
            # ncol=1, 
            )
    
    anchor_y_spare = 1 / 18
    ax.text(
        - anchor_y_spare / aspect_ratio_r[0] * aspect_ratio_r[1], 
        1.0 + anchor_y_spare, 
        fig_label, 
        va='center', ha='center', 
        fontsize=fontsize + 0.5,
        transform=ax.transAxes, # 转而让文本的位置相对于轴的尺寸来确定，而不再是数据坐标
        )
    
    # ax.grid(axis='both', # x, y, both
    #         color='gray', linestyle='--', linewidth=0.5, alpha=0.7)
    
    if xvline is not None:
        ax.axvline(x=xvline, color='gray', linestyle='-', linewidth=0.8, alpha=1)
    if yhline is not None:
        ax.axhline(y=yhline, color='gray', linestyle='-', linewidth=0.8, alpha=1)
    
    # 线性刻度 -> 对数刻度(更清晰展示跨越多个数量级的数据的变化趋势) (NOTE 与ax.set_xticks()和ax.set_xlim()不兼容)
    if yscale_log:
        ax.set_yscale('log')
        # ax2.set_yscale('log')
    
    if not show_but_nosave:
        fig.savefig(filename, dpi=DPI, 
                    # bbox_inches='tight', # 裁剪图像周围的空白区域 (NOTE 会使图像小于声明的figsize)
                    bbox_inches=matplotlib.transforms.Bbox(
                        [[figsize[0]*-0.04, 0], [figsize[0], figsize[1]]]
                        ),
                    ) 
    else:
        plt.show()
    plt.close()
    
    
###############################################################################
###############################################################################
###############################################################################

# Comparison to other ML models 可视化: 单张条形图
def plot_comparisonML_normal(
        untrained_mean: np.array or Tuple, # shape: model
        untrained_std: np.array or Tuple, # shape: model
        trained_mean: np.array or Tuple, # shape: model
        trained_std: np.array or Tuple, # shape: model
        cluster_label: Tuple[str, ...] = (
            'YOLOv5', # YOLOv5basedEPD
            'DeepONetCNN', # DeepONetbasedEPD
            'LaPON'),
        bar_label: Tuple[str, ...] = (
            'Untrained',
            'Trained'),
        ylim_origin: Tuple[float, ...] = [0., 1.], # None
        xvline: float = None, # None
        yhline: Tuple[float, ...] or float or int = [0.95,], # None
        yscale_log: bool = False, # 对数轴 (与xlim等冲突, 不可同时使用)
        y1label: str = 'Vorticity correlation (untrained)', 
        y2label: str = 'Vorticity correlation (trained)', 
        cancel_y2: bool = True, # only shoiw one y axis
        fig_label: str = '(a)', # ''
        legend: bool = True,
        save_dir: Path = Path('runs/test/exp'),
        figname: str = "comparisonML_normal.png",
        figsize: Tuple[float, ...] = None, # w, h # (7, 7 * 0.618) # (7, 7 / 0.8)
        show_but_nosave: bool = False
        ) -> None:
    """
    case1:
        untrained_mean = [0.1, 0.2, 0.15, ]
        untrained_std = [0.02, 0.03, 0.02, ]
        trained_mean = [0.2, 0.3, 0.25, ]
        trained_std = [0.03, 0.04, 0.03, ]
        plot_comparisonML_normal(untrained_mean, untrained_std, trained_mean, trained_std, show_but_nosave=True)
        
    case2:
        untrained_mean = [0.1, 0.2, 0.15, ]
        untrained_std = [0.02, 0.03, 0.02, ]
        trained_mean = [0.2, 0.3, 0.25, ]
        trained_std = [0.03, 0.04, 0.03, ]
        plot_comparisonML_normal(untrained_mean, untrained_std, trained_mean, trained_std, save_dir=Path('./'))
    """
    filename = save_dir / figname
    figsize = (7, 7 * 0.618) if figsize is None else figsize
    aspect_ratio_r = figsize # w, h 
    fontsize = FONTSIZE
    
    if not show_but_nosave:
        with open(filename.with_suffix('.pkl'), 'wb') as fi: 
            cPickle.dump([untrained_mean, untrained_std, 
                          trained_mean, trained_std], fi) # save data
    
    fig, ax = plt.subplots(1, 1, figsize=aspect_ratio_r, 
                           # layout='tight'
                           # layout='constrained'
                           )
    # plt.subplots_adjust(wspace=0.05, hspace=0.05) # 子图之间的间距(NOTE 与 layout 不兼容)
    
    ax2 = ax.twinx() # 共享x轴但有不同y轴的图
    
    x = np.arange(untrained_mean.shape[0] if isinstance(untrained_mean, np.ndarray) else len(untrained_mean))
    ax.bar(
        x - 0.2, 
        untrained_mean, 
        yerr=untrained_std, 
        # bottom, # 条形图的起始y值（堆叠条形图时使用）
        width=0.357, 
        linewidth=2, # bar边缘线宽(貌似与edgecolor参数控制的不是同一个边缘线)
        color=COLORS[0], # bar颜色 # cm.batlowS, COLORS[i], COLORS[-1]
        alpha=0.6,
        edgecolor=COLORS[0], # bar边缘颜色
        error_kw=dict(
            elinewidth=2, # 误差棒粗细
            ecolor=COLORS[0], # 误差棒颜色
            capsize=5, # 误差棒末端横杠的长短
            capthick=1.5 # 误差棒末端横杠的线宽
            ), 
        # align, # 条形图的对齐方式，可以是'center'、'edge'或'zero'
        # orientation, # 'vertical'或'horizontal'
        label=bar_label[0]
        )
    ax2.bar(
        x + 0.2, 
        trained_mean, 
        yerr=trained_std, 
        # bottom, # 条形图的起始y值（堆叠条形图时使用）
        width=0.357, 
        linewidth=2, # bar边缘线宽(貌似与edgecolor参数控制的不是同一个边缘线)
        color=COLORS[-1], # bar颜色 # cm.batlowS, COLORS[i], COLORS[-1]
        alpha=0.6,
        edgecolor=COLORS[-1], # bar边缘颜色
        error_kw=dict(
            elinewidth=2, # 误差棒粗细
            ecolor=COLORS[-1], # 误差棒颜色
            capsize=5, # 误差棒末端横杠的长短
            capthick=1.5 # 误差棒末端横杠的线宽
            ), 
        # align, # 条形图的对齐方式，可以是'center'、'edge'或'zero'
        # orientation, # 'vertical'或'horizontal'
        label=bar_label[1]
        )
    
    # ax.set_xticks(x)
    # ax.set_xticklabels(cluster_label)
    ax.set_xticks(x, cluster_label)

    if cancel_y2:
        y1label = 'Vorticity correlation'

    ax.set_ylabel(
        y1label, 
        # color=COLORS[0]
        )
    if not cancel_y2:
        ax2.set_ylabel(
            y2label, 
            # color=COLORS[-1]
            )
    
    if not yscale_log:
        # ax.set_xticks([]) 
        ax.set_yticks(np.linspace(0., 1., 6)) 
        if not cancel_y2:
            ax2.set_yticks(np.linspace(0., 1., 6)) 
        else:
            ax2.set_yticks([])
    
    if not yscale_log:
        if ylim_origin is not None:
            y_spare = 0.03 * (ylim_origin[1] - ylim_origin[0])
            ylim = (ylim_origin[0] - y_spare, ylim_origin[1] + y_spare)
            ax.set_ylim(ylim)
            ax2.set_ylim(ylim)
    
    for spine in ax.spines.values(): # ax.spines['bottom'], left, right, top
        spine.set_visible(True) # NOTE 与 ax.axis('off') 不兼容
        spine.set_color('black')
        spine.set_linewidth(SPINE_LINEWIDTH)
    
    if legend:
        # fig.legend()
        # 合并图例
        handles, labels = ax.get_legend_handles_labels()
        handles2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(
            handles + handles2, labels + labels2,
            # loc='lower left',
            # fontsize=fontsize,
            # frameon=False, # 边框显示
            framealpha=0.6, # 背景透明度 
            # ncol=1, 
            )
    
    anchor_y_spare = 1 / 18
    ax.text(
        - anchor_y_spare / aspect_ratio_r[0] * aspect_ratio_r[1], 
        1.0 + anchor_y_spare, 
        fig_label, 
        va='center', ha='center', 
        fontsize=fontsize + 0.5,
        transform=ax.transAxes, # 转而让文本的位置相对于轴的尺寸来确定，而不再是数据坐标
        )
    
    # ax.grid(axis='both', # x, y, both
    #         color='gray', linestyle='--', linewidth=0.5, alpha=0.7)
    
    ax.axhline(y=0, color='gray', linestyle='-', linewidth=SPINE_LINEWIDTH - 0.5, alpha=1)
    if xvline is not None:
        ax.axvline(x=xvline, color='gray', linestyle='-', linewidth=0.8, alpha=1)
    if yhline is not None:
        if isinstance(yhline, [float, int]):
            ax.axhline(y=yhline, color='gray', linestyle='-', linewidth=0.8, alpha=1)
        else:
            for yhline_i in yhline:
                ax.axhline(y=yhline_i, color='gray', linestyle='--', linewidth=0.8, alpha=1)
    
    # 线性刻度 -> 对数刻度(更清晰展示跨越多个数量级的数据的变化趋势) (NOTE 与ax.set_xticks()和ax.set_xlim()不兼容)
    if yscale_log:
        ax.set_yscale('log')
        ax2.set_yscale('log')
    
    if not show_but_nosave:
        fig.savefig(filename, dpi=DPI, 
                    bbox_inches='tight', # 裁剪图像周围的空白区域 (NOTE 会使图像小于声明的figsize)
                    ) 
    else:
        plt.show()
    plt.close()
    

# Comparison to other ML models 可视化: 多张横向条形图 (* 在 plot_comparisonML_normal 基础上扩展)
def plot_comparisonML_comprehensiveness(
        data: Tuple[np.array, ...] or np.array, # shape: loader(row), dtdx(col), model(sub-row/cluster)
        top_title: str = '1% low value of vorticity correlation',
        left_title: Tuple[str, ...] = (
            # "Forced turbulence",
            "Extra time", 
            "Extra space", 
            "Decaying", 
            "More turbulent"
            ),
        bottom_title: Tuple[str, ...] = (
            "$\\Delta t$, 4$\\Delta x_i$",  
            "$\\Delta t$, $\\Delta x_i$",  
            "4$\\Delta t$, $\\Delta x_i$"
            ),
        cluster_label: Tuple[str, ...] = (
            'YOLOv5', # YOLOv5basedEPD
            'DeepONetCNN', # DeepONetbasedEPD
            'LaPON'),
        xlim_origin: Tuple[float, ...] = [0., 1.], # None
        xvline: float = 0.95, # None
        yhline: float = None, 
        xscale_log: bool = False, # 对数轴 (与xlim等冲突, 不可同时使用)
        fig_label: str = '(b)', # ''
        # legend: bool = True,
        save_dir: Path = Path('runs/test/exp'),
        figname: str = "comparisonML_comprehensiveness.png",
        figsize: Tuple[float, ...] = None, # w, h # (7, 7 * 0.618) # (7, 7 / 0.8)
        show_but_nosave: bool = False
        ) -> None:
    """
    case1:
        # data = np.random.rand(4, 3, 3)
        data = np.arange(4*3*3).reshape(4, 3, 3)
        data = data / data.max()
        plot_comparisonML_comprehensiveness(data, show_but_nosave=True)
        
    case2:
        data = np.random.rand(4, 3, 3)
        plot_comparisonML_comprehensiveness(data, save_dir=Path('./'))
    """
    filename = save_dir / figname
    nrows = len(left_title)
    ncols = len(bottom_title)
    figsize = (7, 7 / ncols * nrows * 0.618) if figsize is None else figsize
    fontsize = FONTSIZE # - 0.5
    
    if not show_but_nosave:
        with open(filename.with_suffix('.pkl'), 'wb') as fi: 
            cPickle.dump(data, fi) # save data
    
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, 
                           # layout='tight'
                           layout='constrained'
                           )
    # plt.subplots_adjust(wspace=0.05, hspace=0.05) # 子图之间的间距(NOTE 与 layout 不兼容)
    
    # data = np.array(data)
    indices = np.arange(len(cluster_label))[::-1] # 从上往下画
    
    for i in range(len(left_title)):
        for j in range(len(bottom_title)):
            ax = axes[i, j]
            ax2 = ax.twiny() # 共享y轴但有不同x轴的图
            ax2.barh(
                indices, 
                data[i, j, :], 
                # xerr, 
                # left, # 条形图的起始x值（堆叠条形图时使用）
                # height=0.357,
                linewidth=2, # bar边缘线宽(貌似与edgecolor参数控制的不是同一个边缘线)
                color=COLORS[:len(cluster_label)], # bar颜色 # cm.batlowS, COLORS[i], COLORS[-1]
                alpha=0.8,
                edgecolor=COLORS[:len(cluster_label)], # bar边缘颜色
                # align, # 条形图的对齐方式，可以是'center'、'edge'或'zero'
                # label=bar_label[0],
                )
            
            if i == len(left_title) - 1:
                ax.set_xlabel(bottom_title[j], fontsize = fontsize)
            
            if j == 0:
                # ax.set_yticks(indices)
                # ax.set_yticklabels(cluster_label)
                ax.set_yticks(indices, cluster_label, 
                              fontsize = fontsize - 0.5)
                
                # ax2.set_ylabel(left_title[i], fontsize = fontsize - 0.5) # 无效
                ax.set_ylabel(left_title[i], fontsize = fontsize - 0.5)
            else:
                ax2.set_yticks(indices, [])
            
            ax.set_xticks([])
            
            if xlim_origin is not None:
                x_spare = 0.03 * (xlim_origin[1] - xlim_origin[0])
                xlim = (xlim_origin[0] - x_spare, xlim_origin[1] + x_spare)
                # ax.set_xlim(xlim)
                ax2.set_xlim(xlim)
            
            if i == 0:
                if not xscale_log:
                    # ax2.set_xticks(np.linspace(0., 1., 6)) 
                    ax2.set_xticks(np.linspace(0., 1., 6), 
                                   ['', 0.2, 0.4, 0.6, 0.8, ''], 
                                   fontsize = fontsize) 
            else:
                # ax2.set_xticklabels([])
                ax2.set_xticks(np.linspace(0., 1., 6), [])
            
            for spine in ax.spines.values(): # ax.spines['bottom'], left, right, top
                spine.set_visible(True) # NOTE 与 ax.axis('off') 不兼容
                spine.set_color('black')
                spine.set_linewidth(SPINE_LINEWIDTH)
                
            ax.grid(axis='y', # x, y, both
                    color='gray', linestyle='--', linewidth=0.5, alpha=0.7)
            ax2.axvline(x=0, color='gray', linestyle='-', linewidth=SPINE_LINEWIDTH - 0.5, alpha=1)
            
            if xvline is not None:
                ax2.axvline(x=xvline, color='gray', linestyle='-', linewidth=0.8, alpha=1)
            if yhline is not None:
                ax2.axhline(y=yhline, color='gray', linestyle='-', linewidth=0.8, alpha=1)
            
            # 线性刻度 -> 对数刻度(更清晰展示跨越多个数量级的数据的变化趋势) (NOTE 与ax.set_xticks()和ax.set_xlim()不兼容)
            if xscale_log:
                # ax.set_xscale('log')
                ax2.set_xscale('log')
    
    # if legend:
    #     # fig.legend()
    #     # 合并图例
    #     handles, labels = ax.get_legend_handles_labels()
    #     handles2, labels2 = ax2.get_legend_handles_labels()
    #     axes[-1, -1].legend(
    #         handles + handles2, labels + labels2,
    #         # loc='lower left',
    #         # fontsize=fontsize,
    #         # frameon=False, # 边框显示
    #         framealpha=0.6, # 背景透明度 
    #         # ncol=1, 
    #         )
    
    axes[0, 1].set_title(top_title, fontsize = fontsize)
    
    ax = axes[0, 0]
    lim = ax.axis() # xlim, ylim
    anchor_x_spare = 1 / 18
    ax.text(
        # lim[0] - anchor_x_spare * (lim[1] - lim[0]), 
        # lim[3] + anchor_x_spare * (lim[3] - lim[2]), 
        -0.65,
        1.35,
        fig_label, 
        va='center', ha='center', 
        fontsize=fontsize + 0.5,
        transform=ax.transAxes, # 转而让文本的位置相对于轴的尺寸来确定，而不再是数据坐标
        )
    
    if not show_but_nosave:
        fig.savefig(filename, dpi=DPI, 
                    bbox_inches='tight', # 裁剪图像周围的空白区域 (NOTE 会使图像小于声明的figsize)
                    ) 
    else:
        plt.show()
    plt.close()


###############################################################################
###############################################################################
###############################################################################

def resize_image(
    image: Image.Image,
    longest_side: int,
    resample: int = Image.Resampling.NEAREST,
) -> Image.Image:
    """Resize an image, preserving its aspect ratio."""
    resize_factor = longest_side / max(image.size)
    new_size = tuple(round(s * resize_factor) for s in image.size)
    return image.resize(new_size, resample)


def concat_fig(
        img_path_ls: Tuple[Path, ...] = [
            Path('accuracy_heat.png'),
            Path('accuracy_correlation.png'),
            Path('accuracy_spectrum.png'),
            ],
        orientation: str = 'h', # v, vertical, h, horizontal
        out_figname: str = None,
        ) -> None:
    '''
    auto resize and concatenate figs
    
    case:
        concat_fig(orientation='h')
        concat_fig(orientation='v')
    
    case2:
        concat_fig([
            'stability_heat.png',
            'stability_correlation.png',
            'stability_noise_immunity.png',
            ], orientation='h')
        concat_fig([
            'stability_heat.png',
            'stability_correlation.png',
            'stability_noise_immunity.png',
            ], orientation='v')
    '''
    imgs = [Image.open(img_path) for img_path in img_path_ls]
    out_dir = img_path_ls[0].parent
    out_figname = (img_path_ls[0].with_suffix('').name + f'_etc_concatenated_{orientation}.png' 
                   if out_figname is None else out_figname)
    out_figpath = out_dir / out_figname
    
    if orientation in ['h', 'horizontal']:
        total_height = max(img.height for img in imgs)
    elif orientation in ['v', 'vertical']:
        total_width = max(img.width for img in imgs)
    else:
        raise ValueError(f"orientation={orientation}, but must in [v, vertical, h, horizontal]")
    
    for i, img in enumerate(imgs):
        size = img.size # w, h
        if orientation in ['h', 'horizontal']:
            new_shape = (int(size[0] / size[1] * total_height),
                         total_height)
        elif orientation in ['v', 'vertical']:
            new_shape = (total_width, 
                         int(size[1] / size[0] * total_width))
        imgs[i] = img.resize(new_shape)
    
    if orientation in ['h', 'horizontal']:
        total_width = sum(img.width for img in imgs)
    elif orientation in ['v', 'vertical']:
        total_height = sum(img.height for img in imgs)
        
    new_image = Image.new('RGB', (total_width, total_height))
    
    if orientation in ['h', 'horizontal']:
        x_offset = 0
        for img in imgs:
            new_image.paste(img, (x_offset, 0))
            x_offset += img.width
    elif orientation in ['v', 'vertical']:
        y_offset = 0
        for img in imgs:
            new_image.paste(img, (0, y_offset))
            y_offset += img.height
    
    new_image.save(out_figpath)

    
