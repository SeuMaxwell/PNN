# -*- coding: utf-8 -*-
"""
学术期刊绘图风格配置模块

本模块提供符合主流学术期刊要求的 Matplotlib 全局风格配置。
支持单栏 (single-column) 和双栏 (double-column) 两种布局。

典型期刊要求参考:
- Nature/Science: 单栏 89 mm, 双栏 183 mm
- IEEE: 单栏 3.5 in (88.9 mm), 双栏 7.16 in (181.9 mm)
- ACS/RSC: 单栏 3.25 in (82.5 mm), 双栏 7 in (177.8 mm)
- Elsevier: 单栏 90 mm, 双栏 190 mm
"""

import matplotlib as mpl

# ============================================================================
#                           物理尺寸常量 (英寸)
# ============================================================================
# 根据常见期刊双栏排版要求，建议使用以下尺寸

# 单栏宽度 (适用于 Nature, Science, IEEE 等)
SINGLE_COLUMN_WIDTH_INCH = 3.5     # ≈ 89 mm

# 双栏宽度 (适用于期刊全页宽图像)
DOUBLE_COLUMN_WIDTH_INCH = 7.0     # ≈ 178 mm

# 最大高度限制 (建议不超过 9 英寸)
MAX_HEIGHT_INCH = 9.0

# 默认 DPI 设置
SCREEN_DPI = 150     # 屏幕预览
PRINT_DPI = 300      # 打印质量
PUBLICATION_DPI = 600  # 出版质量

# Colorbar 相关设置
COLORBAR_WIDTH_RATIO = 0.08  # Colorbar 占图像宽度的比例


# ============================================================================
#                           学术期刊风格配置
# ============================================================================

JOURNAL_STYLE = {
    # ----- 字体设置 -----
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'font.size': 8,                    # 默认字体大小 (pt)

    # ----- 坐标轴标签 -----
    'axes.labelsize': 8,               # 轴标签字体大小
    'axes.titlesize': 9,               # 子图标题字体大小
    'axes.labelweight': 'normal',      # 轴标签字重
    'axes.linewidth': 0.8,             # 坐标轴线宽

    # ----- 刻度设置 -----
    'xtick.labelsize': 7,
    'ytick.labelsize': 7,
    'xtick.direction': 'in',           # 刻度朝内 (符合多数期刊要求)
    'ytick.direction': 'in',
    'xtick.major.size': 3.0,           # 主刻度长度
    'ytick.major.size': 3.0,
    'xtick.major.width': 0.6,          # 主刻度线宽
    'ytick.major.width': 0.6,
    'xtick.minor.size': 1.5,           # 次刻度长度
    'ytick.minor.size': 1.5,
    'xtick.minor.width': 0.4,
    'ytick.minor.width': 0.4,
    'xtick.top': True,                 # 顶部刻度
    'ytick.right': True,               # 右侧刻度

    # ----- 图例设置 -----
    'legend.fontsize': 7,
    'legend.frameon': False,           # 无边框图例
    'legend.loc': 'best',

    # ----- 线条和标记 -----
    'lines.linewidth': 1.0,
    'lines.markersize': 4,

    # ----- 图像输出 -----
    'figure.dpi': SCREEN_DPI,
    'savefig.dpi': PRINT_DPI,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.02,

    # ----- 其他 -----
    'axes.grid': False,
    'image.cmap': 'viridis',
}


# ============================================================================
#                           配色方案
# ============================================================================

# 推荐的科学可视化色图
RECOMMENDED_CMAPS = {
    'sequential': ['viridis', 'plasma', 'inferno', 'magma', 'cividis'],
    'diverging': ['coolwarm', 'RdBu_r', 'seismic'],
    'qualitative': ['tab10', 'Set2', 'Dark2'],
}

# 电场强度推荐色图
EFIELD_CMAP = 'inferno'

# ============================================================================
#                           Unified dual-colour palette (colour-blind safe)
# ============================================================================
COLOR_COMPUTER   = '#4CAF50'   # medium green  – 64-bit Computer (full precision)
COLOR_PNN        = '#2196F3'   # medium blue   – PNN (LUT quantised)
COLOR_HIGHLIGHT  = '#FF5722'   # orange-red    – error bars / emphasis
COLOR_REFERENCE  = '#9E9E9E'   # mid-grey      – reference / dashed lines
COLOR_UNIFORM    = '#FF9800'   # amber         – Uniform LUT baseline

# Confusion-matrix colour maps
CM_CMAP_COMPUTER = 'Greens'
CM_CMAP_PNN      = 'Blues'

def get_standard_dpi() -> int:
    """Return the standard save DPI for all figures (PRINT_DPI = 300)."""
    return PRINT_DPI


# ============================================================================
#                           辅助函数
# ============================================================================

def apply_journal_style():
    mpl.rcParams.update(JOURNAL_STYLE)


def get_figure_size(width='double', aspect_ratio=0.618):
    """
    根据期刊栏宽和宽高比计算图像尺寸。
    
    Parameters
    ----------
    width : str or float
        'single' - 单栏宽度 (~89 mm)
        'double' - 双栏宽度 (~178 mm)
        或直接指定宽度数值 (英寸)
    aspect_ratio : float
        高度/宽度比，默认黄金分割比 0.618
    
    Returns
    -------
    tuple
        (width_inch, height_inch)
    
    Example
    -------
    >>> get_figure_size('double', aspect_ratio=0.4)
    (7.0, 2.8)
    """
    if width == 'single':
        w = SINGLE_COLUMN_WIDTH_INCH
    elif width == 'double':
        w = DOUBLE_COLUMN_WIDTH_INCH
    else:
        w = float(width)
    
    h = min(w * aspect_ratio, MAX_HEIGHT_INCH)
    return (w, h)


def get_subplot_figsize(n_cols, n_rows=1, width='double', subplot_aspect=1.0, 
                        cbar_width_ratio=0.05):
    """
    计算多子图布局的图像尺寸。
    
    Parameters
    ----------
    n_cols : int
        子图列数
    n_rows : int
        子图行数，默认1
    width : str or float
        总宽度，'single', 'double' 或数值
    subplot_aspect : float
        单个子图的高宽比
    cbar_width_ratio : float
        colorbar占总宽度的比例
    
    Returns
    -------
    tuple
        (fig_width, fig_height)
    """
    if width == 'single':
        total_width = SINGLE_COLUMN_WIDTH_INCH
    elif width == 'double':
        total_width = DOUBLE_COLUMN_WIDTH_INCH
    else:
        total_width = float(width)
    
    # 每个子图的有效宽度
    subplot_width = total_width * (1 - cbar_width_ratio) / n_cols
    subplot_height = subplot_width * subplot_aspect
    fig_height = subplot_height * n_rows
    
    return (total_width, min(fig_height, MAX_HEIGHT_INCH))
