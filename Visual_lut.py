import numpy as np
import matplotlib
matplotlib.use('Agg')  # 无GUI后端，适用于服务器环境

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import sys
import os

# ── 路径处理，确保能找到同目录下的模块 ──────────────────────────────────────
sys.path.insert(0, os.path.dirname(__file__))

try:
    from LUT import lut as hardware_lut
except ImportError:
    print("错误：找不到 LUT.py 文件。请确保两个文件在同一目录下。")
    sys.exit(1)

try:
    from plot_style import (
        apply_journal_style,
        get_figure_size,
        DOUBLE_COLUMN_WIDTH_INCH,
        PRINT_DPI,
    )
    apply_journal_style()
except ImportError:
    print("警告：找不到 plot_style.py，将使用默认样式。")


# ── 数据准备 ─────────────────────────────────────────────────────────────────
def prepare_data(lut: np.ndarray):
    """计算理想线性 LUT 及其与硬件 LUT 的偏差。"""
    num_levels = len(lut)
    levels     = np.arange(1, num_levels + 1)
    ideal_lut  = np.linspace(lut.min(), lut.max(), num_levels)
    deviation  = lut - ideal_lut          # 硬件 LUT 相对理想值的偏差
    return levels, ideal_lut, deviation


# ── 可视化主函数 ──────────────────────────────────────────────────────────────
def generate_lut_visualization(output_filename: str = "lut_nonuniformity_visualization.png"):
    """
    生成符合学术期刊规范的 LUT 非均匀性可视化图表。

    上子图：理想线性 LUT 与硬件实测 LUT 的对比。
    下子图：硬件 LUT 相对理想线性 LUT 的逐级偏差（非均匀性量化）。
    """
    print("正在生成 LUT 非均匀性可视化图表…")

    levels, ideal_lut, deviation = prepare_data(hardware_lut)
    num_levels = len(hardware_lut)

    # ── 画布与子图布局 ────────────────────────────────────────────────────────
    fig_w, _ = get_figure_size("double", aspect_ratio=0.75)
    fig = plt.figure(figsize=(fig_w, fig_w * 0.75))

    gs = gridspec.GridSpec(
        2, 1,
        figure=fig,
        height_ratios=[2, 1],   # 上图占 2/3，下图占 1/3
        hspace=0.08,             # 子图间距（共享 x 轴时保持紧凑）
    )

    ax_main = fig.add_subplot(gs[0])
    ax_dev  = fig.add_subplot(gs[1], sharex=ax_main)

    # ── 上子图：LUT 对比 ──────────────────────────────────────────────────────
    ax_main.plot(
        levels, ideal_lut,
        color="#2166ac", linewidth=1.2, linestyle="--",
        label="Ideal linear $\\gamma$", zorder=2,
    )
    ax_main.scatter(
        levels, hardware_lut,
        color="#d6604d", s=8, linewidths=0,
        label="Hardware $\\gamma$", zorder=3,
        alpha=0.85,
    )
    ax_main.axhline(
        y=0.0, color="#888888", linewidth=0.7,
        linestyle=":", label="$\\gamma = 0$ reference", zorder=1,
    )

    ax_main.set_xlim(0, num_levels + 1)
    ax_main.set_ylim(-1.05, 1.05)
    ax_main.set_ylabel("Power contrast $\\gamma$")
    ax_main.legend(loc="upper left", ncol=3)

    # 隐藏上子图的 x 轴刻度标签（与下子图共享）
    plt.setp(ax_main.get_xticklabels(), visible=False)

    # ── 下子图：逐级偏差 ──────────────────────────────────────────────────────
    ax_dev.bar(
        levels, deviation,
        width=0.8, color="#4dac26", alpha=0.75,
        label="Nonuniformity $\\Delta\\gamma$",
    )
    ax_dev.axhline(y=0.0, color="#888888", linewidth=0.7, linestyle=":")

    # 标注 RMS 偏差
    rms = np.sqrt(np.mean(deviation ** 2))
    ax_dev.text(
        0.98, 0.92,
        f"RMS $\\Delta\\gamma$ = {rms:.4f}",
        transform=ax_dev.transAxes,
        ha="right", va="top",
        fontsize=7,
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="none", alpha=0.8),
    )

    ax_dev.set_xlim(0, num_levels + 1)
    ax_dev.set_ylim(
        deviation.min() * 1.3 if deviation.min() < 0 else -0.05,
        deviation.max() * 1.3 if deviation.max() > 0 else  0.05,
    )
    ax_dev.set_xlabel("Level index")
    ax_dev.set_ylabel("$\\Delta\\gamma$")
    ax_dev.legend(loc="upper left")

    # ── 共享刻度设置 ─────────────────────────────────────────────────────────
    tick_positions = [1, 32, 64, 96, 128]
    ax_dev.set_xticks(tick_positions)

    # ── 保存 ─────────────────────────────────────────────────────────────────
    fig.savefig(output_filename, dpi=PRINT_DPI)
    plt.close(fig)
    print(f"图表已保存至 '{output_filename}'")


# ── 入口 ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    generate_lut_visualization()
