"""
generate_summary_from_reports.py
从现有报告文件生成综合摘要
运行: python generate_summary_from_reports.py
"""

import os
import re
import json

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from LUT import lut as hardware_lut

try:
    from plot_style import (
        apply_journal_style, get_figure_size, PRINT_DPI,
        COLOR_COMPUTER, COLOR_PNN, COLOR_HIGHLIGHT, COLOR_REFERENCE, COLOR_UNIFORM,
    )
    apply_journal_style()
    _PLOT_DPI = PRINT_DPI
except ImportError:
    _PLOT_DPI = 150
    COLOR_COMPUTER, COLOR_PNN = '#4CAF50', '#2196F3'
    COLOR_HIGHLIGHT, COLOR_REFERENCE, COLOR_UNIFORM = '#FF5722', '#9E9E9E', '#FF9800'


def read_file_safe(filepath, mode='r'):
    """通用文件读取函数，自动尝试多种编码解决编码问题。
    
    Args:
        filepath: 文件路径
        mode: 读取模式，'r' 返回字符串，'lines' 返回行列表
    
    Returns:
        字符串或行列表
    """
    # 尝试的编码顺序：utf-8 优先，然后是中文编码
    encodings = ['utf-8', 'gbk', 'gb2312', 'gb18030', 'latin-1']
    
    for encoding in encodings:
        try:
            with open(filepath, 'r', encoding=encoding) as f:
                if mode == 'lines':
                    return f.readlines()
                else:
                    return f.read()
        except UnicodeDecodeError:
            continue
    
    # 所有编码都失败时，使用 utf-8 并忽略错误
    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
        if mode == 'lines':
            return f.readlines()
        else:
            return f.read()


def parse_recognition_report(filepath):
    """解析识别任务报告文件，提取准确率"""
    content = read_file_safe(filepath)

    # 匹配 "Full Precision (64-bit Computer):  66.95%"
    computer_match = re.search(r'(?:Full Precision|Full Precision\s*\(64-bit Computer\))[:\s]+(\d+\.\d+)%', content)
    # 匹配 "PNN Simulation (LUT Quantized):  66.89%"
    pnn_match = re.search(r'PNN Simulation[^:]*:\s*(\d+\.\d+)%', content)
    # 匹配 "Number of classes:  10"
    classes_match = re.search(r'Number of classes[:\s]+(\d+)', content)

    return {
        'computer': float(computer_match.group(1)) / 100 if computer_match else None,
        'pnn': float(pnn_match.group(1)) / 100 if pnn_match else None,
        'num_classes': int(classes_match.group(1)) if classes_match else 10,
    }


def parse_bit_sensitivity_report(filepath):
    """解析比特敏感性报告"""
    lines = read_file_safe(filepath, mode='lines')

    baseline = None
    results = []

    for line in lines:
        if 'Baseline' in line:
            match = re.search(r'(\d+\.\d+)%', line)
            if match:
                baseline = float(match.group(1)) / 100
        elif re.match(r'\s*\d+\s+\d+', line):
            parts = line.split()
            if len(parts) >= 4:
                # 去除准确率中的 % 符号
                acc_str = parts[2].replace('%', '')
                results.append({
                    'bits': int(parts[0]),
                    'levels': int(parts[1]),
                    'accuracy': float(acc_str) / 100,
                })

    return {'baseline_acc': baseline, 'results': results}



def parse_noise_robustness_report(filepath):
    """解析噪声鲁棒性报告"""
    lines = read_file_safe(filepath, mode='lines')

    baseline = clean_acc = None
    results = []

    for line in lines:
        if 'Baseline' in line:
            match = re.search(r'(\d+\.\d+)%', line)
            if match:
                baseline = float(match.group(1)) / 100
        elif 'Clean PNN' in line:
            match = re.search(r'(\d+\.\d+)%', line)
            if match:
                clean_acc = float(match.group(1)) / 100
        elif re.match(r'\s*\d+\.\d+%', line):
            parts = line.split()
            if len(parts) >= 3:
                results.append({
                    'sigma': float(parts[0].replace('%', '')) / 100,
                    'mean': float(parts[1].replace('%', '')) / 100,
                    'std': float(parts[2].replace('%', '')) / 100,
                })

    return {'baseline_acc': baseline, 'clean_acc': clean_acc, 'results': results}



def parse_lut_comparison_report(filepath):
    """解析 LUT 对比报告"""
    content = read_file_safe(filepath)

    results = {}

    for ds_name in ['MNIST','Fashion-MNIST', 'MedMNIST', 'CIFAR-10']:
        pattern = rf'--- {ds_name} ---\s*' \
                  r'Computer:\s*(\d+\.\d+)%\s*' \
                  r'Real LUT:\s*(\d+\.\d+)%\s*' \
                  r'Uniform LUT:\s*(\d+\.\d+)%'
        match = re.search(pattern, content)
        if match:
            results[ds_name] = {
                'computer': float(match.group(1)) / 100,
                'real_lut': float(match.group(2)) / 100,
                'uniform_lut': float(match.group(3)) / 100,
            }

    return {'results': results}


def generate_summary_from_files(output_dir='results_summary'):
    """从现有报告文件生成综合摘要"""
    os.makedirs(output_dir, exist_ok=True)

    # ========== 1. 读取识别任务结果 ==========
    task_configs = [
        # (任务名, 报告文件路径, 报告文件名)
        ('mnist', 'results_mnist', 'final_recognition_report.txt'),
        ('fashion_mnist', 'results_fashion_mnist', 'final_recognition_report.txt'),
        ('speech', 'results_speech', 'final_recognition_report.txt'),
        ('cifar10', 'results_cifar10', 'report.txt'),
        ('medmnist', 'results_medmnist', 'report.txt'),
        ('emnist_letters', 'results_emnist_letters', 'report.txt'),
        ('cifar100_coarse', 'results_cifar100_coarse', 'report.txt'),
    ]

    combined_results = []
    for task_name, task_dir, report_file in task_configs:
        filepath = os.path.join(task_dir, report_file)
        if os.path.exists(filepath):
            data = parse_recognition_report(filepath)
            if data['computer'] is not None:
                combined_results.append({
                    'task': task_name,
                    'computer': data['computer'],
                    'pnn': data['pnn'],
                    'num_classes': data['num_classes'],
                })
                print(f"  ✓ 读取 {task_name}: Computer {data['computer'] * 100:.2f}%, PNN {data['pnn'] * 100:.2f}%")

    if not combined_results:
        print("未找到任何任务结果！")
        return

    # ========== 2. 读取硬件分析结果 ==========
    hw_results = {}

    bit_report = 'results_bit_sensitivity/report.txt'
    if os.path.exists(bit_report):
        hw_results['bit_sensitivity'] = parse_bit_sensitivity_report(bit_report)
        print(f"  ✓ 读取比特敏感性分析")

    noise_report = 'results_noise_robustness/report.txt'
    if os.path.exists(noise_report):
        hw_results['noise_robustness'] = parse_noise_robustness_report(noise_report)
        print(f"  ✓ 读取噪声鲁棒性分析")

    lut_report = 'results_lut_comparison/report.txt'
    if os.path.exists(lut_report):
        hw_results['lut_comparison'] = parse_lut_comparison_report(lut_report)
        print(f"  ✓ 读取LUT对比分析")

    # ========== 3. 生成图表和报告 ==========
    _generate_comparison_chart(combined_results, output_dir)
    _generate_radar_chart(combined_results, output_dir)
    _generate_text_report(combined_results, hw_results, output_dir)
    _generate_json_results(combined_results, output_dir)

    if hw_results:
        _generate_hw_summary_figure(hw_results, output_dir)

    print(f"\n  ✓ 摘要已生成 → '{output_dir}/'")


def _generate_comparison_chart(combined_results, output_dir):
    """生成所有任务对比柱状图"""
    _TASK_DISPLAY_NAMES = {
        'mnist': 'MNIST\n(10 classes)',
        'fashion_mnist': 'Fashion-MNIST\n(10 classes)',
        'speech': 'Speech FSDD\n(10 classes)',
        'cifar10': 'CIFAR-10\n(10 classes)',
        'medmnist': 'MedMNIST\n(9 classes)',
        'emnist_letters': 'EMNIST\n(26 classes)',
        'cifar100_coarse': 'CIFAR-100\n(20 classes)',
    }

    # 按PNN准确率降序排列
    combined_results = sorted(combined_results, key=lambda r: r['pnn'], reverse=True)

    task_names = [_TASK_DISPLAY_NAMES.get(r['task'], r['task'].replace('_', '\n'))
                  for r in combined_results]
    computer_accs = [r['computer'] * 100 for r in combined_results]
    pnn_accs = [r['pnn'] * 100 for r in combined_results]

    try:
        _fs_sum = get_figure_size('double', aspect_ratio=0.5)
    except NameError:
        _fs_sum = (7.0, 3.5)
    fig, ax = plt.subplots(figsize=_fs_sum)
    x = np.arange(len(task_names))
    w = 0.38

    bars1 = ax.bar(x - w / 2, computer_accs, w, label='64-bit Computer',
                   color=COLOR_COMPUTER, edgecolor='white', linewidth=0.8)
    bars2 = ax.bar(x + w / 2, pnn_accs, w, label='7-bit PNN',
                   color=COLOR_PNN, edgecolor='white', linewidth=0.8)

    # 添加数值标签
    for bars in [bars1, bars2]:
        for bar in bars:
            h = bar.get_height()
            ax.annotate(f'{h:.1f}%', xy=(bar.get_x() + bar.get_width() / 2, h),
                        xytext=(0, 4), textcoords='offset points',
                        ha='center', va='bottom', fontsize=7, fontweight='bold')

    # 添加下降幅度标注
    # for i, (c_acc, n_acc) in enumerate(zip(computer_accs, pnn_accs)):
    #     drop = c_acc - n_acc
    #     ax.annotate(f'↓{drop:.2f}%', xy=(x[i], (c_acc + n_acc) / 2),
    #                 xytext=(6, 0), textcoords='offset points',
    #                 ha='left', va='center', fontsize=7, color='#c0392b')

    ax.set_xticks(x)
    ax.set_xticklabels(task_names, fontsize=7)
    ax.set_ylabel('Test Accuracy (%)', fontsize=8)
    ax.legend(fontsize=7, loc='upper right')
    ax.grid(False)
    ax.set_ylim(0, 115)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, 'all_tasks_comparison.png'), dpi=_PLOT_DPI)
    plt.close(fig)


def _generate_radar_chart(combined_results, output_dir):
    """生成雷达图：各任务的 PNN 精度保持率"""
    _TASK_SHORT_NAMES = {
        'mnist':           'MNIST',
        'fashion_mnist':   'Fashion-MNIST',
        'speech':          'FSDD',
        'cifar10':         'CIFAR-10',
        'medmnist':        'MedMNIST',
        'emnist_letters':  'EMNIST',
        'cifar100_coarse': 'CIFAR-100',
    }

    combined_results = sorted(combined_results, key=lambda r: r['pnn'], reverse=True)

    task_names = [_TASK_SHORT_NAMES.get(r['task'], r['task']) for r in combined_results]
    computer_accs = [r['computer'] * 100 for r in combined_results]
    pnn_accs = [r['pnn'] * 100 for r in combined_results]

    retention_ratios = [
        (n / c * 100) if c > 0 else 0.0
        for c, n in zip(computer_accs, pnn_accs)
    ]
    n_tasks = len(task_names)
    angles = np.linspace(0, 2 * np.pi, n_tasks, endpoint=False).tolist()
    angles_closed = angles + [angles[0]]
    retention_closed = retention_ratios + [retention_ratios[0]]

    try:
        _fs_radar = get_figure_size('double', aspect_ratio=1.0)
    except NameError:
        _fs_radar = (7.0, 7.0)
    fig, ax = plt.subplots(figsize=_fs_radar, subplot_kw=dict(polar=True))

    # 100% reference ring — prominent solid line (key visual anchor)
    ax.plot(angles_closed, [100] * (n_tasks + 1),
            color=COLOR_COMPUTER, linestyle='-', linewidth=2, alpha=1, label='64-bit Computer (100%)')
    # 80% reference ring — thin dashed
    ax.plot(angles_closed, [80] * (n_tasks + 1),
            color='#9E9E9E', linestyle='--', linewidth=1.0, alpha=0.7, label='Threshold (80%)')

    # Filled polygon for PNN retention
    ax.fill(angles, retention_ratios, alpha=0.2, color=COLOR_PNN)
    ax.plot(angles_closed, retention_closed,
            color=COLOR_PNN, linewidth=2, linestyle='-', marker='o', markersize=3, label='7-bit PNN')


    # Annotate each vertex — offset radially inward to avoid label-edge overlap
    for angle, val in zip(angles, retention_ratios):
        ax.annotate(f'{val:.1f}%',
                    xy=(angle, val+6),
                    xytext=(0, 0), textcoords='offset points',
                    ha='center', va='top',
                    fontsize=6, fontweight='bold', color='#1565C0',
                    bbox=dict(boxstyle='round,pad=0.15', facecolor='white', alpha=0.1, edgecolor='none'))


    # Task labels — horizontal orientation (perpendicular to radial)
    ax.set_xticks(angles)
    ax.set_xticklabels(task_names, fontsize=7)
    # 将标签放在最大半径之外
    ax.tick_params(axis='x', pad=9)


    # Radial axis range and ticks
    max_val = max(max(retention_ratios) + 15, 115)
    ax.set_ylim(0, max_val)
    ax.set_yticks([20, 40, 60, 80, 100])
    ax.set_yticklabels(['20%', '40%', '60%', '80%', '100%'], fontsize=6)
    ax.legend(loc='upper right', bbox_to_anchor=(0.95, 1.05), fontsize=7, framealpha=0.9)


    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, 'all_tasks_radar.png'), dpi=_PLOT_DPI)
    plt.close(fig)



def _generate_text_report(combined_results, hw_results, output_dir):
    """生成文本报告"""
    with open(os.path.join(output_dir, 'summary_report.txt'), 'w', encoding='utf-8') as f:
        f.write("=" * 65 + "\n")
        f.write("  PNN COMPREHENSIVE EVALUATION – SUMMARY REPORT\n")
        f.write("=" * 65 + "\n\n")
        f.write(f"Hardware: Sb₂Se₃ phase-change device, {len(hardware_lut)} levels (7-bit)\n")
        f.write(f"LUT range: [{hardware_lut.min():.4f}, {hardware_lut.max():.4f}]\n\n")

        # 识别任务结果
        f.write("─" * 65 + "\n")
        f.write("  RECOGNITION TASKS\n")
        f.write("─" * 65 + "\n")
        f.write(f"{'Task':<25s}  {'Classes':>8s}  {'Computer':>10s}  {'PNN':>10s}  {'Drop':>8s}\n")
        f.write("-" * 65 + "\n")
        for r in combined_results:
            drop = (r['computer'] - r['pnn']) * 100
            f.write(f"{r['task']:<25s}  {r['num_classes']:>8d}  "
                    f"{r['computer'] * 100:>9.2f}%  {r['pnn'] * 100:>9.2f}%  {drop:>7.2f}%\n")

        f.write("\nKey Findings:\n")
        best = max(combined_results, key=lambda r: r['pnn'])
        worst = min(combined_results, key=lambda r: r['pnn'])
        f.write(f"  Best PNN performance:  {best['task']} ({best['pnn'] * 100:.2f}%)\n")
        f.write(f"  Most challenging task:   {worst['task']} ({worst['pnn'] * 100:.2f}%)\n")
        avg_drop = np.mean([(r['computer'] - r['pnn']) * 100 for r in combined_results])
        f.write(f"  Average PTQ accuracy drop: {avg_drop:.2f}%\n\n")

        # 硬件分析摘要
        if hw_results:
            f.write("─" * 65 + "\n")
            f.write("  HARDWARE ANALYSIS EXPERIMENTS\n")
            f.write("─" * 65 + "\n")

            if 'bit_sensitivity' in hw_results:
                bs = hw_results['bit_sensitivity']
                f.write(f"\nBit-Width Sensitivity (MNIST baseline: {bs['baseline_acc'] * 100:.2f}%):\n")
                for r in bs['results']:
                    drop = (bs['baseline_acc'] - r['accuracy']) * 100
                    f.write(f"  {r['bits']}-bit ({r['levels']:3d} levels): "
                            f"{r['accuracy'] * 100:.2f}%  (drop: {drop:.2f}%)\n")

            if 'noise_robustness' in hw_results:
                nr = hw_results['noise_robustness']
                clean_acc_str = f"{nr['clean_acc'] * 100:.2f}%" if nr['clean_acc'] is not None else "N/A"
                f.write(f"\nNoise Robustness (clean PNN: {clean_acc_str}):\n")
                for r in nr['results']:
                    f.write(f"  σ = {r['sigma'] * 100:5.1f}%: "
                            f"{r['mean'] * 100:.2f}% ± {r['std'] * 100:.2f}%\n")

            if 'lut_comparison' in hw_results:
                lc = hw_results['lut_comparison']['results']
                f.write("\nUniform vs Non-Uniform LUT:\n")
                for ds_name, r in lc.items():
                    f.write(f"  {ds_name}: Computer {r['computer'] * 100:.2f}%  "
                            f"Real LUT {r['real_lut'] * 100:.2f}%  "
                            f"Uniform LUT {r['uniform_lut'] * 100:.2f}%\n")


def _generate_json_results(combined_results, output_dir):
    """生成JSON格式结果"""
    json_results = [
        {
            'task': r['task'],
            'computer_accuracy': round(r['computer'] * 100, 2),
            'pnn_accuracy': round(r['pnn'] * 100, 2),
            'accuracy_drop': round((r['computer'] - r['pnn']) * 100, 2),
        }
        for r in combined_results
    ]
    with open(os.path.join(output_dir, 'results.json'), 'w', encoding='utf-8') as f:
        json.dump(json_results, f, indent=2)


def _generate_hw_summary_figure(hw_results, output_dir):
    """生成三个独立的硬件分析图（与 PNN_Extended_Tasks.py 格式一致）"""
    try:
        _fs_hw = get_figure_size('double', aspect_ratio=0.6)
    except NameError:
        _fs_hw = (7.0, 4.2)

    # ── 图1: bit-width sensitivity ───────────────────────────────────────────
    if 'bit_sensitivity' in hw_results:
        bs = hw_results['bit_sensitivity']
        bits_arr = [r['bits'] for r in bs['results']]
        accs_arr = [r['accuracy'] * 100 for r in bs['results']]

        fig, ax = plt.subplots(figsize=_fs_hw)
        ax.plot(bits_arr, accs_arr, 'bo-', ms=3, lw=2, label='PNN')
        ax.axhline(y=bs['baseline_acc'] * 100, color=COLOR_COMPUTER, ls='--', lw=1.5,
                   label=f"64-bit Computer ({bs['baseline_acc'] * 100:.2f}%)")
        ax.set_xlabel('Bit Width', fontsize=8)
        ax.set_ylabel('Test Accuracy (%)', fontsize=8)
        ax.set_xticks(bits_arr)
        ax.set_xticklabels([f'{b}-bit\n({2**b} levels)' for b in bits_arr])
        ax.legend(fontsize=7); ax.grid(False)
        ax.set_ylim(0, max(accs_arr) + 10)
        for b, a in zip(bits_arr, accs_arr):
            ax.annotate(f'{a:.1f}%', (b, a), textcoords="offset points",
                        xytext=(0, -16), ha='center', fontsize=7, fontweight='bold')
        fig.tight_layout()
        fig.savefig(os.path.join(output_dir, 'bit_sensitivity.png'), dpi=_PLOT_DPI)
        plt.close(fig)

    # ── 图2: noise robustness ────────────────────────────────────────────────
    if 'noise_robustness' in hw_results:
        nr = hw_results['noise_robustness']
        sigmas = [r['sigma'] * 100 for r in nr['results']]
        means = [r['mean'] * 100 for r in nr['results']]
        stds = [r['std'] * 100 for r in nr['results']]

        fig, ax = plt.subplots(figsize=_fs_hw)
        ax.errorbar(sigmas, means, yerr=stds, fmt='o-', color=COLOR_PNN,
                    ecolor=COLOR_HIGHLIGHT, elinewidth=2, capsize=5, capthick=2,
                    ms=3, lw=2, label='7-bit PNN (mean ± std)')
        ax.axhline(y=nr['baseline_acc'] * 100, color=COLOR_COMPUTER, ls='--', lw=1.5,
                   label=f"64-bit Computer ({nr['baseline_acc'] * 100:.2f}%)")
        # if nr['clean_acc'] is not None:
        #     ax.axhline(y=nr['clean_acc'] * 100, color=COLOR_COMPUTER, ls=':', lw=1.5,
        #                label=f"PNN ({nr['clean_acc'] * 100:.2f}%)")
        ax.set_xlabel('Noise Level σ (% of weight range)', fontsize=8)
        ax.set_ylabel('Test Accuracy (%)', fontsize=8)
        ax.legend(fontsize=7); ax.grid(False)
        ax.set_ylim(50, 100)
        fig.tight_layout()
        fig.savefig(os.path.join(output_dir, 'noise_robustness.png'), dpi=_PLOT_DPI)
        plt.close(fig)

    # ── 图3: uniform vs non-uniform LUT ──────────────────────────────────────
    if 'lut_comparison' in hw_results:
        lc = hw_results['lut_comparison']['results']
        ds_names = list(lc.keys())
        x = np.arange(len(ds_names))
        w = 0.25
        comp_vals = [lc[d]['computer'] * 100 for d in ds_names]
        real_vals = [lc[d]['real_lut'] * 100 for d in ds_names]
        unif_vals = [lc[d]['uniform_lut'] * 100 for d in ds_names]

        fig, ax = plt.subplots(figsize=_fs_hw)
        bars1 = ax.bar(x - w, comp_vals, w, label='64-bit Computer', color=COLOR_COMPUTER)
        bars2 = ax.bar(x, real_vals, w, label='Real LUT (non-uniform)', color=COLOR_PNN)
        bars3 = ax.bar(x + w, unif_vals, w, label='Ideal LUT (uniform)', color=COLOR_UNIFORM)
        for bars in [bars1, bars2, bars3]:
            for bar in bars:
                h = bar.get_height()
                ax.annotate(f'{h:.1f}%',
                            xy=(bar.get_x() + bar.get_width() / 2, h),
                            xytext=(0, 4), textcoords="offset points",
                            ha='center', va='bottom', fontsize=7, fontweight='bold')
        ax.set_xticks(x); ax.set_xticklabels(ds_names, fontsize=8)
        ax.set_ylabel('Test Accuracy (%)', fontsize=8)
        ax.legend(fontsize=7); ax.grid(False)
        ax.set_ylim(0, 110)
        fig.tight_layout()
        fig.savefig(os.path.join(output_dir, 'lut_comparison.png'), dpi=_PLOT_DPI)
        plt.close(fig)

    # ── 合并三面板汇总图 ─────────────────────────────────────────────────────
    try:
        _fs_hw3 = get_figure_size('double', aspect_ratio=0.4)
    except NameError:
        _fs_hw3 = (7.0, 2.8)
    fig, axes = plt.subplots(1, 3, figsize=_fs_hw3)

    ax = axes[0]
    if 'bit_sensitivity' in hw_results:
        bs = hw_results['bit_sensitivity']
        bits_arr = [r['bits'] for r in bs['results']]
        accs_arr = [r['accuracy'] * 100 for r in bs['results']]
        ax.plot(bits_arr, accs_arr, 'bo-', ms=3, lw=2, label='PNN')
        ax.axhline(y=bs['baseline_acc'] * 100, color=COLOR_COMPUTER, ls='--', lw=1.5,
                   label=f"Computer ({bs['baseline_acc'] * 100:.1f}%)")

        ax.set_xticks(bits_arr)
        ax.set_xticklabels([f'{b}-bit' for b in bits_arr], fontsize=7)
        ax.set_xlabel('Bit Width', fontsize=8)
        ax.set_ylabel('Test Accuracy (%)', fontsize=8)
        ax.legend(fontsize=7); ax.grid(False)
        ax.set_ylim(0, max(accs_arr) + 5)
        # ax.set_ylim(0, 70)
        ax.legend(loc='lower right', bbox_to_anchor=(1.0, 0), fontsize=7)

    ax = axes[1]
    if 'noise_robustness' in hw_results:
        nr = hw_results['noise_robustness']
        sigmas = [r['sigma'] * 100 for r in nr['results']]
        means = [r['mean'] * 100 for r in nr['results']]
        stds = [r['std'] * 100 for r in nr['results']]
        ax.errorbar(sigmas, means, yerr=stds, fmt='o-', color=COLOR_PNN,
                    ecolor=COLOR_HIGHLIGHT, elinewidth=2, capsize=5, capthick=2,
                    ms=3, lw=2, label='PNN (mean ± std)')
        ax.axhline(y=nr['baseline_acc'] * 100, color=COLOR_COMPUTER, ls='--', lw=1.5,
                   label=f"Computer ({nr['baseline_acc'] * 100:.1f}%)")
        ax.legend(loc='lower right', bbox_to_anchor=(1.0, -0.15), fontsize=7)
        # if nr['clean_acc'] is not None:
        #     ax.axhline(y=nr['clean_acc'] * 100, color=COLOR_COMPUTER, ls=':', lw=1.5,
        #                label=f"Clean PNN ({nr['clean_acc'] * 100:.1f}%)")
        ax.set_xlabel('Noise Level σ (%)', fontsize=8)
        ax.set_ylabel('Test Accuracy (%)', fontsize=8)
        ax.legend(fontsize=7); ax.grid(False)
        ax.set_ylim(0, max(means) + 5)
        # ax.set_ylim(0, 90)
        ax.legend(loc='lower right', bbox_to_anchor=(1.0, 0), fontsize=7)

    ax = axes[2]
    if 'lut_comparison' in hw_results:
        lc = hw_results['lut_comparison']['results']
        ds_names = list(lc.keys())
        x = np.arange(len(ds_names))
        w = 0.25
        comp_vals = [lc[d]['computer'] * 100 for d in ds_names]
        real_vals = [lc[d]['real_lut'] * 100 for d in ds_names]
        unif_vals = [lc[d]['uniform_lut'] * 100 for d in ds_names]
        ax.bar(x - w, comp_vals, w, label='Computer', color=COLOR_COMPUTER)
        ax.bar(x, real_vals, w, label='Real LUT', color=COLOR_PNN)
        ax.bar(x + w, unif_vals, w, label='Uniform LUT', color=COLOR_UNIFORM)
        ax.set_xticks(x); ax.set_xticklabels(ds_names, fontsize=7)
        ax.set_ylabel('Test Accuracy (%)', fontsize=8)
        ax.legend(fontsize=7); ax.grid(False)
        ax.set_ylim(0, 115)

    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, 'hardware_analysis_summary.png'), dpi=_PLOT_DPI)
    plt.close(fig)


if __name__ == '__main__':
    print("\n" + "=" * 60)
    print("  从现有报告文件生成综合摘要")
    print("=" * 60 + "\n")
    generate_summary_from_files()