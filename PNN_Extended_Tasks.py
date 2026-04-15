"""
PNN_Extended_Tasks.py
$env:PYTHONIOENCODING="utf-8"
======================
Extended scene recognition tasks and hardware analysis experiments for the
PNN (Nonvolatile Photonic Neural Network) system, building on the verified
PNN_Scene_Recognition.py framework.

New Tasks:
  1. CIFAR-10 natural image recognition
  2. MedMNIST / PathMNIST medical-image classification (9 tissue classes)
  3. EMNIST-Letters handwritten letter recognition (26 classes)
  4. CIFAR-100 coarse-grained recognition (20 superclasses)

Hardware Analysis:
  5. Quantization bit-width sensitivity analysis
  6. Manufacturing noise robustness analysis
  7. Uniform vs non-uniform LUT comparison

Usage:
  python PNN_Extended_Tasks.py --task all             # run everything
  python PNN_Extended_Tasks.py --task new_tasks       # all 4 new tasks
  python PNN_Extended_Tasks.py --task hw_analysis     # all 3 hardware analyses
  python PNN_Extended_Tasks.py --task cifar10         # CIFAR-10 only
  python PNN_Extended_Tasks.py --task medmnist        # PathMNIST only
  python PNN_Extended_Tasks.py --task emnist          # EMNIST Letters only
  python PNN_Extended_Tasks.py --task cifar100        # CIFAR-100 coarse only
  python PNN_Extended_Tasks.py --task bit_analysis    # bit-width sensitivity
  python PNN_Extended_Tasks.py --task noise_analysis  # noise robustness
  python PNN_Extended_Tasks.py --task lut_comparison  # uniform vs non-uniform LUT
"""

import argparse
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

try:
    from plot_style import (
        apply_journal_style, get_figure_size, PRINT_DPI,
        COLOR_COMPUTER, COLOR_PNN, COLOR_HIGHLIGHT, COLOR_REFERENCE, COLOR_UNIFORM,
    )
    apply_journal_style()
    _PLOT_DPI = PRINT_DPI
except ImportError:
    _PLOT_DPI = 300
    COLOR_COMPUTER, COLOR_PNN = '#4CAF50', '#2196F3'
    COLOR_HIGHLIGHT, COLOR_REFERENCE, COLOR_UNIFORM = '#FF5722', '#9E9E9E', '#FF9800'

# Real measured Sb2Se3 hardware LUT (DO NOT MODIFY LUT.py)
from LUT import lut as hardware_lut

# Shared utilities (training, evaluation, PTQ, plotting, seed)
from pnn_utils import (EPSILON, set_seed, train_epoch, evaluate,
                         apply_ptq_with_lut, plot_confusion_matrix)

# ─────────────────────────────────────────────────────────────────────────────
# 1.  MODEL DEFINITIONS
# ────────────────────────────────────────────────���────────────────────────────

class FlexibleCNN2D(nn.Module):
    """
    Flexible 2-D CNN following the PNN paper architecture.
    Supports variable num_classes and input channels.
      Conv2d(in_ch, 32, 3, pad=1) → ReLU → MaxPool(2,2)
      Conv2d(32, 64, 3, pad=1)    → ReLU → MaxPool(2,2)
      Linear(64*7*7, 128) → ReLU
      Linear(128, num_classes) → Softmax
    """

    def __init__(self, num_classes=10, in_channels=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, num_classes)
        self.num_classes = num_classes

    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.softmax(x, dim=1)


# ─────────────────────────────────────────────────────────────────────────────
# 2.  DATA LOADERS
# ─────────────────────────────────────────────────────────────────────────────

def get_cifar10_loaders(batch_size=64):
    """CIFAR-10: RGB 32×32 → grayscale 28×28."""
    print("Loading CIFAR-10 (converting RGB→grayscale, resizing to 28×28)...")
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize((0.4809,), (0.2370,)),
    ])
    train_ds = datasets.CIFAR10('./data', train=True, download=True, transform=transform)
    test_ds = datasets.CIFAR10('./data', train=False, download=True, transform=transform)
    return (DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0),
            DataLoader(test_ds, batch_size=1000, shuffle=False, num_workers=0))


def get_medmnist_loaders(batch_size=64):
    """PathMNIST: RGB 28×28 colon-pathology images → grayscale, 9 tissue classes."""
    print("Loading PathMNIST (converting RGB→grayscale, 9 tissue classes)...")
    try:
        from medmnist import PathMNIST
    except ImportError:
        raise ImportError(
            "medmnist library is required. Install with: pip install medmnist"
        )
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ])
    # PathMNIST returns labels as numpy arrays of shape (1,); extract scalar
    # for compatibility with PyTorch's one-hot encoding and DataLoader collation.
    target_transform = transforms.Lambda(lambda y: int(np.asarray(y).ravel()[0]))
    train_ds = PathMNIST(split='train', transform=transform,
                         target_transform=target_transform,
                         download=True, root='./data')
    test_ds = PathMNIST(split='test', transform=transform,
                        target_transform=target_transform,
                        download=True, root='./data')
    return (DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0),
            DataLoader(test_ds, batch_size=1000, shuffle=False, num_workers=0))


def get_emnist_letters_loaders(batch_size=64):
    """EMNIST-Letters: 28×28 grayscale, 26 classes (a-z)."""
    print("Loading EMNIST-Letters (26 classes)...")
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1722,), (0.3310,)),
    ])
    # EMNIST Letters labels are 1-26, need to shift to 0-25
    train_ds = datasets.EMNIST('./data', split='letters', train=True,
                               download=True, transform=transform)
    test_ds = datasets.EMNIST('./data', split='letters', train=False,
                              download=True, transform=transform)
    # Shift labels from 1-26 to 0-25
    train_ds.targets = train_ds.targets - 1
    test_ds.targets = test_ds.targets - 1
    return (DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0),
            DataLoader(test_ds, batch_size=1000, shuffle=False, num_workers=0))


def get_cifar100_coarse_loaders(batch_size=64):
    """CIFAR-100 with 20 superclass labels: RGB 32×32 → grayscale 28×28."""
    print("Loading CIFAR-100 coarse (20 superclasses, RGB→grayscale 28×28)...")
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize((0.4809,), (0.2370,)),
    ])
    train_ds = datasets.CIFAR100('./data', train=True, download=True, transform=transform)
    test_ds = datasets.CIFAR100('./data', train=False, download=True, transform=transform)
    # Use coarse labels (20 superclasses)
    train_ds.targets = train_ds.coarse_targets if hasattr(train_ds, 'coarse_targets') else _get_coarse_targets(train_ds.targets)
    test_ds.targets = test_ds.coarse_targets if hasattr(test_ds, 'coarse_targets') else _get_coarse_targets(test_ds.targets)
    return (DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0),
            DataLoader(test_ds, batch_size=1000, shuffle=False, num_workers=0))


# CIFAR-100 fine-to-coarse mapping (standard 20 superclasses)
_CIFAR100_COARSE_MAP = [
    4, 1, 14, 8, 0, 6, 7, 7, 18, 3, 3, 14, 9, 18, 7, 11, 3, 9, 7, 11,
    6, 11, 5, 10, 7, 6, 13, 15, 3, 15, 0, 11, 1, 10, 12, 14, 16, 9, 11, 5,
    5, 19, 8, 8, 15, 13, 14, 17, 18, 10, 16, 4, 17, 4, 2, 0, 17, 4, 18, 17,
    10, 3, 2, 12, 12, 16, 12, 1, 9, 19, 2, 10, 0, 1, 16, 12, 9, 13, 15, 13,
    16, 19, 2, 4, 6, 19, 5, 5, 8, 19, 18, 1, 2, 15, 6, 0, 17, 8, 14, 13,
]

def _get_coarse_targets(fine_targets):
    """Convert fine labels (0-99) to coarse labels (0-19)."""
    return [_CIFAR100_COARSE_MAP[t] for t in fine_targets]


# ─────────────────────────────────────────────────────────────────────────────
# 3.  VISUALIZATION AND REPORTING
# ─────────────────────────────────────────────────────────────────────────────

CIFAR10_CLASSES = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                   'dog', 'frog', 'horse', 'ship', 'truck']
MEDMNIST_CLASSES = ['ADI', 'BACK', 'DEB', 'LYM', 'MUC', 'MUS', 'NORM', 'STR', 'TUM']
EMNIST_CLASSES = [chr(ord('A') + i) for i in range(26)]
CIFAR100_COARSE_CLASSES = [
    'aquatic mammals', 'fish', 'flowers', 'food containers', 'fruit/veg',
    'household electrical', 'household furniture', 'insects', 'large carnivores',
    'large outdoor man-made', 'large outdoor natural', 'large omnivores/herbivores',
    'medium mammals', 'non-insect invertebrates', 'people', 'reptiles',
    'small mammals', 'trees', 'vehicles 1', 'vehicles 2',
]


def _get_class_names(task_name):
    mapping = {
        'cifar10': CIFAR10_CLASSES,
        'medmnist': MEDMNIST_CLASSES,
        'emnist_letters': EMNIST_CLASSES,
        'cifar100_coarse': [f"SC{i}" for i in range(20)],  # short labels for plot
    }
    return mapping.get(task_name, [str(i) for i in range(10)])


def generate_task_results(output_dir, history, task_name):
    """Generate all visualizations and report for a recognition task."""
    os.makedirs(output_dir, exist_ok=True)
    epochs = range(1, len(history['train_losses']) + 1)
    title_name = task_name.replace('_', ' ').title()
    class_names = _get_class_names(task_name)

    # 1 & 2. Combined dual-axis figure: left Y = Accuracy, right Y = Loss
    try:
        _fs_comb = get_figure_size('double', aspect_ratio=0.5)
    except NameError:
        _fs_comb = (7.0, 3.5)
    fig, ax_acc = plt.subplots(figsize=_fs_comb)
    ax_loss = ax_acc.twinx()
    ln1 = ax_acc.plot(epochs, [a * 100 for a in history['test_accuracies']],
                      '-', color=COLOR_COMPUTER, ms=0, lw=1.5,
                      label='Test Accuracy')
    ln2 = ax_loss.plot(epochs, history['train_losses'],
                       '-', color=COLOR_PNN, ms=0, lw=1.5,
                       label='Training Loss')
    ax_acc.set_xlabel('Epoch', fontsize=8)
    ax_acc.set_ylabel('Accuracy (%)', fontsize=8)
    ax_loss.set_ylabel('Loss', fontsize=8)
    ax_acc.grid(False); ax_loss.grid(False)
    lns = ln1 + ln2
    ax_acc.set_ylim(0, max(history['test_accuracies']) * 100 + 10)
    ax_acc.legend(lns, [l.get_label() for l in lns], fontsize=7, loc='center right',
                  framealpha=0.9, facecolor='white')
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, 'loss_curve and accuracy_curve.png'), dpi=_PLOT_DPI)
    plt.close(fig)

    # 3 & 4. Confusion matrices
    plot_confusion_matrix(
        history['computer_cm'], class_names,
        f'Confusion Matrix – {title_name} (64-bit Computer)',
        os.path.join(output_dir, 'cm_computer.png'), cmap='Greens')
    plot_confusion_matrix(
        history['pnn_cm'], class_names,
        f'Confusion Matrix – {title_name} (PNN with LUT)',
        os.path.join(output_dir, 'cm_pnn.png'), cmap='Blues')

    # 5. Per-class accuracy comparison
    try:
        _fs_bar = get_figure_size('double', aspect_ratio=0.5)
    except NameError:
        _fs_bar = (7.0, 3.5)
    fig, ax = plt.subplots(figsize=_fs_bar)
    x = np.arange(len(class_names))
    w = 0.35
    ax.bar(x - w/2, history['computer_per_class'] * 100, w,
           label='64-bit Computer', color=COLOR_COMPUTER)
    ax.bar(x + w/2, history['pnn_per_class'] * 100, w,
           label='PNN (LUT)', color=COLOR_PNN)
    ax.set_xticks(x); ax.set_xticklabels(class_names, rotation=45, ha='right', fontsize=7)
    ax.set_ylabel('Accuracy (%)', fontsize=8); ax.set_ylim(0, 110)
    ax.legend(); ax.grid(False)
    fig.tight_layout(); fig.savefig(os.path.join(output_dir, 'per_class_accuracy.png'), dpi=_PLOT_DPI)
    plt.close(fig)

    # 6. Text report
    report_path = os.path.join(output_dir, 'report.txt')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(f"=== PNN Extended Task: {title_name} ===\n\n")
        f.write(f"Number of classes:  {len(class_names)}\n")
        f.write(f"Training epochs:    {len(history['train_losses'])}\n\n")
        f.write(f"Full Precision (64-bit Computer):  {history['computer_acc'] * 100:.2f}%\n")
        f.write(f"PNN Simulation (LUT Quantized):  {history['pnn_acc'] * 100:.2f}%\n")
        f.write(f"Accuracy drop after PTQ:           "
                f"{(history['computer_acc'] - history['pnn_acc']) * 100:.2f}%\n\n")
        f.write("Per-Class Accuracy:\n")
        f.write(f"{'Class':>20s}  {'Computer':>10s}  {'PNN':>10s}  {'Drop':>8s}\n")
        f.write("-" * 55 + "\n")
        for i, cls in enumerate(class_names):
            c = history['computer_per_class'][i] * 100
            n = history['pnn_per_class'][i] * 100
            f.write(f"{cls:>20s}  {c:>9.2f}%  {n:>9.2f}%  {c - n:>7.2f}%\n")

    print(f"  ✓ Results → '{output_dir}/'")
    print(f"    Computer: {history['computer_acc'] * 100:.2f}%  |  "
          f"PNN: {history['pnn_acc'] * 100:.2f}%")


# ─────────────────────────────────────────────────────────────────────────────
# 6.  TASK RUNNERS
# ─────────────────────────────────────────────────────────────────────────────

def run_recognition_task(task_name, loader_fn, num_classes, num_epochs,
                         output_dir, batch_size=64):
    """Generic runner for any image classification task."""
    print(f"\n{'=' * 60}")
    print(f"  TASK: {task_name.upper().replace('_', ' ')}")
    print(f"{'=' * 60}\n")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"  Device: {device}")

    train_loader, test_loader = loader_fn(batch_size)

    model = FlexibleCNN2D(num_classes=num_classes).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    history = {'train_losses': [], 'test_accuracies': []}

    # Stage 1: Full-precision training
    print(f"\n  [Stage 1] Training for {num_epochs} epochs …")
    for epoch in range(1, num_epochs + 1):
        loss = train_epoch(model, device, train_loader, optimizer, criterion)
        acc, _, _ = evaluate(model, device, test_loader)
        history['train_losses'].append(loss)
        history['test_accuracies'].append(acc)
        if epoch % 5 == 0 or epoch == num_epochs:
            print(f"    Epoch {epoch:3d}/{num_epochs} | Loss: {loss:.4f} | Acc: {acc * 100:.2f}%")

    computer_acc, computer_cm, computer_per_class = evaluate(
        model, device, test_loader, compute_details=True)
    print(f"\n  → Full Precision: {computer_acc * 100:.2f}%")

    # Stage 2: PTQ with hardware LUT
    print("\n  [Stage 2] PTQ with Sb2Se3 LUT …")
    q_model = apply_ptq_with_lut(model, hardware_lut.astype(np.float32))
    pnn_acc, pnn_cm, pnn_per_class = evaluate(
        q_model, device, test_loader, compute_details=True)
    print(f"  → PNN (LUT): {pnn_acc * 100:.2f}%")

    history.update({
        'computer_acc': computer_acc,
        'computer_cm': computer_cm,
        'computer_per_class': computer_per_class,
        'pnn_acc': pnn_acc,
        'pnn_cm': pnn_cm,
        'pnn_per_class': pnn_per_class,
    })
    generate_task_results(output_dir, history, task_name)
    return {'task': task_name, 'computer': computer_acc, 'pnn': pnn_acc}


# ─────────────────────────────────────────────────────────────────────────────
# 7.  HARDWARE ANALYSIS EXPERIMENTS
# ─────────────────────────────────────────────────────────────────────────────

def _train_mnist_baseline(device, num_epochs=15):
    """Train a baseline MNIST model for hardware analysis experiments."""
    print("  Training baseline MNIST model …")
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    train_ds = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_ds = datasets.MNIST('./data', train=False, download=True, transform=transform)
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=1000, shuffle=False, num_workers=0)

    model = FlexibleCNN2D(num_classes=10).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    for epoch in range(1, num_epochs + 1):
        train_epoch(model, device, train_loader, optimizer, criterion)
        if epoch % 5 == 0:
            acc, _, _ = evaluate(model, device, test_loader)
            print(f"    Epoch {epoch}/{num_epochs} → Acc: {acc * 100:.2f}%")

    final_acc, _, _ = evaluate(model, device, test_loader)
    print(f"  Baseline model ready: {final_acc * 100:.2f}%\n")
    return model, test_loader, final_acc


def _train_cifar10_baseline(device, num_epochs=25):
    """Train a baseline CIFAR-10 (grayscale 28×28) model for bit-sensitivity analysis."""
    print("  Training baseline CIFAR-10 (grayscale 28×28) model …")
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize((0.4809,), (0.2370,)),
    ])
    train_ds = datasets.CIFAR10('./data', train=True, download=True, transform=transform)
    test_ds = datasets.CIFAR10('./data', train=False, download=True, transform=transform)
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=1000, shuffle=False, num_workers=0)

    model = FlexibleCNN2D(num_classes=10).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    for epoch in range(1, num_epochs + 1):
        train_epoch(model, device, train_loader, optimizer, criterion)
        if epoch % 5 == 0:
            acc, _, _ = evaluate(model, device, test_loader)
            print(f"    Epoch {epoch}/{num_epochs} → Acc: {acc * 100:.2f}%")

    final_acc, _, _ = evaluate(model, device, test_loader)
    print(f"  Baseline model ready: {final_acc * 100:.2f}%\n")
    return model, test_loader, final_acc


def _train_emnist_baseline(device, num_epochs=20):
    """Train a baseline EMNIST-Letters (26 classes) model for noise-robustness analysis."""
    print("  Training baseline EMNIST-Letters (26 classes) model …")
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1722,), (0.3310,)),
    ])
    train_ds = datasets.EMNIST('./data', split='letters', train=True,
                               download=True, transform=transform)
    test_ds = datasets.EMNIST('./data', split='letters', train=False,
                              download=True, transform=transform)
    # Shift labels from 1-26 to 0-25
    train_ds.targets = train_ds.targets - 1
    test_ds.targets = test_ds.targets - 1
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=1000, shuffle=False, num_workers=0)

    model = FlexibleCNN2D(num_classes=26).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    for epoch in range(1, num_epochs + 1):
        train_epoch(model, device, train_loader, optimizer, criterion)
        if epoch % 5 == 0:
            acc, _, _ = evaluate(model, device, test_loader)
            print(f"    Epoch {epoch}/{num_epochs} → Acc: {acc * 100:.2f}%")

    final_acc, _, _ = evaluate(model, device, test_loader)
    print(f"  Baseline model ready: {final_acc * 100:.2f}%\n")
    return model, test_loader, final_acc


def _load_medmnist_data(batch_size=64):
    """Load PathMNIST (MedMNIST) data for LUT comparison experiment."""
    try:
        from medmnist import PathMNIST
    except ImportError:
        raise ImportError(
            "medmnist library is required. Install with: pip install medmnist"
        )
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ])
    target_transform = transforms.Lambda(lambda y: int(np.asarray(y).ravel()[0]))
    train_ds = PathMNIST(split='train', transform=transform,
                         target_transform=target_transform,
                         download=True, root='./data')
    test_ds = PathMNIST(split='test', transform=transform,
                        target_transform=target_transform,
                        download=True, root='./data')
    return (DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0),
            DataLoader(test_ds, batch_size=1000, shuffle=False, num_workers=0))


def run_bit_sensitivity(output_dir='results_bit_sensitivity'):
    """
    Experiment 1: How does accuracy change with different bit-widths?
    Subsample the real 128-level LUT to simulate 2-bit through 7-bit.
    """
    print(f"\n{'=' * 60}")
    print(f"  EXPERIMENT: BIT-WIDTH SENSITIVITY ANALYSIS")
    print(f"{'=' * 60}\n")

    os.makedirs(output_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model, test_loader, baseline_acc = _train_cifar10_baseline(device)

    lut_full = hardware_lut.astype(np.float32)
    bit_widths = [7, 6, 5, 4, 3, 2]
    n_levels_list = [128, 64, 32, 16, 8, 4]
    results = []

    for bits, n_levels in zip(bit_widths, n_levels_list):
        # Subsample: pick evenly-spaced indices from real LUT
        if n_levels >= len(lut_full):
            sub_lut = lut_full
        else:
            indices = np.round(np.linspace(0, len(lut_full) - 1, n_levels)).astype(int)
            sub_lut = lut_full[indices]

        q_model = apply_ptq_with_lut(model, sub_lut, verbose=False)
        acc, _, _ = evaluate(q_model, device, test_loader)
        results.append({'bits': bits, 'levels': n_levels, 'accuracy': acc})
        print(f"  {bits}-bit ({n_levels:3d} levels): {acc * 100:.2f}%")

    # Plot
    try:
        _fs_hw = get_figure_size('double', aspect_ratio=0.6)
    except NameError:
        _fs_hw = (7.0, 4.2)
    fig, ax = plt.subplots(figsize=_fs_hw)
    bits_arr = [r['bits'] for r in results]
    accs_arr = [r['accuracy'] * 100 for r in results]
    # accs_arr = [66.52, 58.47, 52.89, 39.30, 13.14, 10.0]  # 自定义7-bit 到 2-bit 的准确率结果(%)
    ax.plot(bits_arr, accs_arr, 'bo-', ms=8, lw=2, label='PNN Accuracy')
    ax.axhline(y=baseline_acc * 100, color=COLOR_REFERENCE, ls='--', lw=1.5,
               label=f'64-bit Computer ({baseline_acc * 100:.2f}%)')
    ax.set_xlabel('Bit Width', fontsize=8)
    ax.set_ylabel('Test Accuracy (%)', fontsize=8)
    ax.set_xticks(bits_arr)
    ax.set_xticklabels([f'{b}-bit\n({2**b} levels)' for b in bits_arr])
    ax.legend(fontsize=7); ax.grid(False)

    # Annotate accuracy values
    for b, a in zip(bits_arr, accs_arr):
        ax.annotate(f'{a:.1f}%', (b, a), textcoords="offset points",
                    xytext=(0, 12), ha='center', fontsize=7, fontweight='bold')

    fig.tight_layout(); fig.savefig(os.path.join(output_dir, 'bit_sensitivity.png'), dpi=_PLOT_DPI)
    plt.close(fig)

    # Report
    with open(os.path.join(output_dir, 'report.txt'), 'w', encoding='utf-8') as f:
        f.write("=== Bit-Width Sensitivity Analysis (CIFAR-10) ===\n\n")
        f.write(f"Dataset: CIFAR-10 (grayscale 28×28)\n")
        f.write(f"Baseline (64-bit float): {baseline_acc * 100:.2f}%\n\n")
        f.write(f"{'Bits':>6s}  {'Levels':>8s}  {'Accuracy':>10s}  {'Drop':>8s}\n")
        f.write("-" * 40 + "\n")
        for r in results:
            drop = (baseline_acc - r['accuracy']) * 100
            f.write(f"{r['bits']:>6d}  {r['levels']:>8d}  "
                    f"{r['accuracy'] * 100:>9.2f}%  {drop:>7.2f}%\n")

    print(f"  ✓ Results → '{output_dir}/'")
    return {'results': results, 'baseline_acc': baseline_acc}


def run_noise_robustness(output_dir='results_noise_robustness'):
    """
    Experiment 2: How robust is the system to manufacturing noise?
    Add Gaussian noise to quantized weights and measure accuracy degradation.
    """
    print(f"\n{'=' * 60}")
    print(f"  EXPERIMENT: MANUFACTURING NOISE ROBUSTNESS")
    print(f"{'=' * 60}\n")

    os.makedirs(output_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model, test_loader, baseline_acc = _train_emnist_baseline(device)

    # First, quantize the model
    lut_fp32 = hardware_lut.astype(np.float32)
    q_model_clean = apply_ptq_with_lut(model, lut_fp32, verbose=False)
    clean_acc, _, _ = evaluate(q_model_clean, device, test_loader)
    print(f"  Clean PNN accuracy: {clean_acc * 100:.2f}%\n")

    noise_levels = [0.0, 0.01, 0.02, 0.03, 0.05, 0.08, 0.10]
    n_trials = 5
    results = []

    for sigma_rel in noise_levels:
        trial_accs = []
        for trial in range(n_trials):
            noisy_model = copy.deepcopy(q_model_clean)
            if sigma_rel > 0:
                for name, param in noisy_model.named_parameters():
                    if param.dim() == 0:
                        continue
                    w = param.data
                    w_range = w.max() - w.min()
                    noise = torch.randn_like(w) * (sigma_rel * w_range.item())
                    param.data = w + noise

            acc, _, _ = evaluate(noisy_model, device, test_loader)
            trial_accs.append(acc)

        mean_acc = np.mean(trial_accs)
        std_acc = np.std(trial_accs)
        results.append({
            'sigma': sigma_rel, 'mean': mean_acc, 'std': std_acc,
            'trials': trial_accs
        })
        print(f"  σ = {sigma_rel * 100:5.1f}%: {mean_acc * 100:.2f}% ± {std_acc * 100:.2f}%")

    # Plot
    try:
        _fs_hw = get_figure_size('double', aspect_ratio=0.6)
    except NameError:
        _fs_hw = (7.0, 4.2)
    fig, ax = plt.subplots(figsize=_fs_hw)
    sigmas = [r['sigma'] * 100 for r in results]
    means = [r['mean'] * 100 for r in results]
    stds = [r['std'] * 100 for r in results]

    ax.errorbar(sigmas, means, yerr=stds, fmt='o-', color=COLOR_PNN,
                ecolor=COLOR_HIGHLIGHT, elinewidth=2, capsize=5, capthick=2,
                ms=8, lw=2, label='PNN Accuracy (mean ± std)')
    ax.axhline(y=baseline_acc * 100, color=COLOR_REFERENCE, ls='--', lw=1.5,
               label=f'64-bit Computer ({baseline_acc * 100:.2f}%)')
    ax.axhline(y=clean_acc * 100, color=COLOR_COMPUTER, ls=':', lw=1.5,
               label=f'Clean PNN ({clean_acc * 100:.2f}%)')

    ax.set_xlabel('Noise Level σ (% of weight range)', fontsize=8)
    ax.set_ylabel('Test Accuracy (%)', fontsize=8)
    ax.legend(fontsize=7); ax.grid(False)
    fig.tight_layout(); fig.savefig(os.path.join(output_dir, 'noise_robustness.png'), dpi=_PLOT_DPI)
    plt.close(fig)

    # Report
    with open(os.path.join(output_dir, 'report.txt'), 'w', encoding='utf-8') as f:
        f.write("=== Manufacturing Noise Robustness Analysis (EMNIST-Letters 26 classes) ===\n\n")
        f.write(f"Dataset: EMNIST-Letters (26 classes)\n")
        f.write(f"Baseline (64-bit float): {baseline_acc * 100:.2f}%\n")
        f.write(f"Clean PNN (no noise):  {clean_acc * 100:.2f}%\n")
        f.write(f"Trials per noise level:  {n_trials}\n\n")
        f.write(f"{'σ (%)':>8s}  {'Mean Acc':>10s}  {'Std':>8s}  {'Drop from clean':>16s}\n")
        f.write("-" * 50 + "\n")
        for r in results:
            drop = (clean_acc - r['mean']) * 100
            f.write(f"{r['sigma'] * 100:>7.1f}%  {r['mean'] * 100:>9.2f}%  "
                    f"{r['std'] * 100:>7.2f}%  {drop:>15.2f}%\n")

    print(f"  ✓ Results → '{output_dir}/'")
    return {'results': results, 'baseline_acc': baseline_acc, 'clean_acc': clean_acc}


def run_lut_comparison(output_dir='results_lut_comparison'):
    """
    Experiment 3: Compare real non-uniform LUT vs ideal uniform LUT.
    Tests on Fashion-MNIST and MedMNIST (PathMNIST).
    """
    print(f"\n{'=' * 60}")
    print(f"  EXPERIMENT: UNIFORM vs NON-UNIFORM LUT COMPARISON")
    print(f"{'=' * 60}\n")

    os.makedirs(output_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    real_lut = hardware_lut.astype(np.float32)
    lut_min, lut_max = float(real_lut.min()), float(real_lut.max())
    uniform_lut = np.linspace(lut_min, lut_max, len(real_lut)).astype(np.float32)

    results = {}

    for ds_name in ['Fashion-MNIST', 'MedMNIST']:
        print(f"\n  --- {ds_name} ---")

        if ds_name == 'MedMNIST':
            num_classes = 9
            n_epochs = 20
            train_loader, test_loader = _load_medmnist_data(batch_size=64)
        else:
            num_classes = 10
            n_epochs = 20
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.2860,), (0.3530,)),
            ])
            train_ds = datasets.FashionMNIST('./data', train=True, download=True,
                                             transform=transform)
            test_ds = datasets.FashionMNIST('./data', train=False, download=True,
                                            transform=transform)
            train_loader = DataLoader(train_ds, batch_size=64, shuffle=True, num_workers=0)
            test_loader = DataLoader(test_ds, batch_size=1000, shuffle=False, num_workers=0)

        # Train
        model = FlexibleCNN2D(num_classes=num_classes).to(device)
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.MSELoss()

        for epoch in range(1, n_epochs + 1):
            train_epoch(model, device, train_loader, optimizer, criterion)
        computer_acc, _, _ = evaluate(model, device, test_loader)
        print(f"  Computer: {computer_acc * 100:.2f}%")

        # PTQ with real LUT
        q_real = apply_ptq_with_lut(model, real_lut, verbose=False)
        real_acc, _, _ = evaluate(q_real, device, test_loader)
        print(f"  Real LUT (non-uniform): {real_acc * 100:.2f}%")

        # PTQ with uniform LUT
        q_uniform = apply_ptq_with_lut(model, uniform_lut, verbose=False)
        uniform_acc, _, _ = evaluate(q_uniform, device, test_loader)
        print(f"  Ideal LUT (uniform):    {uniform_acc * 100:.2f}%")

        results[ds_name] = {
            'computer': computer_acc,
            'real_lut': real_acc,
            'uniform_lut': uniform_acc,
        }

    # Plot
    try:
        _fs_lut = get_figure_size('double', aspect_ratio=0.6)
    except NameError:
        _fs_lut = (7.0, 4.2)
    fig, ax = plt.subplots(figsize=_fs_lut)
    ds_names = list(results.keys())
    x = np.arange(len(ds_names))
    w = 0.25

    computer_accs = [results[d]['computer'] * 100 for d in ds_names]
    real_accs = [results[d]['real_lut'] * 100 for d in ds_names]
    uniform_accs = [results[d]['uniform_lut'] * 100 for d in ds_names]

    bars1 = ax.bar(x - w, computer_accs, w, label='64-bit Computer', color=COLOR_COMPUTER)
    bars2 = ax.bar(x, real_accs, w, label='Real LUT (non-uniform)', color=COLOR_PNN)
    bars3 = ax.bar(x + w, uniform_accs, w, label='Ideal LUT (uniform)', color=COLOR_UNIFORM)

    # Add value labels
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            h = bar.get_height()
            ax.annotate(f'{h:.1f}%', xy=(bar.get_x() + bar.get_width() / 2, h),
                        xytext=(0, 4), textcoords="offset points",
                        ha='center', va='bottom', fontsize=7, fontweight='bold')

    ax.set_xticks(x); ax.set_xticklabels(ds_names, fontsize=8)
    ax.set_ylabel('Test Accuracy (%)', fontsize=8)
    ax.legend(fontsize=7); ax.grid(False)
    ax.set_ylim(0, 110)
    fig.tight_layout(); fig.savefig(os.path.join(output_dir, 'lut_comparison.png'), dpi=_PLOT_DPI)
    plt.close(fig)

    # LUT distribution visualization
    try:
        _fs_dist = get_figure_size('double', aspect_ratio=0.5)
    except NameError:
        _fs_dist = (7.0, 3.5)
    fig, axes = plt.subplots(1, 2, figsize=_fs_dist)
    axes[0].hist(real_lut, bins=30, color=COLOR_PNN, alpha=0.7, edgecolor='black')
    axes[0].set_xlabel('γ value', fontsize=8); axes[0].set_ylabel('Count', fontsize=8)
    axes[0].grid(False)
    axes[1].hist(uniform_lut, bins=30, color=COLOR_UNIFORM, alpha=0.7, edgecolor='black')
    axes[1].set_xlabel('γ value', fontsize=8); axes[1].set_ylabel('Count', fontsize=8)
    axes[1].grid(False)
    fig.tight_layout(); fig.savefig(os.path.join(output_dir, 'lut_distributions.png'), dpi=_PLOT_DPI)
    plt.close(fig)

    # Report
    with open(os.path.join(output_dir, 'report.txt'), 'w', encoding='utf-8') as f:
        f.write("=== Uniform vs Non-Uniform LUT Comparison ===\n\n")
        f.write(f"Datasets: Fashion-MNIST, MedMNIST (PathMNIST)\n")
        f.write(f"Real LUT:    {len(real_lut)} levels, range [{lut_min:.4f}, {lut_max:.4f}]\n")
        f.write(f"Uniform LUT: {len(uniform_lut)} levels, range [{lut_min:.4f}, {lut_max:.4f}]\n\n")
        for ds_name, r in results.items():
            f.write(f"--- {ds_name} ---\n")
            f.write(f"  Computer:     {r['computer'] * 100:.2f}%\n")
            f.write(f"  Real LUT:     {r['real_lut'] * 100:.2f}%\n")
            f.write(f"  Uniform LUT:  {r['uniform_lut'] * 100:.2f}%\n")
            f.write(f"  Difference:   {(r['real_lut'] - r['uniform_lut']) * 100:+.2f}%\n\n")

    print(f"\n  ✓ Results → '{output_dir}/'")
    return {'results': results}


# ─────────────────────────────────────────────────────────────────────────────
# 8.  ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='PNN Extended Tasks & Hardware Analysis'
    )
    parser.add_argument('--task', type=str, default='all',
                        choices=['all', 'new_tasks', 'hw_analysis',
                                 'cifar10', 'medmnist', 'emnist', 'cifar100',
                                 'bit_analysis', 'noise_analysis', 'lut_comparison'],
                        help='Which task(s) to run')
    parser.add_argument('--epochs_cifar10', type=int, default=100)
    parser.add_argument('--epochs_medmnist', type=int, default=100)
    parser.add_argument('--epochs_emnist', type=int, default=100)
    parser.add_argument('--epochs_cifar100', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=64)
    args = parser.parse_args()

    set_seed(42)

    lut_np = hardware_lut.astype(np.float32)
    print(f"\n  Hardware LUT: {len(lut_np)} levels, "
          f"range [{lut_np.min():.4f}, {lut_np.max():.4f}]")

    all_results = []
    hw_results = {}
    task = args.task

    # ── New recognition tasks ────────────────────────────────────────────────

    if task in ('all', 'new_tasks', 'cifar10'):
        r = run_recognition_task('cifar10', get_cifar10_loaders, 10,
                                 args.epochs_cifar10, 'results_cifar10', args.batch_size)
        r['num_classes'] = 10
        all_results.append(r)

    if task in ('all', 'new_tasks', 'medmnist'):
        r = run_recognition_task('medmnist', get_medmnist_loaders, 9,
                                 args.epochs_medmnist, 'results_medmnist', args.batch_size)
        r['num_classes'] = 9
        all_results.append(r)

    if task in ('all', 'new_tasks', 'emnist'):
        r = run_recognition_task('emnist_letters', get_emnist_letters_loaders, 26,
                                 args.epochs_emnist, 'results_emnist_letters', args.batch_size)
        r['num_classes'] = 26
        all_results.append(r)

    if task in ('all', 'new_tasks', 'cifar100'):
        r = run_recognition_task('cifar100_coarse', get_cifar100_coarse_loaders, 20,
                                 args.epochs_cifar100, 'results_cifar100_coarse', args.batch_size)
        r['num_classes'] = 20
        all_results.append(r)

    # ── Hardware analysis experiments ────────────────────────────────────────

    if task in ('all', 'hw_analysis', 'bit_analysis'):
        hw_results['bit_sensitivity'] = run_bit_sensitivity()

    if task in ('all', 'hw_analysis', 'noise_analysis'):
        hw_results['noise_robustness'] = run_noise_robustness()

    if task in ('all', 'hw_analysis', 'lut_comparison'):
        hw_results['lut_comparison'] = run_lut_comparison()

    # ── Summary dashboard (delegated to Load_report_to_abstract.py) ─────────

    if all_results or hw_results:
        from Load_report_to_abstract import generate_summary_from_files
        generate_summary_from_files()

    print("\n\n  ✓ All requested tasks completed.")


if __name__ == '__main__':
    main()

