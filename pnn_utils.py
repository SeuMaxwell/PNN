"""
pnn_utils.py
=============
Shared utilities for PNN_Scene_Recognition.py and PNN_Extended_Tasks.py.

Provides common building blocks used by both scripts:
  - EPSILON constant
  - set_seed()            – fix random seeds for reproducibility
  - train_epoch()         – one MSE-loss training epoch
  - evaluate()            – accuracy + confusion matrix evaluation
  - _nearest_lut()        – chunked nearest-LUT-value mapping
  - apply_ptq_with_lut()  – post-training quantization with hardware LUT
  - plot_confusion_matrix() – seaborn confusion-matrix figure helper
"""

import copy
import random

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix

try:
    from plot_style import (
        apply_journal_style, get_figure_size, PRINT_DPI,
        COLOR_COMPUTER, COLOR_PNN, COLOR_HIGHLIGHT, COLOR_REFERENCE,
        CM_CMAP_COMPUTER, CM_CMAP_PNN,
    )
    apply_journal_style()
    _PLOT_DPI = PRINT_DPI
    _CM_CMAP_COMPUTER = CM_CMAP_COMPUTER
    _CM_CMAP_PNN      = CM_CMAP_PNN
except ImportError:
    _PLOT_DPI = 150
    COLOR_COMPUTER, COLOR_PNN = '#4CAF50', '#2196F3'
    COLOR_HIGHLIGHT, COLOR_REFERENCE = '#FF5722', '#9E9E9E'
    _CM_CMAP_COMPUTER, _CM_CMAP_PNN = 'Greens', 'Blues'

# Tiny value added to denominators to prevent division by zero
EPSILON = 1e-9


def set_seed(seed=42):
    """Fix random seeds for reproducibility across Python, NumPy, and PyTorch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def train_epoch(model, device, loader, optimizer, criterion):
    """One training epoch with MSE loss + one-hot targets.

    Uses ``model.num_classes`` to determine the one-hot encoding width, so
    every model passed here must expose a ``num_classes`` attribute.
    """
    model.train()
    total_loss = 0.0
    for data, target in loader:
        data, target = data.to(device), target.to(device)
        target_onehot = F.one_hot(target.long(), num_classes=model.num_classes).float()
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target_onehot)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


def evaluate(model, device, loader, compute_details=False):
    """Return (accuracy, confusion_matrix_or_None, per_class_acc_or_None)."""
    model.eval()
    correct = 0
    all_preds, all_targets = [], []
    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            pred = model(data).argmax(dim=1)
            correct += pred.eq(target).sum().item()
            if compute_details:
                all_preds.extend(pred.cpu().numpy())
                all_targets.extend(target.cpu().numpy())

    acc = correct / len(loader.dataset)
    if not compute_details:
        return acc, None, None

    cm = confusion_matrix(all_targets, all_preds)
    per_class = cm.diagonal() / (cm.sum(axis=1) + EPSILON)
    return acc, cm, per_class


def _nearest_lut(tensor_np, lut_np):
    """Map every value in *tensor_np* to its nearest entry in *lut_np*.

    Processes the tensor in chunks to avoid allocating a large intermediate
    broadcast array, which could cause out-of-memory errors for big weight
    tensors.
    """
    flat = tensor_np.flatten()
    result = np.empty_like(flat)
    chunk_size = 100_000
    for i in range(0, len(flat), chunk_size):
        chunk = flat[i:i + chunk_size]
        diff = np.abs(chunk[:, np.newaxis] - lut_np[np.newaxis, :])
        indices = np.argmin(diff, axis=-1)
        result[i:i + chunk_size] = lut_np[indices]
    return result.reshape(tensor_np.shape)


def apply_ptq_with_lut(model, lut, verbose=True):
    """Layer-wise dynamic-range-matching post-training quantization.

    Algorithm (per layer):
      1. w_max  = max(|weights|)
      2. lut_max = max(|lut|)
      3. scale  = w_max / lut_max
      4. w_norm = weights / scale
      5. Quantize w_norm → nearest LUT value
      6. weights = quantized_norm × scale   (restore amplitude)

    This simulates programming the trained weights onto the Sb₂Se₃ photonic
    chip using the real measured LUT from LUT.py.

    Args:
        model:   PyTorch model to quantize (not modified in-place).
        lut:     1-D numpy array of hardware LUT values.
        verbose: Print per-layer statistics when True.

    Returns:
        Deep-copied, quantized model.
    """
    q_model = copy.deepcopy(model)
    lut_np = lut.astype(np.float64)
    lut_max = float(np.max(np.abs(lut_np)))

    if verbose:
        print("Applying PTQ with hardware LUT …")
    for name, param in q_model.named_parameters():
        if param.dim() == 0:
            continue
        w_np = param.data.cpu().numpy().astype(np.float64)
        w_max = float(np.max(np.abs(w_np)))
        if w_max > 1e-9:
            scale = w_max / lut_max
            w_norm = w_np / scale
            w_q_norm = _nearest_lut(w_norm, lut_np)
            w_q = (w_q_norm * scale).astype(np.float32)
        else:
            w_q = w_np.astype(np.float32)
            scale = 0.0
        param.data = torch.from_numpy(w_q).to(param.device)
        if verbose:
            print(f"  {name:30s} | w_max={w_max:.4f} | scale={scale:.4f}")
    if verbose:
        print("Quantization complete.\n")
    return q_model


def plot_confusion_matrix(cm, class_names, title, save_path, cmap=None):
    """Save a seaborn confusion-matrix heatmap to *save_path*."""
    if cmap is None:
        cmap = _CM_CMAP_PNN
    try:
        figsize = get_figure_size('double', aspect_ratio=0.75)
    except NameError:
        figsize = (7.0, 5.25)
    fig, ax = plt.subplots(figsize=figsize)
    annot = len(class_names) <= 12
    sns.heatmap(cm, annot=annot, fmt='d', cmap=cmap,
                xticklabels=class_names, yticklabels=class_names,
                ax=ax, linewidths=0.3)
    ax.set_xlabel('Predicted Label', fontsize=8)
    ax.set_ylabel('True Label', fontsize=8)
    # x-axis labels horizontal; ticks outward on bottom/left only
    plt.xticks(rotation=0, ha='center', fontsize=7)
    plt.yticks(rotation=0, fontsize=7)
    ax.tick_params(axis='x', which='both', direction='out',
                   bottom=True, top=False, labelbottom=True, labeltop=False)
    ax.tick_params(axis='y', which='both', direction='out',
                   left=True, right=False, labelleft=True, labelright=False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    fig.tight_layout()
    fig.savefig(save_path, dpi=_PLOT_DPI)
    plt.close(fig)
