"""
Visual_dataset.py
==================
Visualize CIFAR-10 and MedMNIST (PathMNIST) raw sample images.
Save each image as a separate PNG file.

Usage:
  python Visual_dataset.py --dataset cifar10
  python Visual_dataset.py --dataset medmnist
  python Visual_dataset.py --dataset both
"""

import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

from torchvision import datasets

# Dataset class names
CIFAR10_CLASSES = [
    'airplane', 'automobile', 'bird', 'cat', 'deer',
    'dog', 'frog', 'horse', 'ship', 'truck'
]

MEDMNIST_CLASSES = [
    'ADI', 'BACK', 'DEB', 'LYM', 'MUC', 'MUS', 'NORM', 'STR', 'TUM'
]

# Plot settings
PLOT_DPI = 300


def save_cifar10_samples(output_dir='visualizations', num_samples=10):
    """Save CIFAR-10 sample images as individual PNG files."""
    output_path = os.path.join(output_dir, 'cifar10')
    os.makedirs(output_path, exist_ok=True)

    print("Loading CIFAR-10 dataset...")
    dataset = datasets.CIFAR10('./data', train=False, download=True, transform=None)

    np.random.seed(42)
    indices = np.random.choice(len(dataset), num_samples, replace=False)

    for i, idx in enumerate(indices):
        img, label = dataset[idx]
        fig, ax = plt.subplots(figsize=(2, 2))
        ax.imshow(img)
        ax.set_title(f'{CIFAR10_CLASSES[label]}', fontsize=10)
        ax.axis('off')

        save_path = os.path.join(output_path, f'sample_{i+1:02d}_{CIFAR10_CLASSES[label]}.png')
        fig.savefig(save_path, dpi=PLOT_DPI, bbox_inches='tight', pad_inches=0.1)
        plt.close(fig)
        print(f"  Saved: {save_path}")

    print(f"\n  CIFAR-10: {num_samples} images saved to '{output_path}/'")


def save_medmnist_samples(output_dir='visualizations', num_samples=10):
    """Save MedMNIST (PathMNIST) sample images as individual PNG files."""
    output_path = os.path.join(output_dir, 'medmnist')
    os.makedirs(output_path, exist_ok=True)

    try:
        from medmnist import PathMNIST
    except ImportError:
        print("Error: medmnist library required. Install with: pip install medmnist")
        return

    print("Loading PathMNIST dataset...")
    dataset = PathMNIST(split='test', download=True, root='./data', transform=None)

    np.random.seed(42)
    indices = np.random.choice(len(dataset), num_samples, replace=False)

    for i, idx in enumerate(indices):
        img, label = dataset[idx]
        label_idx = int(np.array(label).ravel()[0])
        class_name = MEDMNIST_CLASSES[label_idx]

        fig, ax = plt.subplots(figsize=(2, 2))
        if isinstance(img, np.ndarray):
            if img.ndim == 2:
                ax.imshow(img, cmap='gray')
            else:
                ax.imshow(img)
        else:
            ax.imshow(np.array(img))
        ax.set_title(f'{class_name}', fontsize=10)
        ax.axis('off')

        save_path = os.path.join(output_path, f'sample_{i+1:02d}_{class_name}.png')
        fig.savefig(save_path, dpi=PLOT_DPI, bbox_inches='tight', pad_inches=0.1)
        plt.close(fig)
        print(f"  Saved: {save_path}")

    print(f"\n  MedMNIST: {num_samples} images saved to '{output_path}/'")


def main():
    parser = argparse.ArgumentParser(description='Visualize CIFAR-10 and MedMNIST datasets')
    parser.add_argument('--dataset', type=str, default='both',
                        choices=['cifar10', 'medmnist', 'both'],
                        help='Which dataset to visualize')
    parser.add_argument('--output', type=str, default='visualizations',
                        help='Output directory for PNG files')
    parser.add_argument('--samples', type=int, default=10,
                        help='Number of samples to save per dataset')
    args = parser.parse_args()

    print(f"\n{'=' * 50}")
    print("  Dataset Visualizer")
    print(f"{'=' * 50}\n")

    if args.dataset in ('cifar10', 'both'):
        print("\n[CIFAR-10]")
        save_cifar10_samples(args.output, args.samples)

    if args.dataset in ('medmnist', 'both'):
        print("\n[MedMNIST (PathMNIST)]")
        save_medmnist_samples(args.output, args.samples)

    print(f"\n{'=' * 50}")
    print(f"  All images saved to: {args.output}/")
    print(f"{'=' * 50}\n")


if __name__ == '__main__':
    main()