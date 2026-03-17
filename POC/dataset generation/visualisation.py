import os
import math
import numpy as np
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
from dataclasses import dataclass, field
from typing import List, Optional
import random

ROOT = 'C:/Users/ern yon/OneDrive/Desktop/FYP/Project Documents/POC/dataset generation/SHAPES'
SEED = 0
rng = np.random.default_rng(SEED)
random.seed(SEED)

SHAPE_NAMES = ["square", "circle", "triangle", "star"]
COLOR_NAMES = ["red", "green", "blue", "yellow"]
COLOR_TO_RGB = {"red":(255,0,0),"green":(0,200,0),"blue":(0,128,255),"yellow":(255,200,0)}

class DataSpec:
    image_size: int = 128
    shapes: List[str] = field(default_factory=lambda: SHAPE_NAMES)
    colors: List[str] = field(default_factory=lambda: COLOR_NAMES)
    n_per_class: int = 250
    train_split: float = 0.7
    val_split: float = 0.15
    seed: int = 42

def visualize_sample(output_dir, file_name):
    """Visualize a sample from the dataset"""
    fig, axes = plt.subplots(1, 6, figsize=(18, 3))
    
    # Load images
    image = Image.open(os.path.join(output_dir, 'images', f'{file_name}.png'))
    seg = Image.open(os.path.join(output_dir, 'segmentation', f'{file_name}_seg.png'))
    edge = Image.open(os.path.join(output_dir, 'edges', f'{file_name}_edge.png'))
    normal = Image.open(os.path.join(output_dir, 'normals', f'{file_name}_normal.png'))
    sal = Image.open(os.path.join(output_dir, 'saliency', f'{file_name}_sal.png'))
    depth = Image.open(os.path.join(output_dir, 'depth', f'{file_name}_depth.png'))
    
    # Plot
    axes[0].imshow(image)
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    axes[1].imshow(seg, cmap='tab10')
    axes[1].set_title('Segmentation')
    axes[1].axis('off')
    
    axes[2].imshow(edge, cmap='gray')
    axes[2].set_title('Edges')
    axes[2].axis('off')
    
    axes[3].imshow(normal)
    axes[3].set_title('Normals')
    axes[3].axis('off')
    
    axes[4].imshow(sal, cmap='gray')
    axes[4].set_title('Saliency')
    axes[4].axis('off')

    axes[5].imshow(depth, cmap='inferno')
    axes[5].set_title('Depth')
    axes[5].axis('off')
    
    plt.tight_layout()
    plt.show()
    
    print(f"File: {file_name}")
    shape, color, idx = file_name.split('_')
    print(f"Shape: {shape}, Color: {color}")

# Configuration
output_directory = ROOT


all_files = []
file_counter = 0

for shape in SHAPE_NAMES:
        for color in COLOR_NAMES:
            for _ in range(250):
                file_name = f"{shape}_{color}_{file_counter:06d}"
                all_files.append(file_name)
                file_counter += 1

print("\nVisualizing random samples...")
random_samples = np.random.choice(all_files, 3, replace=False)

for sample in random_samples:
    visualize_sample(output_directory, sample)
