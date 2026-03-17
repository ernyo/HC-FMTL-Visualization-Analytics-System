import os
import math
import numpy as np
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
from dataclasses import dataclass, field
from typing import List, Optional
import random
from shape import Shape

ROOT = 'C:/Users/ern yon/OneDrive/Desktop/FYP/Project Documents/POC/dataset generation/SHAPES'
SEED = 0
rng = np.random.default_rng(SEED)
random.seed(SEED)

shape_drawer = Shape()

SHAPE_NAMES = ["square", "circle", "triangle", "star"]
COLOR_NAMES = ["red", "green", "blue", "yellow"]
COLOR_TO_RGB = {"red":(255,0,0),"green":(0,200,0),"blue":(0,128,255),"yellow":(255,200,0)}
@dataclass
class DataSpec:
    image_size: int = 64
    shapes: List[str] = field(default_factory=lambda: SHAPE_NAMES)
    colors: List[str] = field(default_factory=lambda: COLOR_NAMES)
    n_per_class: int = 200
    train_split: float = 0.7
    val_split: float = 0.15
    seed: int = 42

# Shape and color definitions
SHAPE_TO_LABEL = {name: idx + 1 for idx, name in enumerate(SHAPE_NAMES)}  # +1 for background=0

def draw_shape_all(draw, seg_img, seg_draw, edge_draw, normal_img, sal_draw,
                   shape, color, bbox, shape_label, depth):
    """
    Draw the shape on RGB/seg/edge/saliency; then synthesize depth inside the shape
    and compute normals from it. Writes depth and normals per-pixel.
    Returns the updated normal_img and depth array.
    """
    color_rgb = COLOR_TO_RGB[color]
    W, H = seg_img.size  # PIL returns (width, height)

    # 1) Draw shape on RGB/seg/edge/saliency
    if shape == "square":
        draw.rectangle(bbox, fill=color_rgb)
        seg_draw.rectangle(bbox, fill=shape_label)
        edge_draw.rectangle(bbox, outline=255, width=2)
        sal_draw.rectangle(bbox, fill=255)

    elif shape == "circle":
        draw.ellipse(bbox, fill=color_rgb)
        seg_draw.ellipse(bbox, fill=shape_label)
        edge_draw.ellipse(bbox, outline=255, width=2)
        sal_draw.ellipse(bbox, fill=255)

    elif shape == "triangle":
        x0, y0, x1, y1 = bbox
        pts = [(x0 + (x1 - x0) // 2, y0), (x1, y1), (x0, y1)]
        draw.polygon(pts, fill=color_rgb)
        seg_draw.polygon(pts, fill=shape_label)
        edge_draw.polygon(pts, outline=255, width=2)
        sal_draw.polygon(pts, fill=255)

    elif shape == "star":
        x0, y0, x1, y1 = bbox
        cx, cy = (x0 + x1) // 2, (y0 + y1) // 2
        outer = (min(x1 - x0, y1 - y0)) // 2
        inner = max(1, outer // 2)
        points = []
        for i in range(10):
            ang = math.pi / 2 + i * math.pi / 5
            r = outer if i % 2 == 0 else inner
            x = cx + r * math.cos(ang)
            y = cy - r * math.sin(ang)
            points.append((x, y))
        draw.polygon(points, fill=color_rgb)
        seg_draw.polygon(points, fill=shape_label)
        edge_draw.polygon(points, outline=255, width=2)
        sal_draw.polygon(points, fill=255)

    # 2) Build mask from seg_img (reliable)
    seg_arr_now = np.array(seg_img)  # (H, W) uint8
    mask = (seg_arr_now == shape_label)

    if mask.any():
        # coordinate grids normalized to [-1, 1]
        ys, xs = np.mgrid[0:seg_arr_now.shape[0], 0:seg_arr_now.shape[1]].astype(np.float32)
        Xn = (xs / max(1, W - 1)) * 2 - 1
        Yn = (ys / max(1, H - 1)) * 2 - 1

        # 3) Procedural depth field d (per-shape)
        if shape == "square":
            # Stronger plane tilt  # <<<
            a = (np.random.rand() - 0.5) * 2.0   # was 0.8
            b = (np.random.rand() - 0.5) * 2.0
            d = a * Xn + b * Yn

        elif shape == "circle":
            # Bowl with higher curvature  # <<<
            x0, y0, x1, y1 = bbox
            cx = (x0 + x1) / 2.0
            cy = (y0 + y1) / 2.0
            rmax = max(8.0, 0.5 * (min(x1 - x0, y1 - y0)))
            X = np.arange(W, dtype=np.float32)[None, :]
            Y = np.arange(H, dtype=np.float32)[:, None]
            r = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2)
            d = 1.2 - np.clip(r / rmax, 0, 1)    # a bit steeper

        elif shape == "triangle":
            # Saddle with stronger slopes  # <<<
            a = (np.random.rand() - 0.5) * 2.0
            b = (np.random.rand() - 0.5) * 2.0
            d = 0.8 * (a * Xn - b * Yn)

        elif shape == "star":
            # Radial ripple with more contrast  # <<<
            x0, y0, x1, y1 = bbox
            cx = (x0 + x1) / 2.0
            cy = (y0 + y1) / 2.0
            X = np.arange(W, dtype=np.float32)[None, :]
            Y = np.arange(H, dtype=np.float32)[:, None]
            rr = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2)
            d = 0.6 * np.cos(0.2 * rr)

        else:
            a = (np.random.rand() - 0.5) * 2.0
            b = (np.random.rand() - 0.5) * 2.0
            d = a * Xn + b * Yn

        # Limit d to inside the shape (outside remains 0)
        d_masked = np.zeros_like(d, dtype=np.float32)
        d_masked[mask] = d[mask].astype(np.float32)

        # 4) Stronger amplitude before gradient  # <<<
        depth_scale = 8.0
        d_scaled = d_masked * depth_scale

        # (Optional) tiny smoothing to reduce jaggies on boundaries
        # uncomment if needed:
        # d_scaled[1:-1, 1:-1] = (d_scaled[1:-1, 1:-1] +
        #                         d_scaled[:-2, 1:-1] + d_scaled[2:, 1:-1] +
        #                         d_scaled[1:-1, :-2] + d_scaled[1:-1, 2:]) / 5.0

        # Write into the global depth (so you also save it)
        depth[mask] = d_scaled[mask]

        # 5) Normals from local field (use d_scaled so background=0 doesn't wash it out)
        dzdy, dzdx = np.gradient(d_scaled)   # <<<
        nx = -dzdx
        ny = -dzdy
        nz = np.ones_like(d_scaled, dtype=np.float32)

        norm = np.sqrt(nx * nx + ny * ny + nz * nz) + 1e-8
        nx /= norm; ny /= norm; nz /= norm

        normals_rgb = np.stack([(nx + 1.0) * 127.5,
                                (ny + 1.0) * 127.5,
                                (nz + 1.0) * 127.5], axis=-1).astype(np.uint8)

        normal_arr = np.array(normal_img, dtype=np.uint8)
        normal_arr[mask] = normals_rgb[mask]
        normal_img = Image.fromarray(normal_arr)

    return normal_img, depth



def create_single_image(shape, color, spec, rng):
    """Create a single synthetic image with all task annotations"""
    H = W = spec.image_size
    
    # Create base image
    base_img = Image.new("RGB", (W, H), (30, 30, 30))
    draw = ImageDraw.Draw(base_img)
    
    # Create segmentation map
    seg_img = Image.new("L", (W, H), 0)  # 0 = background
    seg_draw = ImageDraw.Draw(seg_img)
    
    # Create edge map
    edge_img = Image.new("L", (W, H), 0)
    edge_draw = ImageDraw.Draw(edge_img)
    
    # Create normal map
    normal_img = Image.new("RGB", (W, H), (128, 128, 255))

    # Create depth map (grayscale, float values later converted to uint8 for saving)
    depth = np.zeros((H, W), dtype=np.float32)

    
    # Create saliency map
    sal_img = Image.new("L", (W, H), 0)
    sal_draw = ImageDraw.Draw(sal_img)
    
    # Random position and size
    size = rng.randint(low=H//3, high=H//2)
    x0 = rng.randint(low=5, high=W-size-5)
    y0 = rng.randint(low=5, high=H-size-5)
    x1 = x0 + size
    y1 = y0 + size
    bbox = (x0, y0, x1, y1)
    
    # Draw shape on all images (returns updated normal_img)
    shape_label = SHAPE_TO_LABEL[shape]
    normal_img, depth = draw_shape_all(
        draw, seg_img, seg_draw, edge_draw, normal_img, sal_draw,
        shape, color, bbox, shape_label, depth
    )

    if depth.max() > depth.min():
        depth_norm = ((depth - depth.min()) / (depth.max()-depth.min()) * 255).astype(np.uint8)
    else:
        depth_norm = np.zeros_like(depth, dtype=np.uint8)
        
    # Convert to numpy arrays
    img_arr = np.array(base_img).astype(np.uint8)
    seg_arr = np.array(seg_img).astype(np.uint8)
    edge_arr = np.array(edge_img).astype(np.uint8)
    normal_arr = np.array(normal_img).astype(np.uint8)
    sal_arr = np.array(sal_img).astype(np.uint8)
    
    return img_arr, seg_arr, edge_arr, normal_arr, sal_arr, depth_norm

def generate_shapes_dataset(output_dir, spec=None):
    """Generate complete SHAPES dataset"""
    if spec is None:
        spec = DataSpec()
    
    # Create directories
    dirs = ['images', 'segmentation', 'edges', 'normals', 'saliency', 'depth', 'splits']
    for dir_name in dirs:
        os.makedirs(os.path.join(output_dir, dir_name), exist_ok=True)
    
    # Set random seed for reproducibility
    rng = np.random.RandomState(spec.seed)
    
    # Generate all images
    all_files = []
    file_counter = 0
    
    for shape in spec.shapes:
        for color in spec.colors:
            for i in range(spec.n_per_class):
                # Create image with all annotations
                img, seg, edge, normal, sal, depth = create_single_image(shape, color, spec, rng)
                
                # Generate unique filename
                file_name = f"{shape}_{color}_{file_counter:06d}"
                all_files.append(file_name)
                
                # Save all versions
                Image.fromarray(img).save(os.path.join(output_dir, 'images', f'{file_name}.png'))
                Image.fromarray(seg).save(os.path.join(output_dir, 'segmentation', f'{file_name}_seg.png'))
                Image.fromarray(edge).save(os.path.join(output_dir, 'edges', f'{file_name}_edge.png'))
                Image.fromarray(normal).save(os.path.join(output_dir, 'normals', f'{file_name}_normal.png'))
                Image.fromarray(sal).save(os.path.join(output_dir, 'saliency', f'{file_name}_sal.png'))
                Image.fromarray(depth).save(os.path.join(output_dir, 'depth', f'{file_name}_depth.png'))
                
                file_counter += 1
    
    # Create splits
    n_total = len(all_files)
    n_train = int(n_total * spec.train_split)
    n_val = int(n_total * spec.val_split)
    
    indices = rng.permutation(n_total)
    train_indices = indices[:n_train]
    val_indices = indices[n_train:n_train + n_val]
    test_indices = indices[n_train + n_val:]
    
    # Write split files
    with open(os.path.join(output_dir, 'splits', 'train.txt'), 'w') as f:
        for idx in train_indices:
            f.write(f"{all_files[idx]}\n")
    
    with open(os.path.join(output_dir, 'splits', 'val.txt'), 'w') as f:
        for idx in val_indices:
            f.write(f"{all_files[idx]}\n")
    
    with open(os.path.join(output_dir, 'splits', 'test.txt'), 'w') as f:
        for idx in test_indices:
            f.write(f"{all_files[idx]}\n")
    
    print(f"Generated {n_total} images:")
    print(f"  - Train: {n_train} images")
    print(f"  - Validation: {n_val} images") 
    print(f"  - Test: {len(test_indices)} images")
    print(f"Dataset saved to: {output_dir}")
    
    return all_files, train_indices, val_indices, test_indices

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
dataset_spec = DataSpec(
    image_size=128,      # Larger images for better quality
    n_per_class=250,     # More samples per class
    seed=42              # Fixed seed for reproducibility
)

# Generate the dataset
print("Generating SHAPES dataset...")
all_files, train_idx, val_idx, test_idx = generate_shapes_dataset(output_directory, dataset_spec)

# Visualize some samples
print("\nVisualizing random samples...")
random_samples = np.random.choice(all_files, 3, replace=False)

for sample in random_samples:
    visualize_sample(output_directory, sample)

# Show dataset statistics
print("\nDataset Statistics:")
print(f"Total images: {len(all_files)}")
print(f"Number of shapes: {len(dataset_spec.shapes)}")
print(f"Number of colors: {len(dataset_spec.colors)}")
print(f"Images per class: {dataset_spec.n_per_class}")
print(f"Image size: {dataset_spec.image_size}x{dataset_spec.image_size}")

# Create a summary file
summary_path = os.path.join(output_directory, "dataset_summary.txt")
with open(summary_path, 'w') as f:
    f.write("SHAPES Dataset Summary\n")
    f.write("=====================\n\n")
    f.write(f"Total images: {len(all_files)}\n")
    f.write(f"Shapes: {', '.join(dataset_spec.shapes)}\n")
    f.write(f"Colors: {', '.join(dataset_spec.colors)}\n")
    f.write(f"Images per class: {dataset_spec.n_per_class}\n")
    f.write(f"Image size: {dataset_spec.image_size}x{dataset_spec.image_size}\n")
    f.write(f"Random seed: {dataset_spec.seed}\n\n")
    
    f.write("Split sizes:\n")
    f.write(f"  Training: {len(train_idx)} images\n")
    f.write(f"  Validation: {len(val_idx)} images\n")
    f.write(f"  Test: {len(test_idx)} images\n")

print(f"Dataset summary saved to: {summary_path}")

# Verify the dataset structure
print("\nDataset structure:")
for root, dirs, files in os.walk(output_directory):
    level = root.replace(output_directory, '').count(os.sep)
    indent = ' ' * 2 * level
    print(f"{indent}{os.path.basename(root)}/")
    subindent = ' ' * 2 * (level + 1)
    for file in files[:3]:  # Show first 3 files in each directory
        if file.endswith('.png') or file.endswith('.txt'):
            print(f"{subindent}{file}")
    if len(files) > 3:
        print(f"{subindent}... and {len(files) - 3} more files")

print("\nDataset generation complete! 🎉")
print(f"You can now use the dataset at: {output_directory}")
print("\nTo use with FedHCA², update the path in datasets/utils/mypath.py:")
print(f"db_root = '{os.path.abspath(output_directory)}'")