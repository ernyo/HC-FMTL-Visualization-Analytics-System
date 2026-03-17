import os
import math
import numpy as np
import torch
from PIL import Image, ImageDraw
from torch.utils.data import Dataset
from dataclasses import dataclass, field
from typing import List, Tuple, Optional
from .utils.mypath import MyPath
import random

# Shape and color definitions
SHAPE_NAMES = ["square", "circle", "triangle", "star"]
COLOR_NAMES = ["red", "green", "blue", "yellow"]

class SHAPES(Dataset):
    def __init__(self, root=MyPath.db_root_dir('SHAPES'), train=True, tasks=None, transform=None, dataidxs=None):
        """
        Synthetic SHAPES dataset for multi-task learning
        """
        self.root = root
        self.transform = transform
        self.dataidxs = dataidxs
        self.tasks = tasks if tasks is not None else []
        self.train = train
        
        # Create dataset if it doesn't exist
        if not os.path.exists(self.root):
            raise RuntimeError('Dataset not found!')
        
        # Load data
        self.images = []
        self.segs = []
        self.edges = []
        self.normals = []
        self.sals = []
        
        # Task flags
        self.do_seg = 'seg' in self.tasks
        self.do_edge = 'edge' in self.tasks
        self.do_normals = 'normals' in self.tasks
        self.do_sal = 'sal' in self.tasks
        
        # Load file lists
        split = 'train' if train else 'val'
        split_file = os.path.join(self.root, 'splits', f'{split}.txt')
        
        with open(split_file, 'r') as f:
            self.file_names = [line.strip() for line in f.readlines()]
        
        # Build file paths
        for file_name in self.file_names:
            self.images.append(os.path.join(self.root, 'images', f'{file_name}.png'))
            
            if self.do_seg:
                self.segs.append(os.path.join(self.root, 'segmentation', f'{file_name}_seg.png'))
            if self.do_edge:
                self.edges.append(os.path.join(self.root, 'edges', f'{file_name}_edge.png'))
            if self.do_normals:
                self.normals.append(os.path.join(self.root, 'normals', f'{file_name}_normal.png'))
            if self.do_sal:
                self.sals.append(os.path.join(self.root, 'saliency', f'{file_name}_sal.png'))
        
        self.__build_truncated_dataset__()
    
    def __build_truncated_dataset__(self):
        if self.dataidxs is not None:
            self.file_names = [self.file_names[idx] for idx in self.dataidxs]
            self.images = [self.images[idx] for idx in self.dataidxs]
            if self.do_seg:
                self.segs = [self.segs[idx] for idx in self.dataidxs]
            if self.do_edge:
                self.edges = [self.edges[idx] for idx in self.dataidxs]
            if self.do_normals:
                self.normals = [self.normals[idx] for idx in self.dataidxs]
            if self.do_sal:
                self.sals = [self.sals[idx] for idx in self.dataidxs]
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, index):
        sample = {}
        
        # Load image
        _img = np.array(Image.open(self.images[index]).convert('RGB'), dtype=np.float32)
        sample['image'] = _img
        
        # Load task-specific annotations
        if self.do_seg:
            _seg = np.array(Image.open(self.segs[index]), dtype=np.float32)
            sample['seg'] = np.expand_dims(_seg, axis=-1)
        
        if self.do_edge:
            _edge = np.array(Image.open(self.edges[index]), dtype=np.float32) / 255.0
            sample['edge'] = np.expand_dims(_edge, axis=-1)
        
        if self.do_normals:
            _normal = np.array(Image.open(self.normals[index]), dtype=np.float32)
            sample['normals'] = _normal
        
        if self.do_sal:
            _sal = np.array(Image.open(self.sals[index]), dtype=np.float32) / 255.0
            sample['sal'] = np.expand_dims(_sal, axis=-1)
        
        # Apply transformations
        if self.transform is not None:
            sample = self.transform(sample)
        
        sample['meta'] = {
            'file_name': str(self.file_names[index]),
            'size': (_img.shape[0], _img.shape[1])
        }
        
        return sample
    
    def get_task_dims(self):
        """Return the number of output dimensions for each task"""
        task_dims = {}
        if self.do_seg:
            task_dims['seg'] = len(SHAPE_NAMES) + 1  # +1 for background
        if self.do_edge:
            task_dims['edge'] = 1
        if self.do_normals:
            task_dims['normals'] = 3
        if self.do_sal:
            task_dims['sal'] = 1
        return task_dims
    
    def get_task_loss_weights(self):
        """Return loss weights for each task"""
        weights = {
            'seg': 1.0,
            'edge': 0.5,
            'normals': 0.8,
            'sal': 0.7
        }
        return {task: weights[task] for task in self.tasks if task in weights}