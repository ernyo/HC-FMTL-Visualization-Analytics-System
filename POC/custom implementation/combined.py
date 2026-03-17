import os
import math, random, copy
from dataclasses import dataclass, field
from typing import List, Optional
import collections.abc as collections
import re
import yaml
from tqdm import tqdm

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from IPython.display import display
from scipy.optimize import minimize

import argparse
import datetime
import shutil
import time
from contextlib import nullcontext

import cv2
import torch
import torch.backends.cudnn as cudnn
import torchvision.transforms

#=============================#
#         CONSTANTS           #
#=============================#

# configs
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ROOT = 'C:/Users/ern yon/OneDrive/Desktop/FYP/Project Documents/POC/custom implementation' 
DATA_DIR = os.path.join(ROOT, 'SHAPES')
SEED = 0
rng = np.random.default_rng(SEED)
torch.manual_seed(SEED)
random.seed(SEED)

# constants
SHAPE_NAMES = ["square", "circle", "triangle", "star"]
COLOR_NAMES = ["red", "green", "blue", "yellow"]
COLOR_TO_RGB = {"red":(255,0,0),"green":(0,200,0),"blue":(0,128,255),"yellow":(255,200,0)}

# dataset parameters
IMAGE_SIZE = 64
COUNT = 200
TRAIN_SPLIT = 0.7
VAL_SPLIT = 0.15
BATCH_SIZE = 64

# training parameters
SHAPES_OUT_CHANNELS = {'semseg': 5, 'normals': 3, 'edge': 1, 'depth': 1, 'saliency': 2}
TRAIN_SCALE = {'shapes': (128, 128)}
TEST_SCALE = {'shapes': (128, 128)}
NUM_TRAIN_IMAGES = {'shapes': 2800}
NUM_TEST_IMAGES = {'shapes': 600}

#=============================#
#         DATASET             #
#=============================#

# Data Specification
@dataclass
class DataSpec:
    image_size: int = IMAGE_SIZE
    shapes: List[str] = field(default_factory=lambda: SHAPE_NAMES)
    colors: List[str] = field(default_factory=lambda: COLOR_NAMES)
    n_per_class: int = COUNT
    train_split: float = TRAIN_SPLIT
    val_split: float = VAL_SPLIT
    batch_size: int = BATCH_SIZE
    seed: int = SEED

class SHAPES(torch.utils.data.Dataset):

    def __init__(self, root='', train=True, tasks=None, transform=None, dataidxs=None):
        """
        Initialize the SHAPES dataset
        :param str root: Root directory of dataset
        :param bool train: True for training set, False for validation set
        :param list tasks: Tasks to be loaded
        :param Compose transform: Data augmentation
        :param list dataidxs: Indexes of the data to be loaded (for small dataset loading, default is None for all data)
        """
        self.root = root
        self.transform = transform
        self.dataidxs = dataidxs

        if not os.path.exists(self.root):
            raise RuntimeError('Dataset not found!')

        # Original Images
        self.images = []
        images_dir = os.path.join(self.root, 'images')

        # Edge Detection
        self.do_edge = 'edge' in tasks
        self.edges = []
        edge_gt_dir = os.path.join(root, 'edge')

        # Semantic Segmentation
        self.do_semseg = 'semseg' in tasks
        self.semsegs = []
        semseg_gt_dir = os.path.join(root, 'segmentation')

        # Surface Normals Estimation
        self.do_normals = 'normals' in tasks
        self.normals = []
        normal_gt_dir = os.path.join(root, 'normals')

        # Depth Estimation
        self.do_depth = 'depth' in tasks
        self.depths = []
        depth_gt_dir = os.path.join(root, 'depth')

        # Saliency Detection
        self.do_saliency = 'saliency' in tasks
        self.sals = []
        saliency_gt_dir = os.path.join(root, 'saliency')

        # Separation of training set and validation set
        splits_dir = os.path.join(self.root, 'splits')
        if train:
            split_f = os.path.join(splits_dir, 'train.txt')
        else:
            split_f = os.path.join(splits_dir, 'val.txt')

        with open(split_f, "r") as f:
            file_names = [x.strip() for x in f.readlines()]

        self.im_ids = file_names
        for x in file_names:
            _image = os.path.join(images_dir, x + '.png')
            assert os.path.isfile(_image)
            self.images.append(_image)

            _edge = os.path.join(self.root, edge_gt_dir, x + '.png')
            assert os.path.isfile(_edge)
            self.edges.append(_edge)

            _semseg = os.path.join(self.root, semseg_gt_dir, x + '.png')
            assert os.path.isfile(_semseg)
            self.semsegs.append(_semseg)

            _normal = os.path.join(self.root, normal_gt_dir, x + '.png')
            assert os.path.isfile(_normal)
            self.normals.append(_normal)

            _depth = os.path.join(self.root, depth_gt_dir, x + '.png')
            assert os.path.isfile(_depth)
            self.depths.append(_depth)

            _saliency = os.path.join(self.root, saliency_gt_dir, x + '.png')
            assert os.path.isfile(_saliency)
            self.sals.append(_saliency)

        if self.do_edge:
            assert (len(self.images) == len(self.edges))
        if self.do_semseg:
            assert (len(self.images) == len(self.semsegs))
        if self.do_normals:
            assert (len(self.images) == len(self.normals))
        if self.do_depth:
            assert (len(self.images) == len(self.depths))
        if self.do_saliency:
            assert (len(self.images) == len(self.sals))

        self.__build_truncated_dataset__()

    def __build_truncated_dataset__(self):
        if self.dataidxs is not None:
            self.images = [self.images[idx] for idx in self.dataidxs]
            self.edges = [self.edges[idx] for idx in self.dataidxs if self.do_edge]
            self.semsegs = [self.semsegs[idx] for idx in self.dataidxs if self.do_semseg]
            self.normals = [self.normals[idx] for idx in self.dataidxs if self.do_normals]
            self.depths = [self.depths[idx] for idx in self.dataidxs if self.do_depth]
            self.sals = [self.sals[idx] for idx in self.dataidxs if self.do_saliency]

    def __getitem__(self, index):
        sample = {}

        _img = self._load_img(index)
        sample['image'] = _img

        if self.do_edge:
            _edge = self._load_edge(index)
            assert _img.shape[0:2] == _edge.shape[0:2]
            sample['edge'] = np.expand_dims(_edge, axis=-1)

        if self.do_semseg:
            _semseg = self._load_semseg(index)
            assert _img.shape[0:2] == _semseg.shape[0:2]
            sample['semseg'] = np.expand_dims(_semseg, axis=-1)

        if self.do_normals:
            _normals = self._load_normals(index)
            assert _img.shape[0:2] == _normals.shape[0:2]
            sample['normals'] = _normals

        if self.do_depth:
            _depth = self._load_depth(index)
            assert _img.shape[0:2] == _depth.shape[0:2]
            sample['depth'] = np.expand_dims(_depth, axis=-1)

        if self.do_saliency:
            _saliency = self._load_saliency(index)
            assert _img.shape[0:2] == _saliency.shape[0:2]
            sample['saliency'] = np.expand_dims(_saliency, axis=-1)

        # Make transforms and augmentations
        if self.transform is not None:
            sample = self.transform(sample)

        sample['meta'] = {'file_name': str(self.im_ids[index]), 'size': (_img.shape[0], _img.shape[1])}
        return sample

    def __len__(self):
        return len(self.images)

    def _load_img(self, index):
        _img = np.array(Image.open(self.images[index]).convert('RGB'), dtype=np.float32, copy=False)
        return _img

    def _load_edge(self, index):
        _edge = np.array(Image.open(self.edges[index]), dtype=np.float32, copy=False) / 255.
        return _edge

    def _load_semseg(self, index):
        _semseg = np.array(Image.open(self.semsegs[index]), dtype=np.float32, copy=False) - 1
        _semseg[_semseg == -1] = 255
        return _semseg

    def _load_normals(self, index):
        _tmp = np.array(Image.open(self.normals[index]), dtype=np.float32, copy=False)
        _normals = 2.0 * _tmp / 255.0 - 1.0  # [0,255] => [-1，1]

        return _normals

    def _load_depth(self, index):
        _depth = np.load(self.depths[index]).astype(np.float32)
        return _depth
    
    def _load_saliency(self, index):
        _saliency = np.array(Image.open(self.sals[index]), dtype=np.float32, copy=False) / 255.
        return _saliency


#=============================#
#     DATA TRANSFORMATIONS    #
#=============================#

def get_transformations(size, train=True):
    """
    Get data transforms and augmentations
    :param tuple size: Image size
    :param bool train: Training or validation
    :return Compose transform object
    """
    if train:
        augs = torchvision.transforms.Compose([
            RandomScaling(min_scale_factor=0.5, max_scale_factor=2.0),
            RandomCrop(size, cat_max_ratio=0.75),
            RandomHorizontallyFlip(),
            PhotoMetricDistortion(),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            PadImage(size),
            AddIgnoreRegions(),
            ToTensor()
        ])
    else:
        augs = torchvision.transforms.Compose([
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            PadImage(size),
            AddIgnoreRegions(),
            ToTensor()
        ])

    return augs


class RandomScaling(object):
    """
    Randomly scale the image and labels
    :param float min_scale_factor: Minimum scaling value
    :param float max_scale_factor: Maximum scaling value
    """

    def __init__(self, min_scale_factor, max_scale_factor):
        self.min_scale_factor = min_scale_factor
        self.max_scale_factor = max_scale_factor

    def get_random_scale(self):
        if self.min_scale_factor < 0 or self.min_scale_factor > self.max_scale_factor:
            raise ValueError("Unexpected value of 'min_scale_factor'!")

        if self.min_scale_factor == self.max_scale_factor:
            min_scale_factor = float(self.min_scale_factor)
            return min_scale_factor

        # Uniformly sampling of the value from [min, max)
        return np.random.uniform(low=self.min_scale_factor, high=self.max_scale_factor)

    def scale(self, key, unscaled, scale):
        # No random scaling if scale == 1.
        if scale == 1.0:
            return unscaled
        image_shape = unscaled.shape[0:2]
        new_dim = tuple([int(x * scale) for x in image_shape])

        unscaled = np.squeeze(unscaled)
        if key == 'image':  # float value, linear interpolation
            scaled = cv2.resize(unscaled, new_dim[::-1], interpolation=cv2.INTER_LINEAR)
        else:
            scaled = cv2.resize(unscaled, new_dim[::-1], interpolation=cv2.INTER_NEAREST)
        if scaled.ndim == 2:
            scaled = np.expand_dims(scaled, axis=2)

        # Adjust depth maps with rescaling
        if key == 'depth':
            scaled /= scale

        return scaled

    def __call__(self, sample):
        random_scale = self.get_random_scale()

        for key, target in sample.items():
            sample[key] = self.scale(key, target, random_scale)

        return sample


class RandomCrop(object):
    """
    Randomly crop image and labels if it exceeds desired size
    :param int/tuple size: Desired size
    """

    def __init__(self, size, cat_max_ratio=1):
        if isinstance(size, int):
            self.size = (size, size)
        elif isinstance(size, tuple):
            self.size = size
        else:
            raise ValueError('Crop size must be an int or a tuple!')
        self.cat_max_ratio = cat_max_ratio

    def get_random_crop_loc(self, uncropped):
        uncropped_shape = uncropped.shape
        img_height = uncropped_shape[0]
        img_width = uncropped_shape[1]

        desired_height = self.size[0]
        desired_width = self.size[1]
        if img_height == desired_height and img_width == desired_width:
            return None

        # Get random offset uniformly from [0, max_offset)
        max_offset_height = max(img_height - desired_height, 0)
        max_offset_width = max(img_width - desired_width, 0)

        offset_height = random.randint(0, max_offset_height)
        offset_width = random.randint(0, max_offset_width)
        crop_loc = [offset_height, offset_height + desired_height, offset_width, offset_width + desired_width]

        return crop_loc

    def random_crop(self, uncropped, crop_loc):
        if not crop_loc:
            return uncropped

        cropped = uncropped[crop_loc[0]:crop_loc[1], crop_loc[2]:crop_loc[3], :]

        return cropped

    def __call__(self, sample):
        crop_location = self.get_random_crop_loc(sample['image'])

        if self.cat_max_ratio < 1. and 'semseg' in sample.keys():
            # Repeat 10 times
            for _ in range(10):
                seg_tmp = self.random_crop(sample['semseg'], crop_location)
                labels, cnt = np.unique(seg_tmp, return_counts=True)
                cnt = cnt[labels != 255]
                if len(cnt) > 1 and np.max(cnt) / np.sum(cnt) < self.cat_max_ratio:
                    break
                crop_location = self.get_random_crop_loc(sample['image'])

        for key, target in sample.items():
            sample[key] = self.random_crop(target, crop_location)

        return sample


class RandomHorizontallyFlip(object):
    """
    Randomly horizontally flip image and labels with probability of 0.5
    """

    def __call__(self, sample):
        if random.random() < 0.5:
            for key, val in sample.items():
                sample[key] = np.fliplr(val).copy()
                # Flip the normal direction
                if key == 'normals':
                    sample[key][:, :, 0] *= -1

        return sample


class PadImage(object):
    """
    Pad image and labels to have dimensions >= [size_height, size_width]
    :param int/tuple size: Desired size
    """

    def __init__(self, size):
        if isinstance(size, int):
            self.size = (size, size)
        elif isinstance(size, tuple):
            self.size = size
        else:
            raise ValueError('Padding size must be an int or a tuple!')

        self.fill_index = {
            'image': [0, 0, 0],
            'edge': 255,
            'semseg': 255,
            'human_parts': 255,
            'sal': 255,
            'normals': [0., 0., 0.],
            'depth': 0
        }

    def pad(self, key, unpadded):
        unpadded_shape = unpadded.shape

        if unpadded_shape[0] >= self.size[0] and unpadded_shape[1] >= self.size[1]:
            return unpadded

        delta_height = max(self.size[0] - unpadded_shape[0], 0)
        delta_width = max(self.size[1] - unpadded_shape[1], 0)

        # Location to place image
        height_location = [delta_height // 2, (delta_height // 2) + unpadded_shape[0]]
        width_location = [delta_width // 2, (delta_width // 2) + unpadded_shape[1]]

        pad_value = self.fill_index[key]
        max_height = max(self.size[0], unpadded_shape[0])
        max_width = max(self.size[1], unpadded_shape[1])

        padded = np.full((max_height, max_width, unpadded_shape[2]), pad_value, dtype=unpadded.dtype)
        padded[height_location[0]:height_location[1], width_location[0]:width_location[1], :] = unpadded

        return padded

    def __call__(self, sample):
        for key, val in sample.items():
            sample[key] = self.pad(key, val)

        return sample


class Normalize:
    """
    Normalize image by first mapping from [0, 255] to [0, 1] and then applying standardization.
    :param list mean: Mean values for each channel
    :param list std: Standard deviation values for each channel
    """

    def __init__(self, mean, std):
        self.mean = np.array(mean, dtype=np.float32).reshape(1, 1, 3)
        self.std = np.array(std, dtype=np.float32).reshape(1, 1, 3)

    def normalize_img(self, img):
        assert img.dtype == np.float32
        scaled = img.copy() / 255.
        scaled -= self.mean
        scaled /= self.std

        return scaled

    def __call__(self, sample):
        sample['image'] = self.normalize_img(sample['image'])

        return sample


class AddIgnoreRegions:
    """
    Add ignore regions to labels
    """

    def __call__(self, sample):
        for key in sample.keys():
            tmp = sample[key]
            if key == 'normals':
                # Check areas with norm 0
                norm = np.sqrt(tmp[:, :, 0]**2 + tmp[:, :, 1]**2 + tmp[:, :, 2]**2)
                tmp[norm == 0, :] = 255
                sample[key] = tmp
            elif key == 'human_parts':
                # Check for images without human part annotations
                if ((tmp == 0) | (tmp == 255)).all():
                    tmp = np.full(tmp.shape, 255, dtype=tmp.dtype)
                    sample[key] = tmp
            elif key == 'depth':
                tmp[tmp == 0] = 255
                sample[key] = tmp

        return sample


class PhotoMetricDistortion:
    """
    Apply photometric distortion to image sequentially, every transformation
    is applied with a probability of 0.5. The position of random contrast is in
    second or second to last.
    1. random brightness
    2. random contrast (mode 0)
    3. convert color from RGB to HSV
    4. random saturation
    5. random hue
    6. convert color from HSV to RGB
    7. random contrast (mode 1)

    :param int brightness_delta: Delta of brightness, defaults to 32
    :param tuple contrast_range: Range of contrast, defaults to (0.5, 1.5)
    :param tuple saturation_range: Range of saturation, defaults to (0.5, 1.5)
    :param int hue_delta: Delta of hue, defaults to 18
    """

    def __init__(self, brightness_delta=32, contrast_range=(0.5, 1.5), saturation_range=(0.5, 1.5), hue_delta=18):
        self.brightness_delta = brightness_delta
        self.contrast_lower, self.contrast_upper = contrast_range
        self.saturation_lower, self.saturation_upper = saturation_range
        self.hue_delta = hue_delta

    def convert(self, img, alpha=1, beta=0):
        img = img.astype(np.float32) * alpha + beta
        img = np.clip(img, 0, 255)
        return img.astype(np.uint8)

    def brightness(self, img):
        if random.random() < 0.5:
            return self.convert(img, beta=random.uniform(-self.brightness_delta, self.brightness_delta))
        return img

    def contrast(self, img):
        if random.random() < 0.5:
            return self.convert(img, alpha=random.uniform(self.contrast_lower, self.contrast_upper))
        return img

    def saturation(self, img):
        # image in HSV color
        if random.random() < 0.5:
            img[:, :, 1] = self.convert(img[:, :, 1],
                                        alpha=random.uniform(self.saturation_lower, self.saturation_upper))
        return img

    def hue(self, img):
        # image in HSV color
        if random.random() < 0.5:
            img[:, :, 0] = (img[:, :, 0].astype(int) + random.randint(-self.hue_delta, self.hue_delta - 1)) % 180
        return img

    def __call__(self, sample):
        img = sample['image']
        img = img.astype(np.uint8)  # functions need a uint8 image

        # f_mode == True -> do random contrast first, False -> do random contrast last
        f_mode = random.random() < 0.5

        img = self.brightness(img)

        if f_mode:
            img = self.contrast(img)

        img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        img = self.saturation(img)
        img = self.hue(img)
        img = cv2.cvtColor(img, cv2.COLOR_HSV2RGB)

        if not f_mode:
            img = self.contrast(img)

        sample['image'] = img.astype(np.float32)

        return sample


class ToTensor(object):
    """
    Convert ndarrays in sample to Tensors.
    """

    def __call__(self, sample):
        for key, val in sample.items():
            sample[key] = torch.from_numpy(val.transpose((2, 0, 1))).float()

        return sample

#=============================#
#            MIL              #
#=============================#
int_classes = int
string_classes = str

_use_shared_memory = False
r"""Whether to use shared memory in default_collate"""

numpy_type_map = {
    'float64': torch.DoubleTensor,
    'float32': torch.FloatTensor,
    'float16': torch.HalfTensor,
    'int64': torch.LongTensor,
    'int32': torch.IntTensor,
    'int16': torch.ShortTensor,
    'int8': torch.CharTensor,
    'uint8': torch.ByteTensor,
}

def collate_mil(batch):
    """
    Puts each data field into a tensor with outer dimension batch size.
    Custom-made for supporting MIL
    """
    error_msg = "Batch must contain tensors, numbers, dicts or lists; found {}"
    elem_type = type(batch[0])
    if isinstance(batch[0], torch.Tensor):
        out = None
        if _use_shared_memory:
            # If we're in a background process, concatenate directly into a
            # shared memory tensor to avoid an extra copy
            numel = sum([x.numel() for x in batch])
            storage = batch[0].storage()._new_shared(numel)
            out = batch[0].new(storage)
        return torch.stack(batch, 0, out=out)

    elif elem_type.__module__ == 'numpy' and elem_type.__name__ != 'str_' and elem_type.__name__ != 'string_':
        elem = batch[0]
        if elem_type.__name__ == 'ndarray':
            # array of string classes and object
            if re.search('[SaUO]', elem.dtype.str) is not None:
                raise TypeError(error_msg.format(elem.dtype))

            return torch.stack([torch.from_numpy(b) for b in batch], 0)
        if elem.shape == ():  # scalars
            py_type = float if elem.dtype.name.startswith('float') else int
            return numpy_type_map[elem.dtype.name](list(map(py_type, batch)))

    elif isinstance(batch[0], int_classes):
        return torch.LongTensor(batch)

    elif isinstance(batch[0], float):
        return torch.DoubleTensor(batch)

    elif isinstance(batch[0], string_classes):
        return batch

    elif isinstance(batch[0], collections.Mapping):
        batch_modified = {key: collate_mil([d[key] for d in batch]) for key in batch[0] if key.find('idx') < 0}
        if 'edgeidx' in batch[0]:
            batch_modified['edgeidx'] = [batch[x]['edgeidx'] for x in range(len(batch))]
        return batch_modified

    elif isinstance(batch[0], collections.Sequence):
        # from invpt
        out = []
        for samples in batch:
            out.append(collate_mil(samples))
        return out

    raise TypeError((error_msg.format(type(batch[0]))))


#=============================#
#          UTILITIES          #
#=============================#

def get_st_config(dataset_configs, local_rank=0):
    """
    Get single-task client configs
    """

    st_configs = {}
    for data_config in dataset_configs:
        dataname = data_config['dataname']
        train_transforms = get_transformations(TRAIN_SCALE[dataname], train=True)
        val_transforms = get_transformations(TEST_SCALE[dataname], train=False)

        # number of clients is defined in task_dict
        task_dict = data_config['task_dict']
        n_clients = sum(task_dict.values())
        if local_rank == 0:
            print("Training %d single-task models on %s" % (n_clients, dataname))

        task_list = []
        for task_name in task_dict:
            task_list += [task_name] * task_dict[task_name]

        # random partition of dataset
        idxs = np.random.permutation(NUM_TRAIN_IMAGES[dataname])
        batch_idxs = np.array_split(idxs, n_clients)
        net_task_dataidx_map = [{'task_list': [task_list[i]], 'dataidx': batch_idxs[i]} for i in range(n_clients)]

        st_configs[dataname] = data_config  # defined in yml
        st_configs[dataname]['n_clients'] = n_clients
        st_configs[dataname]['train_transforms'] = train_transforms
        st_configs[dataname]['val_transforms'] = val_transforms
        st_configs[dataname]['net_task_dataidx_map'] = net_task_dataidx_map

    return st_configs


def get_mt_config(dataset_configs, local_rank=0):
    """
    Get multi-task client configs
    """

    mt_configs = {}
    for data_config in dataset_configs:
        dataname = data_config['dataname']
        train_transforms = get_transformations(TRAIN_SCALE[dataname], train=True)
        val_transforms = get_transformations(TEST_SCALE[dataname], train=False)

        # number of models is defined in client_num
        n_clients = data_config['client_num']
        if local_rank == 0:
            print("Training %d multi-task models on %s" % (n_clients, dataname))

        task_dict = data_config['task_dict']
        task_list = []
        for task_name in task_dict:
            task_list += [task_name] * (task_dict[task_name] > 0)

        # random partition of dataset
        idxs = np.random.permutation(NUM_TRAIN_IMAGES[dataname])
        batch_idxs = np.array_split(idxs, n_clients)
        net_task_dataidx_map = [{'task_list': task_list, 'dataidx': batch_idxs[i]} for i in range(n_clients)]

        mt_configs[dataname] = data_config  # defined in yml
        mt_configs[dataname]['n_clients'] = n_clients
        mt_configs[dataname]['train_transforms'] = train_transforms
        mt_configs[dataname]['val_transforms'] = val_transforms
        mt_configs[dataname]['net_task_dataidx_map'] = net_task_dataidx_map

    return mt_configs


class RunningMeter(object):
    def __init__(self):
        self.avg = 0
        self.sum = 0
        self.count = 0

    def reset(self):
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def create_dir(directory):
    """
    Create required directory if it does not exist
    """

    if not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)


def create_results_dir(results_dir, exp_name):
    """
    Create required results directory if it does not exist
    :param str results_dir: Directory to create subdirectory in
    :param str exp_name: Name of experiment to be used in the directory created
    :return: Path of experiment directory and checkpoint directory
    """

    exp_dir = os.path.join(results_dir, exp_name)
    checkpoint_dir = os.path.join(exp_dir, 'checkpoints')
    create_dir(results_dir)
    create_dir(exp_dir)
    create_dir(checkpoint_dir)

    return exp_dir, checkpoint_dir


def create_pred_dir(results_dir, exp_name, all_nets):
    """
    Create required prediction directory if it does not exist
    :param str results_dir: Directory to create subdirectory in
    :param str exp_name: Name of experiment to be used in the directory created
    :param dict all_nets: Clients
    :return: Path of checkpoint directory and prediction dictionary
    """

    exp_dir = os.path.join(results_dir, exp_name)
    checkpoint_dir = os.path.join(exp_dir, 'checkpoints')
    pred_dir = os.path.join(exp_dir, 'predictions')
    create_dir(pred_dir)

    for idx in range(len(all_nets)):
        for task in all_nets[idx]['tasks']:
            task_dir = os.path.join(pred_dir, str(idx) + '_' + task)
            create_dir(task_dir)
            if task == 'edge':
                create_dir(os.path.join(task_dir, 'img'))

    return checkpoint_dir, pred_dir


def get_loss_metric(loss_meter, tasks, prefix, idx):
    """
    Get loss statistics
    :param dict loss_meter: Loss meter
    :param str tasks: List of tasks
    :param str prefix: Prefix for the loss, train or val
    :param int idx: Client index
    :return: loss statistics
    """

    if len(tasks) == 1:
        mt = False
        statistics = {}
    else:
        mt = True
        statistics = {prefix + '/' + str(idx) + '_loss_sum': 0.0}

    for task in tasks:
        if mt:
            statistics[prefix + '/' + str(idx) + '_loss_sum'] += loss_meter[task].avg
        statistics[prefix + '/' + str(idx) + '_' + task] = loss_meter[task].avg
        loss_meter[task].reset()

    return statistics


def to_cuda(batch):
    """
    Move batch to GPU
    :param dict batch: Input batch
    :return: Batch on GPU
    """

    if type(batch) is dict:
        out = {}
        for k, v in batch.items():
            if k == 'meta':
                out[k] = v
            else:
                out[k] = to_cuda(v)
        return out
    elif type(batch) is torch.Tensor:
        return batch.cuda(non_blocking=True)
    elif type(batch) is list:
        return [to_cuda(v) for v in batch]
    else:
        return batch


def get_output(output, task):
    """
    Get output prediction in the required range and format
    :param Tensor output: Output tensor
    :param str task: Task
    :return: Tensor
    """

    if task in {'normals'}:
        output = output.permute(0, 2, 3, 1)
        output = (F.normalize(output, p=2, dim=3) + 1.0) * 255 / 2.0

    elif task in {'semseg', 'human_parts'}:
        output = output.permute(0, 2, 3, 1)
        _, output = torch.max(output, dim=3)

    elif task in {'edge'}:
        output = output.permute(0, 2, 3, 1)
        output = torch.sigmoid(output).squeeze(-1) * 255

    elif task in {'sal'}:
        output = output.permute(0, 2, 3, 1)
        output = F.softmax(output, dim=3)[:, :, :, 1] * 255

    elif task in {'depth'}:
        output.clamp_(min=0.)
        output = output.permute(0, 2, 3, 1).squeeze(-1)

    else:
        raise NotImplementedError

    return output


def move_ckpt(ckpt_dict, device):
    for i in range(len(ckpt_dict)):
        for key in ckpt_dict[i].keys():
            ckpt_dict[i][key] = ckpt_dict[i][key].to(device)

    return ckpt_dict


def set_seed(seed):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

#=============================#
#         ARCHITECTURE        #
#=============================#

# -----------------------------
# Lightweight building blocks
# -----------------------------
class ConvBNReLU(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, s=1, p=None, groups=1):
        super().__init__()
        if p is None: p = k // 2
        self.conv = nn.Conv2d(in_ch, out_ch, k, stride=s, padding=p, groups=groups, bias=False)
        self.bn   = nn.BatchNorm2d(out_ch)
        self.act  = nn.ReLU(inplace=True)
    def forward(self, x): return self.act(self.bn(self.conv(x)))

class DWSeparableConv(nn.Module):
    """Depthwise (k×k) + Pointwise (1×1). Keeps ops low for browser."""
    def __init__(self, in_ch, out_ch, k=3, s=1, p=None):
        super().__init__()
        if p is None: p = k // 2
        self.dw = ConvBNReLU(in_ch, in_ch, k=k, s=s, p=p, groups=in_ch)
        self.pw = ConvBNReLU(in_ch, out_ch, k=1, s=1, p=0)
    def forward(self, x): return self.pw(self.dw(x))

# -----------------------------
# Decoder: FPN-Lite (task-specific)
# Takes encoder features [C,2C,4C,8C], outputs a single feature map with 'embed_dim' (=C).
# -----------------------------
class Decoder(nn.Module):
    def __init__(self, input_size, in_dims, embed_dim):
        """
        input_size: (H_enc_out, W_enc_out) — not used strictly here, but kept for API compatibility.
        in_dims: list of 4 ints, e.g. [C, 2C, 4C, 8C]
        embed_dim: int, target channels for fused features (usually C)
        """
        super().__init__()
        C1, C2, C3, C4 = in_dims
        E = embed_dim

        # lateral 1x1 to unify channels
        self.lat4 = nn.Conv2d(C4, E, 1, bias=False)
        self.lat3 = nn.Conv2d(C3, E, 1, bias=False)
        self.lat2 = nn.Conv2d(C2, E, 1, bias=False)
        self.lat1 = nn.Conv2d(C1, E, 1, bias=False)

        # post-fusion depthwise-separable refiners (cheap)
        self.ref3 = DWSeparableConv(E, E, k=3, s=1)
        self.ref2 = DWSeparableConv(E, E, k=3, s=1)
        self.ref1 = DWSeparableConv(E, E, k=3, s=1)

        # final refine at the highest resolution (/4)
        self.final = DWSeparableConv(E, E, k=3, s=1)

    def forward(self, feats):
        # feats: [f1(/4), f2(/8), f3(/16), f4(/32)]
        f1, f2, f3, f4 = feats
        p4 = self.lat4(f4)                         # /32, E

        p3 = self.lat3(f3) + F.interpolate(p4, size=f3.shape[-2:], mode="bilinear", align_corners=False)  # /16
        p3 = self.ref3(p3)

        p2 = self.lat2(f2) + F.interpolate(p3, size=f2.shape[-2:], mode="bilinear", align_corners=False)  # /8
        p2 = self.ref2(p2)

        p1 = self.lat1(f1) + F.interpolate(p2, size=f1.shape[-2:], mode="bilinear", align_corners=False)  # /4
        p1 = self.ref1(p1)

        out = self.final(p1)  # /4, E
        return out

# -----------------------------
# Task Head
# -----------------------------
class Head(nn.Module):
    def __init__(self, dim, out_ch):
        super().__init__()
        # One cheap conv to mix a bit, then 1x1 projection
        self.block = nn.Sequential(
            DWSeparableConv(dim, dim, k=3, s=1),
            nn.Conv2d(dim, out_ch, 1, bias=True),
        )
    def forward(self, x): return self.block(x)

# -----------------------------
# Encoder (Backbone): TinyBackbone
# Produces 4 feature maps: /4, /8, /16, /32 with channels [C, 2C, 4C, 8C]
# -----------------------------
class TinyBackbone(nn.Module):
    def __init__(self, base_channels=24):
        super().__init__()
        C = base_channels

        # Stem: go to /4 before stage features (keeps dims aligned with many decoders)
        self.stem1 = ConvBNReLU(3, C, k=3, s=2)          # /2
        self.stem2 = DWSeparableConv(C, C, k=3, s=2)     # /4

        # Stage 1 (/4): keep C
        self.s1 = nn.Sequential(
            DWSeparableConv(C,   C, k=3, s=1),
            DWSeparableConv(C,   C, k=3, s=1),
        )
        # Stage 2 (/8): 2C
        self.s2 = nn.Sequential(
            DWSeparableConv(C,   2*C, k=3, s=2),
            DWSeparableConv(2*C, 2*C, k=3, s=1),
        )
        # Stage 3 (/16): 4C
        self.s3 = nn.Sequential(
            DWSeparableConv(2*C, 4*C, k=3, s=2),
            DWSeparableConv(4*C, 4*C, k=3, s=1),
        )
        # Stage 4 (/32): 8C
        self.s4 = nn.Sequential(
            DWSeparableConv(4*C, 8*C, k=3, s=2),
            DWSeparableConv(8*C, 8*C, k=3, s=1),
        )

        self.out_channels = C  # base channel count C

    def forward(self, x):
        x = self.stem1(x)  # /2
        x = self.stem2(x)  # /4
        f1 = self.s1(x)    # /4,   C
        f2 = self.s2(f1)   # /8,  2C
        f3 = self.s3(f2)   # /16, 4C
        f4 = self.s4(f3)   # /32, 8C
        # Return list for FPN-like decoders (low->high level)
        return [f1, f2, f3, f4]  # channels: [C, 2C, 4C, 8C]

class MultiDecoderModel(nn.Module):
    """
    Shared encoder + per-task decoders + per-task heads.
    Decoder returns /4 features; we upsample logits to image size outside.
    """
    def __init__(self, backbone: nn.Module, decoders: nn.ModuleDict, heads: nn.ModuleDict, tasks: list):
        super().__init__()
        assert set(decoders.keys()) == set(tasks)
        self.backbone = backbone
        self.decoders = decoders
        self.heads    = heads
        self.tasks    = tasks

    def forward(self, x):
        img_size = x.shape[-2:]
        feats = self.backbone(x)  # list of 4 feature maps
        out = {}
        for t in self.tasks:
            z = self.decoders[t](feats)              # /4, C
            y = self.heads[t](z)                     # /4, out_ch
            out[t] = F.interpolate(y, size=img_size, mode='bilinear', align_corners=False)
        return out

def get_output_num(task, dataname):
    if dataname == 'shapes':
        return SHAPES_OUT_CHANNELS[task]
    else:
        raise NotImplementedError

# -----------------------------
# Wiring into your existing API
# -----------------------------
def get_backbone(backbone_type, backbone_pretrained=False):
    """
    Extend your get_backbone with a 'tiny' option.
    For browser deployment, prefer 'tiny'.
    """
    if backbone_type == 'tiny':
        base_c = 24  # very small; try 24/32 depending on accuracy vs. speed
        backbone = TinyBackbone(base_channels=base_c)
        backbone_channels = base_c  # matches our dims list: [C, 2C, 4C, 8C]
        return backbone, backbone_channels

    raise NotImplementedError(f"Unknown backbone_type: {backbone_type}")

def get_decoder_head(tasks, dataname, backbone_channels):
    # keep your original logic; this matches TinyBackbone outputs
    input_size   = TRAIN_SCALE[dataname]            # e.g., (128,128)
    enc_out_size = (int(input_size[0] / 32), int(input_size[1] / 32))  # not strictly used
    enc_out_dims = [backbone_channels * (2 ** i) for i in range(4)]    # [C,2C,4C,8C]

    decoders = nn.ModuleDict()
    heads    = nn.ModuleDict()
    for task in tasks:
        decoders[task] = Decoder(input_size=enc_out_size,
                                 in_dims=enc_out_dims,
                                 embed_dim=backbone_channels)
        heads[task] = Head(dim=backbone_channels, out_ch=get_output_num(task, dataname))
    return decoders, heads

def build_model(tasks, dataname, backbone_type, backbone_pretrained):
    """
    Initialize the local model
    """

    backbone, backbone_channels = get_backbone(backbone_type, backbone_pretrained)
    decoders, heads = get_decoder_head(tasks, dataname, backbone_channels)
    model = MultiDecoderModel(backbone, decoders, heads, tasks)

    return model


#=============================#
#            LOSS             #
#=============================#
SHAPES_LOSS_CONFIG = {
    'semseg': {
        'loss_function': 'CELoss',
        'weight': 1
    },
    'normals': {
        'loss_function': 'L1Loss',
        'parameters': {
            'normalize': True
        },
        'weight': 10
    },
    'edge': {
        'loss_function': 'BalancedBCELoss',
        'parameters': {
            'pos_weight': 0.95
        },
        'weight': 50
    },
    'depth': {
        'loss_function': 'L1Loss',
        'weight': 1
    },
    'saliency': {
        'loss_function': 'CELoss',
        'parameters': {
            'balanced': True
        },
        'weight': 5
    }
}

class BalancedBCELoss(nn.Module):
    # Edge Detection

    def __init__(self, pos_weight=0.95, ignore_index=255):
        super().__init__()
        self.pos_weight = pos_weight
        self.ignore_index = ignore_index

    def forward(self, output, label):
        mask = (label != self.ignore_index)
        masked_output = torch.masked_select(output, mask)  # 1-d tensor
        masked_label = torch.masked_select(label, mask)  # 1-d tensor

        # pos weight: w, neg weight: 1-w
        w = torch.tensor(self.pos_weight, device=output.device)
        factor = 1. / (1 - w)
        loss = F.binary_cross_entropy_with_logits(masked_output, masked_label, pos_weight=w * factor)
        loss /= factor

        return loss


class CELoss(nn.Module):
    # Semantic Segmentation, Human Parts Segmentation, Saliency Detection

    def __init__(self, balanced=False, ignore_index=255):
        super(CELoss, self).__init__()
        self.ignore_index = ignore_index
        self.balanced = balanced

    def forward(self, output, label):
        label = torch.squeeze(label, dim=1).long()

        if self.balanced:
            mask = (label != self.ignore_index)
            masked_label = torch.masked_select(label, mask)
            assert torch.max(masked_label) < 2  # binary

            num_labels_neg = torch.sum(1.0 - masked_label)
            num_total = torch.numel(masked_label)
            pos_weight = num_labels_neg / num_total
            class_weight = torch.stack((1. - pos_weight, pos_weight), dim=0)
            loss = F.cross_entropy(output, label, weight=class_weight, ignore_index=self.ignore_index, reduction='sum')
        else:
            loss = F.cross_entropy(output, label, ignore_index=self.ignore_index, reduction='sum')

        n_valid = (label != self.ignore_index).sum()
        loss /= max(n_valid, 1)

        return loss


class L1Loss(nn.Module):
    # Normals Estimation, Depth Estimation

    def __init__(self, normalize=False, ignore_index=255):
        super(L1Loss, self).__init__()
        self.normalize = normalize
        self.ignore_index = ignore_index

    def forward(self, output, label):
        if self.normalize:
            # Normalize to unit vector
            output = F.normalize(output, p=2, dim=1)

        mask = (label != self.ignore_index).all(dim=1, keepdim=True)
        masked_output = torch.masked_select(output, mask)
        masked_label = torch.masked_select(label, mask)

        loss = F.l1_loss(masked_output, masked_label, reduction='sum')
        n_valid = torch.sum(mask).item()
        loss /= max(n_valid, 1)

        return loss


def get_loss_functions(task_loss_config):
    """
    Get loss function for each task
    """

    key2loss = {
        "CELoss": CELoss,
        "BalancedBCELoss": BalancedBCELoss,
        "L1Loss": L1Loss,
    }

    # Get loss function for each task
    loss_fx = key2loss[task_loss_config['loss_function']]
    if 'parameters' in task_loss_config:
        loss_ft = loss_fx(**task_loss_config['parameters'])
    else:
        loss_ft = loss_fx()

    return loss_ft

class MultiTaskLoss(nn.Module):
    """
    Multi-Task loss with different loss functions and weights
    """
    def __init__(self, tasks, loss_ft, loss_weights):
        super(MultiTaskLoss, self).__init__()
        assert (set(tasks) == set(loss_ft.keys()))
        assert (set(tasks) == set(loss_weights.keys()))
        self.tasks = tasks
        self.loss_ft = loss_ft
        self.loss_weights = loss_weights

    def forward(self, pred, gt, tasks):
        out = {task: self.loss_ft[task](pred[task], gt[task]) for task in tasks}
        out['total'] = torch.sum(torch.stack([self.loss_weights[t] * out[t] for t in tasks]))

        return out


def get_criterion(dataname, tasks):
    if dataname == 'shapes':
        losses_config = SHAPES_LOSS_CONFIG
    else:
        raise NotImplementedError

    loss_ft = torch.nn.ModuleDict({task: get_loss_functions(losses_config[task]) for task in tasks})
    loss_weights = {task: losses_config[task]['weight'] for task in tasks}

    return MultiTaskLoss(tasks, loss_ft, loss_weights)


#=============================#
#     EVALUATION METRICS      #
#=============================#

class DepthMeter(object):
    def __init__(self, dataname):
        self.dataname = dataname
        self.total_rmses = 0.0
        self.abs_rel = 0.0
        self.n_valid = 0.0
        self.max_depth = 80.0
        self.min_depth = 0.0

    @torch.no_grad()
    def update(self, pred, gt):
        pred, gt = pred.squeeze(), gt.squeeze()

        # Determine valid mask
        mask = torch.logical_and(gt < self.max_depth, gt > self.min_depth)
        self.n_valid += mask.float().sum().item()  # Valid pixels per image

        # Only positive depth values are possible
        # pred = torch.clamp(pred, min=1e-9)
        gt[gt <= 0] = 1e-9
        pred[pred <= 0] = 1e-9

        rmse_tmp = torch.pow(gt[mask] - pred[mask], 2)
        self.total_rmses += rmse_tmp.sum().item()
        self.abs_rel += (torch.abs(gt[mask] - pred[mask]) / gt[mask]).sum().item()

    def reset(self):
        self.total_rmses = 0.0
        self.abs_rel = 0.0
        self.n_valid = 0.0

    def get_score(self):
        if self.dataname == 'nyud':
            eval_result = {'RMSE': np.sqrt(self.total_rmses / self.n_valid)}
        else:
            raise NotImplementedError

        return eval_result
    
class EdgeMeter(object):

    def __init__(self, dataname, ignore_index=255):
        if dataname == 'pascalcontext':
            pos_weight = 0.95
        elif dataname == 'nyud':
            pos_weight = 0.95
        else:
            raise NotImplementedError

        self.loss = 0
        self.n = 0
        self.loss_function = BalancedBCELoss(pos_weight=pos_weight, ignore_index=ignore_index)
        self.ignore_index = ignore_index

    @torch.no_grad()
    def update(self, pred, gt):
        pred, gt = pred.squeeze(), gt.squeeze()
        valid_mask = (gt != self.ignore_index)
        pred = pred[valid_mask]
        gt = gt[valid_mask]

        pred = pred.float().squeeze() / 255.
        loss = self.loss_function(pred, gt).item()
        numel = gt.numel()
        self.n += numel
        self.loss += numel * loss

    def reset(self):
        self.loss = 0
        self.n = 0

    def get_score(self):
        eval_dict = {'loss': (self.loss / self.n)}

        return eval_dict

class NormalsMeter(object):

    def __init__(self, ignore_index=255):
        self.ignore_index = ignore_index
        self.sum_deg_diff = 0
        self.total = 0

    def normalize_tensor(self, input_tensor, dim):
        norm = torch.norm(input_tensor, p='fro', dim=dim, keepdim=True)
        zero_mask = (norm == 0)
        norm[zero_mask] = 1
        out = input_tensor.div(norm)
        out[zero_mask.expand_as(out)] = 0
        return out

    @torch.no_grad()
    def update(self, pred, gt):
        pred = pred.permute(0, 3, 1, 2)  # [B, C, H, W]
        pred = 2 * pred / 255 - 1
        valid_mask = (gt != self.ignore_index).all(dim=1)

        pred = self.normalize_tensor(pred, dim=1)
        gt = self.normalize_tensor(gt, dim=1)
        deg_diff = torch.rad2deg(2 * torch.atan2(torch.norm(pred - gt, dim=1), torch.norm(pred + gt, dim=1)))
        deg_diff = torch.masked_select(deg_diff, valid_mask)

        self.sum_deg_diff += torch.sum(deg_diff).item()
        self.total += deg_diff.numel()

    def get_score(self):
        eval_result = {'mErr': (self.sum_deg_diff / self.total)}

        return eval_result
    
class SemsegMeter(object):
    def __init__(self, dataname, ignore_index=255):
        if dataname == 'pascalcontext':
            n_classes = 20
            has_bg = True
        elif dataname == 'nyud':
            n_classes = 40
            has_bg = False
        else:
            raise NotImplementedError

        self.ignore_index = ignore_index
        self.n_classes = n_classes + int(has_bg)
        self.tp = [0] * self.n_classes
        self.fp = [0] * self.n_classes
        self.fn = [0] * self.n_classes

    @torch.no_grad()
    def update(self, pred, gt):
        pred, gt = pred.squeeze(), gt.squeeze()
        valid = (gt != self.ignore_index)

        for i_part in range(self.n_classes):
            tmp_gt = (gt == i_part)
            tmp_pred = (pred == i_part)
            self.tp[i_part] += torch.sum(tmp_gt & tmp_pred & valid).item()
            self.fp[i_part] += torch.sum(~tmp_gt & tmp_pred & valid).item()
            self.fn[i_part] += torch.sum(tmp_gt & ~tmp_pred & valid).item()

    def reset(self):
        self.tp = [0] * self.n_classes
        self.fp = [0] * self.n_classes
        self.fn = [0] * self.n_classes

    def get_score(self):
        jac = [0] * self.n_classes
        for i_part in range(self.n_classes):
            jac[i_part] = float(self.tp[i_part]) / max(float(self.tp[i_part] + self.fp[i_part] + self.fn[i_part]), 1e-8)

        eval_result = {'mIoU': (np.mean(jac) * 100)}

        return eval_result

class SaliencyMeter(object):

    def __init__(self, ignore_index=255, threshold_step=0.05, beta=0.3):
        self.ignore_index = ignore_index
        self.beta = beta
        self.thresholds = torch.arange(threshold_step, 1, threshold_step)
        self.true_positives = torch.zeros(len(self.thresholds))
        self.predicted_positives = torch.zeros(len(self.thresholds))
        self.actual_positives = torch.zeros(len(self.thresholds))
        self.ious = []

    @torch.no_grad()
    def update(self, preds, targets):
        preds = preds.float() / 255.

        if targets.shape[1] == 1:
            targets = targets.squeeze(1)

        assert preds.shape == targets.shape

        for i in range(preds.size(0)):
            pred = preds[i]
            target = targets[i]
            valid_mask = (target != self.ignore_index)
            iou = np.zeros(len(self.thresholds))

            for idx, thresh in enumerate(self.thresholds):
                # threshold probablities
                f_pred = (pred >= thresh).long()
                f_target = target.long()

                f_pred = torch.masked_select(f_pred, valid_mask)
                f_target = torch.masked_select(f_target, valid_mask)

                self.true_positives[idx] += torch.sum(f_pred * f_target).item()
                self.predicted_positives[idx] += torch.sum(f_pred).item()
                self.actual_positives[idx] += torch.sum(f_target).item()

                iou[idx] = torch.sum(f_pred & f_target).item() / torch.sum(f_pred | f_target).item()

            self.ious.append(iou)

    def get_score(self):
        """
        Computes F-scores over state and returns the max.
        """
        precision = self.true_positives.float() / (self.predicted_positives + 1e-8)
        recall = self.true_positives.float() / (self.actual_positives + 1e-8)

        num = (1 + self.beta) * precision * recall
        denom = self.beta * precision + recall

        # For the rest we need to take care of instances where the denom can be 0
        # for some classes which will produce nans for that class
        fscore = num / (denom + 1e-8)
        fscore[fscore != fscore] = 0

        mIoUs = np.mean(np.array(self.ious), axis=0)

        eval_result = {'maxF': (fscore.max().item() * 100), 'mIoU': (mIoUs.max() * 100)}

        return eval_result
    
#=============================#
#      TRAIN UTILITIES        #    
#=============================#
def get_single_task_meter(dataname, task):
    """
    Retrieve a meter to measure the single-task performance
    """

    if task == 'semseg':
        return SemsegMeter(dataname)

    elif task == 'normals':
        return NormalsMeter()

    elif task == 'sal':
        return SaliencyMeter()

    elif task == 'depth':
        return DepthMeter(dataname)

    elif task == 'edge':  # Single task performance meter uses the loss (True evaluation is based on seism evaluation)
        return EdgeMeter(dataname)

    else:
        raise NotImplementedError
    
class PerformanceMeter(object):
    """
    A general performance meter which shows performance across one or more tasks
    """

    def __init__(self, dataname, tasks):
        self.tasks = tasks
        self.meters = {t: get_single_task_meter(dataname, t) for t in self.tasks}

    def reset(self):
        for t in self.tasks:
            self.meters[t].reset()

    def update(self, pred, gt):
        for t in self.tasks:
            self.meters[t].update(pred[t], gt[t])

    def get_score(self):
        eval_dict = {}
        for t in self.tasks:
            eval_dict[t] = self.meters[t].get_score()

        return eval_dict
    
def local_train(idx, cr, local_epochs, tasks, train_dl, model, optimizer, scheduler, criterion, scaler, train_loss,
                local_rank, fp16, **args):
    """
    Train local_epochs on the client model
    """

    model.train()

    for epoch in range(local_epochs):
        train_dl.sampler.set_epoch(cr * local_epochs + epoch)
        for batch in tqdm(train_dl,
                          desc="CR %d Local Epoch %d Net %d Task: %s" % (cr, epoch, idx, ",".join(tasks)),
                          disable=(local_rank != 0)):
            optimizer.zero_grad()
            batch = to_cuda(batch)
            images = batch['image']

            with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=fp16):
                outputs = model(images)
                loss_dict = criterion(outputs, batch, tasks)

            # Log loss values
            for task in tasks:
                loss_value = loss_dict[task].detach().item()
                batch_size = outputs[task].size(0)
                train_loss[task].update(loss_value / batch_size, batch_size)

            scaler.scale(loss_dict['total']).backward()
            scaler.step(optimizer)
            scaler.update()

        scheduler.step(cr * local_epochs + epoch)


def eval_metric(tasks, dataname, val_dl, model, idx, **args):
    """
    Evaluate client model
    """

    performance_meter = PerformanceMeter(dataname, tasks)

    model.eval()
    with torch.no_grad():
        for batch in tqdm(val_dl, desc="Evaluating Net %d Task: %s" % (idx, ",".join(tasks))):
            batch = to_cuda(batch)
            images = batch['image']
            outputs = model.module(images)
            performance_meter.update({t: get_output(outputs[t], t) for t in tasks}, batch)

    eval_results = performance_meter.get_score()

    results_dict = {}
    for task in tasks:
        for key in eval_results[task]:
            results_dict['eval/' + str(idx) + '_' + task + '_' + key] = eval_results[task][key]

    return results_dict


#=============================#
#           CLIENT            #    
#=============================#

def get_dataset(dataname, train, tasks, transform, dataidxs=None, local_rank=0):
    """
    Get the dataset
    """

    if local_rank == 0:
        if train:
            print("Get training dataset for %s on %s" % (dataname, ", ".join(tasks)))
        else:
            print("Get validation dataset for %s on %s" % (dataname, ", ".join(tasks)))

    if dataname == 'shapes':
        database = SHAPES(train=train, transform=transform, tasks=tasks, dataidxs=dataidxs)
    else:
        raise NotImplementedError("'dataname': Choose among 'shapes'!")

    return database


def get_dataloader(train, configs, dataset, sampler=None):
    """
    Get the dataloader from dataset
    """
    if train:
        dataloader = DataLoader(dataset,
                                batch_size=configs['tr_batch'],
                                drop_last=True,
                                num_workers=configs['nworkers'],
                                collate_fn=collate_mil,
                                pin_memory=True,
                                sampler=sampler)
    else:
        dataloader = DataLoader(dataset,
                                batch_size=configs['val_batch'],
                                shuffle=False,
                                drop_last=False,
                                num_workers=configs['nworkers'],
                                collate_fn=collate_mil,
                                pin_memory=True)
    return dataloader

# ---------------------
# Lightweight utilities
# ---------------------
class IdentityDDP(torch.nn.Module):
    """
    Tiny wrapper to keep a `.module` attribute so code written for DDP keeps working.
    Forward/params just delegate to the wrapped module.
    """
    def __init__(self, module: torch.nn.Module):
        super().__init__()
        self.module = module
    def forward(self, *a, **kw):
        return self.module(*a, **kw)

# --- tiny warmup+cosine scheduler (no external deps) -------------------------
class WarmupCosineLR:
    """
    Minimal epoch-based warmup + cosine LR schedule.
    Call `step(epoch_idx)` once per epoch (0-based).
    """
    def __init__(self, optimizer, max_epochs, base_lr, min_lr=1.25e-6, warmup_epochs=0, warmup_init_lr=1.25e-7):
        self.opt = optimizer
        self.max_epochs = max(1, int(max_epochs))
        self.warmup = max(0, int(warmup_epochs))
        self.base_lr = float(base_lr)
        self.min_lr = float(min_lr)
        self.warmup_init_lr = float(warmup_init_lr)

    def _set_lr(self, lr):
        for pg in self.opt.param_groups:
            pg["lr"] = lr

    def step(self, epoch):
        # epoch: 0..max_epochs-1
        if epoch < self.warmup and self.warmup > 0:
            # linear warmup
            t = (epoch + 1) / self.warmup  # 0->1
            lr = self.warmup_init_lr + t * (self.base_lr - self.warmup_init_lr)
        else:
            # cosine decay over the remaining epochs
            if self.max_epochs == self.warmup:
                lr = self.min_lr
            else:
                t = (epoch - self.warmup) / max(1, (self.max_epochs - self.warmup))
                lr = self.min_lr + 0.5 * (self.base_lr - self.min_lr) * (1.0 + math.cos(math.pi * t))
        self._set_lr(lr)
        return lr


# --- no-op autocast for CPU ---------------------------------------------------
def autocast_if_available(enabled: bool):
    """
    Returns a context manager: CUDA autocast if available+enabled, else nullcontext().
    Use in your train loop as:  `with client['autocast'](): ...`
    """
    if enabled and torch.cuda.is_available():
        return torch.cuda.amp.autocast
    return nullcontext


# --- main: browser-friendly client builder -----------------------------------
def get_clients_browser_friendly(args, model_config, client_configs, device: torch.device | None = None):
    """
    Build lightweight 'clients' without DDP/SyncBN/timm; suitable as a PoC
    that you can later port to JavaScript/TF.js.

    Returns:
        all_clients: list[dict]
        n_decoders:  int  (sum of number of task decoders across clients)
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    all_clients = []
    n_decoders = 0

    for dataname, client_config in client_configs.items():
        net_task_dataidx_map = client_config["net_task_dataidx_map"]
        n_clients            = client_config["n_clients"]

        for idx in range(n_clients):
            task_list = net_task_dataidx_map[idx]["task_list"]
            dataidxs  = net_task_dataidx_map[idx]["dataidx"]

            # --- datasets / dataloaders (no distributed sampler) -------------
            train_ds_local = get_dataset(
                dataname=dataname,
                train=True,
                tasks=task_list,
                transform=client_config.get("train_transforms"),
                dataidxs=dataidxs,
                local_rank=0,  # ignored by your dataset if not needed
            )
            val_ds_local = get_dataset(
                dataname=dataname,
                train=False,
                tasks=task_list,
                transform=client_config.get("val_transforms"),
                local_rank=0,
            )

            # keep worker=0/pin_memory=False for maximum portability
            train_dl_local = get_dataloader(
                train=True,
                configs=client_config,
                dataset=train_ds_local,
                sampler=None,  # shuffle handled inside your get_dataloader/configs
            )
            val_dl_local = get_dataloader(
                train=False,
                configs=client_config,
                dataset=val_ds_local,
                sampler=None,
            )

            # --- model (no SyncBatchNorm / no DDP) ---------------------------
            model = build_model(task_list, dataname, **model_config).to(device)
            model = IdentityDDP(model)  # preserves `.module`

            # --- optimizer ---------------------------------------------------
            opt_name = client_config.get("optimizer", "adamw").lower()
            lr       = float(client_config.get("lr", 3e-4))
            wd       = float(client_config.get("weight_decay", 0.0))

            if opt_name == "sgd":
                optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=wd, nesterov=True)
            elif opt_name == "adamw":
                optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
            else:
                raise NotImplementedError(f"Invalid optimizer {opt_name!r}")

            # --- scheduler: warmup + cosine (epoch-based) --------------------
            local_epochs   = int(client_config.get("local_epochs", 1))
            max_rounds     = int(getattr(args, "max_rounds", 1))
            max_epochs_tot = max_rounds * local_epochs
            warmup_epochs  = int(client_config.get("warmup_epochs", 0))
            scheduler = WarmupCosineLR(
                optimizer=optimizer,
                max_epochs=max_epochs_tot,
                base_lr=lr,
                min_lr=float(client_config.get("min_lr", 1.25e-6)),
                warmup_epochs=warmup_epochs,
                warmup_init_lr=float(client_config.get("warmup_lr_init", 1.25e-7)),
            )

            # --- loss / criterion -------------------------------------------
            criterion = get_criterion(dataname, task_list).to(device)

            # --- mixed precision (optional, single device) -------------------
            use_amp = bool(getattr(args, "fp16", False)) and torch.cuda.is_available()
            scaler  = torch.cuda.amp.GradScaler(enabled=use_amp) if torch.cuda.is_available() else None
            # expose an autocast context manager for your training loop
            autocast_ctx = lambda: autocast_if_available(use_amp)

            # --- pack the client dict ---------------------------------------
            client = {
                "id": f"{dataname}_{idx}",
                "tasks": task_list,
                "dataname": dataname,
                "train_dl": train_dl_local,
                "val_dl": val_dl_local,
                "local_epochs": local_epochs,
                "model": model,
                "optimizer": optimizer,
                "scheduler": scheduler,
                "criterion": criterion,
                "scaler": scaler,          # None on CPU; ok to check for None in your loop
                "autocast": autocast_ctx,  # use: `with client['autocast'](): ...`
                "device": device,
            }

            all_clients.append(client)
            n_decoders += len(task_list)

    return all_clients, n_decoders


#=============================#
#           SERVER            #
#=============================#

# Server side
# HCA2 Algorithm!!
class HyperAggWeight(nn.Module):
    """Hyper Conflict-Averse Aggregation for encoders"""

    def __init__(self, K, init_alpha=1):
        super(HyperAggWeight, self).__init__()
        self.K = K

        # define parameters
        self.alpha = nn.Parameter(torch.ones(K) * init_alpha)

    def forward(self, flatten_last_param_list, flatten_delta, flatten_delta_update):
        flatten_new_param_list = copy.deepcopy(flatten_last_param_list)
        assert self.K == len(flatten_last_param_list)  # number of encoders

        # cut weight into [0, 1]
        alpha = torch.clamp(self.alpha, 0, 1)
        for i in range(self.K):
            flatten_new_param_list[i] += (flatten_delta[i] + alpha[i] * flatten_delta_update)

        return flatten_new_param_list


class HyperCrossAttention(nn.Module):
    """Hyper Cross-Attention Aggregation for decoders"""

    def __init__(self, model, K, init_beta=1):
        super(HyperCrossAttention, self).__init__()
        self.K = K

        # get layer names
        self.layer_names = []
        for name, _ in model.named_parameters():
            self.layer_names.append(".".join(name.split('.')[1:-1]))
        self.layer_names = sorted(set(self.layer_names))
        self.beta_names = [name.replace('.', '_') for name in self.layer_names]

        # define parameters
        self.beta = nn.ParameterDict()
        for name in self.beta_names:
            self.beta[name] = nn.Parameter(torch.ones(K) * init_beta)  # layer-wise

    def forward(self, last_param_dict_list, delta_dict_list):
        new_param_dict_list = copy.deepcopy(last_param_dict_list)
        assert self.K == len(last_param_dict_list)  # number of decoders

        for name in self.layer_names:
            # cut weight into [0, 1]
            layer_beta = torch.clamp(self.beta[name.replace('.', '_')], 0, 1)
            # get keys of each parameter in the layer (weight & bias)
            layer_keys = []
            for key in delta_dict_list[0].keys():
                if name in key:
                    layer_keys.append(key)

            for key in layer_keys:
                cross_delta = torch.stack([delta_dict_list[j][key].reshape(-1) for j in range(self.K)])
                for i in range(self.K):
                    self_delta = delta_dict_list[i][key].reshape(1, -1)
                    cross_attn_delta = CrossAttention(self_delta, cross_delta, cross_delta)

                    beta = layer_beta[i]
                    ori_shape = delta_dict_list[i][key].shape
                    new_delta = delta_dict_list[i][key] + beta * cross_attn_delta.reshape(ori_shape)
                    new_param_dict_list[i][key] += new_delta

        return new_param_dict_list

def CrossAttention(q, k, v):
    scale = q.size(-1)**-0.5
    attn = (q @ k.transpose(-2, -1)) * scale
    attn = nn.Softmax(dim=-1)(attn)
    out = attn @ v

    return out


#=============================#
#         AGGREGATE           #
#=============================#

def get_encoder_keys(all_keys):
    """
    Get keys of encoder parameters
    """

    return list(filter(lambda x: 'backbone' in x, all_keys))


def get_decoder_keys(all_keys):
    """
    Get keys of decoder parameters
    """

    return list(filter(lambda x: 'decoders' in x, all_keys))


def get_model_soup(param_dict_list):
    """
    Get the average of parameters in list
    """

    soup_param_dict = {}
    layers = param_dict_list[0].keys()
    for layer in layers:
        soup_param_dict[layer] = torch.mean(
            torch.stack(
                [param_dict_list[i][layer] for i in range(len(param_dict_list))]
            ),
            dim=0,
        )

    return soup_param_dict


def get_delta_dict_list(param_dict_list, last_param_dict_list):
    """
    Get the difference between current and last parameters
    """

    # a list of length N, each element is a dict of delta parameters
    delta_dict_list = []
    layers = param_dict_list[0].keys()
    for i in range(len(param_dict_list)):
        delta_dict_list.append({})
        for layer in layers:
            delta_dict_list[i][layer] = (
                param_dict_list[i][layer] - last_param_dict_list[i][layer]
            )

    return delta_dict_list


def get_encoder_params(all_clients, ckpt):
    """
    Get encoder parameters from checkpoint
    """

    # encoder_param_list: a list of length n_st, each element is a dict of encoder parameters
    all_name_keys = [
        name for name, _ in all_clients[0]['model'].module.named_parameters()
    ]
    encoder_keys = get_encoder_keys(all_name_keys)
    encoder_param_dict_list = []
    layers = []
    shapes = []

    for model_idx in range(len(ckpt)):
        param_dict = {}
        for key in encoder_keys:
            # key=prefix+'.'+layer
            prefix, layer = key.split('.', 1)
            param_dict[layer] = ckpt[model_idx][key]
        encoder_param_dict_list.append(param_dict)

    # Get layers and shapes (same for all encoders)
    for key in encoder_keys:
        layers.append(key.split('.', 1)[1])
        shapes.append(ckpt[0][key].shape)

    return encoder_param_dict_list, encoder_keys, layers, shapes


def get_decoder_params(all_clients, ckpt):
    """
    Get decoder parameters from checkpoint
    """

    N = len(all_clients)
    n_st = sum(
        [len(all_clients[i]['tasks']) == 1 for i in range(N)]
    )  # number of st clients
    K = sum([len(all_clients[i]['tasks']) for i in range(N)])  # number of decoders

    decoder_keys = []
    layers = []
    shapes = []

    for idx in range(N):
        all_name_keys = [
            key for key, _ in all_clients[idx]['model'].module.named_parameters()
        ]
        decoder_keys += get_decoder_keys(all_name_keys)
    decoder_keys = list(set(decoder_keys))

    decoder_param_dict_list = []
    decoders_prefix = []
    # st client decoders
    for model_idx in range(n_st):
        assert len(all_clients[model_idx]['tasks']) == 1
        param_dict = {}
        for key in decoder_keys:
            if key in ckpt[model_idx].keys():
                # key=prefix+'.'+layer
                prefix = (
                    key.split('.', 2)[0] + '.' + key.split('.', 2)[1]
                )  # decoders.task
                layer = key.split('.', 2)[2]
                param_dict[layer] = ckpt[model_idx][key]

                if model_idx == 0:
                    layers.append(layer)
                    shapes.append(ckpt[0][key].shape)

        decoders_prefix.append(prefix)
        decoder_param_dict_list.append(param_dict)

    # mt client decoders
    for model_idx in range(n_st, N):
        prefix_list = []  # decoder prefixs in one mt client
        for task in all_clients[model_idx]['tasks']:
            prefix_list.append('decoders.' + task)
        prefix_list = sorted((prefix_list))  # keep the order

        for i, prefix in enumerate(prefix_list):
            # Get each task-specific decoder
            param_dict = {}
            for key in decoder_keys:
                if key in ckpt[model_idx].keys() and prefix in key:
                    layer = key.split('.', 2)[2]
                    param_dict[layer] = ckpt[model_idx][key]

                    if model_idx == 0 and i == 0:
                        layers.append(layer)
                        shapes.append(ckpt[0][key].shape)

            decoder_param_dict_list.append(param_dict)
        decoders_prefix += prefix_list

    assert len(decoders_prefix) == K
    assert len(decoder_param_dict_list) == K

    return decoder_param_dict_list, decoders_prefix, decoder_keys, layers, shapes


def get_ca_delta(flatten_delta_list, alpha, rescale=1):
    """
    Solve for aggregated conflict-averse delta
    """

    N = len(flatten_delta_list)
    grads = torch.stack(flatten_delta_list).t()  # [d , N]
    GG = grads.t().mm(grads).cpu()  # [N, N]
    g0_norm = (GG.mean() + 1e-8).sqrt()

    x_start = np.ones(N) / N
    bnds = tuple((0, 1) for x in x_start)
    cons = {'type': 'eq', 'fun': lambda x: 1 - sum(x)}
    A = GG.numpy()
    b = x_start.copy()
    c = (alpha * g0_norm + 1e-8).item()

    def objfn(x):
        return (
            x.reshape(1, -1).dot(A).dot(b.reshape(-1, 1))
            + c * np.sqrt(x.reshape(1, -1).dot(A).dot(x.reshape(-1, 1)) + 1e-8)
        ).sum()

    res = minimize(objfn, x_start, bounds=bnds, constraints=cons)
    ww = torch.Tensor(res.x).to(grads.device)
    gw = (grads * ww.reshape(1, -1)).sum(1)
    gw_norm = gw.norm()
    lmbda = c / (gw_norm + 1e-8)
    g = grads.mean(1) + lmbda * gw
    if rescale == 0:
        final_update = g
    elif rescale == 1:
        final_update = g / (1 + alpha**2)
    else:
        final_update = g / (1 + alpha)

    return final_update


def flatten_param(param_dict_list, layers):
    """
    Flatten a dict of parameters into a vector
    """

    flatten_list = [
        torch.cat([param_dict_list[idx][layer].flatten() for layer in layers])
        for idx in range(len(param_dict_list))
    ]
    assert len(flatten_list[0].shape) == 1

    return flatten_list


def unflatten_param(flatten_list, shapes, layers):
    """
    Unflatten a vector into a dict of parameters
    """

    param_dict_list = []
    for model_idx in range(len(flatten_list)):
        start = 0
        param_dict_list.append({})
        for layer, shape in zip(layers, shapes):
            end = start + np.prod(shape)
            param_dict_list[model_idx][layer] = flatten_list[model_idx][
                start:end
            ].reshape(shape)
            start = end

    return param_dict_list


def aggregate(
    all_clients,
    save_ckpt,
    last_ckpt,
    hyperweight=None,
    encoder_agg='none',
    decoder_agg='none',
    ca_c=0.4,
):
    '''
    Main aggregation function after assigned epochs of local training
    '''
    assert len(all_clients) == len(save_ckpt)
    N = len(all_clients)
    n_st = sum([len(client['tasks']) == 1 for client in all_clients])
    n_mt_tasks = [
        len(all_clients[i]['tasks']) for i in range(n_st, N)
    ]  # number of tasks for each mt client

    if encoder_agg == 'none' and decoder_agg == 'none':
        return  # no aggregation

    update_ckpt = copy.deepcopy(save_ckpt)  # store updated parameters

    # Get encoder parameter list
    encoder_param_list, encoder_keys, enc_layers, enc_shapes = get_encoder_params(
        all_clients, save_ckpt
    )

    # Encoder agg
    if encoder_agg == 'none':
        del encoder_param_list
        pass

    elif encoder_agg in ['fedavg']:
        new_encoder_param = get_model_soup(encoder_param_list)

        for model_idx in range(N):
            for key in encoder_keys:
                layer = key.split('.', 1)[1]
                update_ckpt[model_idx][key] = new_encoder_param[layer]

        del encoder_param_list, new_encoder_param

    elif encoder_agg in ['conflict_averse']:
        last_encoder_param_list, _, _, _ = get_encoder_params(all_clients, last_ckpt)
        encoder_delta_list = get_delta_dict_list(
            encoder_param_list, last_encoder_param_list
        )

        # Flatten
        flatten_last_encoder = flatten_param(last_encoder_param_list, enc_layers)
        del last_encoder_param_list
        flatten_encoder_delta = flatten_param(encoder_delta_list, enc_layers)
        del encoder_delta_list

        # Solve for aggregated conflict-averse delta
        flatten_delta_update = get_ca_delta(
            flatten_encoder_delta, ca_c
        )  # flattened tensor

        # Homo aggregation
        group = [0]
        homo_avg = flatten_encoder_delta[0]
        i = 1
        while i < N:
            if (
                all_clients[i]['dataname'] == all_clients[i - 1]['dataname']
                and all_clients[i]['tasks'] == all_clients[i - 1]['tasks']
            ):
                homo_avg += flatten_encoder_delta[i]
                group.append(i)
            else:
                homo_avg /= len(group)
                for j in group:
                    flatten_encoder_delta[j] = homo_avg
                group = [i]
                homo_avg = flatten_encoder_delta[i]
            i += 1
        homo_avg /= len(group)
        for j in group:
            flatten_encoder_delta[j] = homo_avg

        # Personalized update
        assert hyperweight['enc'] is not None
        flatten_new_encoder = hyperweight['enc'](
            flatten_last_encoder, flatten_encoder_delta, flatten_delta_update
        )
        # Record output of hyperweight for backprop
        hyperweight['last_enc_output'] = flatten_new_encoder

        del flatten_last_encoder, flatten_encoder_delta, flatten_delta_update

        new_encoder_param_list = unflatten_param(
            flatten_new_encoder, enc_shapes, enc_layers
        )

        for model_idx in range(N):
            for key in encoder_keys:
                layer = key.split('.', 1)[1]
                update_ckpt[model_idx][key] = new_encoder_param_list[model_idx][layer]

        del new_encoder_param_list

    else:
        raise NotImplementedError

    # Get decoder parameter list and prefix
    decoder_param_list, decoders_prefix, decoder_keys, dec_layers, dec_shapes = (
        get_decoder_params(all_clients, save_ckpt)
    )

    # Decoder agg
    if decoder_agg == 'none':
        del decoder_param_list
        pass

    elif decoder_agg in ['fedavg']:
        new_decoder_param = get_model_soup(decoder_param_list)

        for i, prefix in enumerate(decoders_prefix):
            # first st clients then mt clients
            if i >= n_st:
                model_idx = n_st + (i - n_st) // (n_mt_tasks[0])
            else:
                model_idx = i

            for layer in dec_layers:
                update_ckpt[model_idx][prefix + '.' + layer] = new_decoder_param[layer]

        del decoder_param_list, new_decoder_param

    elif decoder_agg in ['cross_attention']:
        assert hyperweight['dec'] is not None
        last_decoder_param_list, _, _, _, _ = get_decoder_params(all_clients, last_ckpt)
        decoder_delta_list = get_delta_dict_list(
            decoder_param_list, last_decoder_param_list
        )

        # Personalized update
        new_decoder_param_list = hyperweight['dec'](
            last_decoder_param_list, decoder_delta_list
        )
        # Record output of hyperweight for backprop
        hyperweight['last_dec_output'] = new_decoder_param_list

        for i, (prefix, new_decoder_param) in enumerate(
            zip(decoders_prefix, new_decoder_param_list)
        ):
            # first st clients then mt clients
            if i >= n_st:
                tmp = i - n_st
                k = 0
                while tmp >= n_mt_tasks[k]:
                    tmp -= n_mt_tasks[k]
                    k += 1
                model_idx = n_st + k
            else:
                model_idx = i

            for layer in new_decoder_param.keys():
                update_ckpt[model_idx][prefix + '.' + layer] = new_decoder_param[layer]

        del last_decoder_param_list, decoder_delta_list

    else:
        raise NotImplementedError

    # Update all models
    update_ckpt = move_ckpt(update_ckpt, 'cuda')
    for model_idx in range(N):
        all_clients[model_idx]['model'].module.load_state_dict(update_ckpt[model_idx])

    del update_ckpt


def update_hyperweight(all_nets, hyperweight, save_ckpt, last_ckpt):
    '''
    Update hyperweights with corresponding delta of encoder and decoder parameters
    '''
    if 'enc' in hyperweight.keys():
        # Get encoder parameter list and prefix
        encoder_param_list, encoder_keys, enc_layers, enc_shapes = get_encoder_params(
            all_nets, save_ckpt
        )
        last_encoder_param_list, _, _, _ = get_encoder_params(all_nets, last_ckpt)

        # Calculate difference between current and last encoder parameters
        diff_list = get_delta_dict_list(last_encoder_param_list, encoder_param_list)
        flatten_diff = flatten_param(diff_list, enc_layers)

        # Update hyperweight
        hyperweight['enc'].train()
        optimizer = hyperweight['enc_optimizer']
        optimizer.zero_grad()

        torch.autograd.backward(
            hyperweight['last_enc_output'], flatten_diff, retain_graph=True
        )

        optimizer.step()

    if 'dec' in hyperweight.keys():
        # Get decoder parameter list and prefix
        decoder_param_list, decoders_prefix, decoder_keys, dec_layers, dec_shapes = (
            get_decoder_params(all_nets, save_ckpt)
        )
        last_decoder_param_list, last_decoders_prefix, _, _, _ = get_decoder_params(
            all_nets, last_ckpt
        )
        assert decoders_prefix == last_decoders_prefix

        # Calculate difference between current and last decoder parameters
        diff_list = get_delta_dict_list(last_decoder_param_list, decoder_param_list)

        # Update hyperweight
        hyperweight['dec'].train()
        optimizer = hyperweight['dec_optimizer']
        optimizer.zero_grad()

        for i in range(len(decoder_param_list)):
            # Construct dict of parameters into list
            last_output = list(
                map(lambda x: hyperweight['last_dec_output'][i][x], dec_layers)
            )
            diff_param = list(map(lambda x: diff_list[i][x], dec_layers))
            torch.autograd.backward(last_output, diff_param, retain_graph=True)

        optimizer.step()


#=============================#
#            MAIN             #
#=============================#

# ---------------------
# Main training (single process)
# ---------------------
def main(args, all_clients, hyperweight=None, device=None):
    device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
    N = len(all_clients)

    # Setup loss meters
    train_loss, val_loss = {}, {}
    for idx in range(N):
        train_loss[idx] = {}
        val_loss[idx] = {}
        for task in all_clients[idx]['tasks']:
            train_loss[idx][task] = RunningMeter()
            val_loss[idx][task] = RunningMeter()

    # Save last_ckpt (unwrap .module if present)
    last_ckpt = []
    for idx in range(N):
        sd = copy.deepcopy(all_clients[idx]['model'].module.state_dict())
        last_ckpt.append(sd)
    if args.save_vram:
        last_ckpt = move_ckpt(last_ckpt, 'cpu')
    save_ckpt = copy.deepcopy(last_ckpt)

    # Create hyperweight logs
    if args.encoder_agg == "conflict_averse":
        enc_hw = hyperweight['enc'].module if isinstance(hyperweight['enc'], IdentityDDP) else hyperweight['enc']
        alpha = enc_hw.alpha.detach().cpu().numpy().tolist()
        with open(os.path.join(args.exp_dir, 'enc_alpha.txt'), 'w') as f:
            f.write(str(alpha) + '\n')

    if args.decoder_agg == "cross_attention":
        dec_hw = hyperweight['dec'].module if isinstance(hyperweight['dec'], IdentityDDP) else hyperweight['dec']
        beta = dec_hw.beta
        beta_list = [beta[key].detach().cpu().numpy().tolist() for key in dec_hw.beta_names]
        with open(os.path.join(args.exp_dir, 'dec_beta.txt'), 'w') as f:
            f.write(str(dec_hw.beta_names) + '\n')
            f.write(str(beta_list) + '\n')

    # Rounds
    for cr in range(args.max_rounds):
        start_time = time.time()
        logs = {}
        # Local training per client
        for idx in range(N):
            local_train(
                idx=idx,
                cr=cr,
                train_loss=train_loss[idx],
                local_rank=0,            # kept for API compatibility
                fp16=args.fp16,
                **all_clients[idx]
            )
            train_stats = get_loss_metric(train_loss[idx], all_clients[idx]['tasks'], 'train', idx)
            logs.update(train_stats)

        # Update save_ckpt
        for idx in range(N):
            save_ckpt[idx] = copy.deepcopy(all_clients[idx]['model'].module.state_dict())
        if args.save_vram:
            save_ckpt = move_ckpt(save_ckpt, 'cpu')

        # Update hyperweight (from CR>0)
        if cr > 0:
            update_hyperweight(all_clients, hyperweight, save_ckpt, last_ckpt)
            if args.encoder_agg == "conflict_averse":
                enc_hw = hyperweight['enc'].module if isinstance(hyperweight['enc'], IdentityDDP) else hyperweight['enc']
                alpha = enc_hw.alpha.detach().cpu().numpy().tolist()
                with open(os.path.join(args.exp_dir, 'enc_alpha.txt'), 'a') as f:
                    f.write(str(alpha) + '\n')
            if args.decoder_agg == "cross_attention":
                dec_hw = hyperweight['dec'].module if isinstance(hyperweight['dec'], IdentityDDP) else hyperweight['dec']
                beta = dec_hw.beta
                beta_list = [beta[key].detach().cpu().numpy().tolist() for key in dec_hw.beta_names]
                with open(os.path.join(args.exp_dir, 'dec_beta.txt'), 'a') as f:
                    f.write(str(beta_list) + '\n')

        # Aggregate
        aggregate(
            all_clients,
            save_ckpt,
            last_ckpt,
            hyperweight,
            args.encoder_agg,
            args.decoder_agg,
            args.ca_c,
        )

        # Refresh last_ckpt
        for idx in range(N):
            last_ckpt[idx] = copy.deepcopy(all_clients[idx]['model'].module.state_dict())
        if args.save_vram:
            last_ckpt = move_ckpt(last_ckpt, 'cpu')

        print(f"CR {cr} finished. Time: {time.time() - start_time:.1f}s")

        # Validate & checkpoint
        if (cr + 1) == args.max_rounds or (cr + 1) % args.eval_freq == 0:
            print(f'Validation at CR {cr}.')
            val_logs = {}
            for idx in range(N):
                res = eval_metric(idx=idx, **all_clients[idx])
                val_logs.update(res)
            print(val_logs)

            save_ckpt_temp = {}
            for idx in range(N):
                save_ckpt_temp[idx] = copy.deepcopy(all_clients[idx]['model'].module.state_dict())
            torch.save(save_ckpt_temp, os.path.join(args.checkpoint_dir, 'checkpoint.pth'))
            print('Checkpoint saved.')
            del save_ckpt_temp

    print('Training finished.')

# ---------------------
# Entry point
# ---------------------
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str, required=True, help="Config file path")
    parser.add_argument('--exp', type=str, required=True, help="Experiment name")
    parser.add_argument('--results_dir', type=str, required=True, help='Directory of results')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--wandb_name', type=str, help="Wandb project name")
    parser.add_argument('--fp16', action='store_true', help='Whether to use fp16')
    parser.add_argument('--save_vram', action='store_true', help='Whether to save vram')

    parser.add_argument('--max_rounds', type=int, default=20)
    parser.add_argument('--eval_freq', type=int, default=4)

    parser.add_argument('--encoder_agg', default='conflict_averse', help="none,fedavg,conflict_averse")
    parser.add_argument('--ca_c', type=float, default=0.4)
    parser.add_argument('--enc_alpha_init', type=float, default=0.1)
    parser.add_argument('--decoder_agg', default='cross_attention', help="none,fedavg,cross_attention")
    parser.add_argument('--dec_beta_init', type=float, default=0.1)

    args = parser.parse_args()

    with open(args.config_path, 'r') as stream:
        exp_config = yaml.safe_load(stream)
    exp_config = {**exp_config, **vars(args)}

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cudnn.benchmark = True
    cv2.setNumThreads(0)

    # Output dirs
    os.makedirs(args.results_dir, exist_ok=True)
    args.exp_dir, args.checkpoint_dir = create_results_dir(args.results_dir, args.exp)
    shutil.copy(args.config_path, os.path.join(args.exp_dir, 'config.yml'))
    if args.wandb_name is not None:
        import wandb
        wandb.init(project=args.wandb_name, id=args.exp, name=args.exp, config=exp_config)

    # Build client configs (assuming your config YAML lists SHAPES under ST_Datasets or MT_Datasets)
    client_configs = {}
    if 'ST_Datasets' in exp_config:
        client_configs.update(get_st_config(exp_config['ST_Datasets'], local_rank=0))
    if 'MT_Datasets' in exp_config:
        client_configs.update(get_mt_config(exp_config['MT_Datasets'], local_rank=0))

    # Get clients (browser-friendly)
    all_clients, n_decoders = get_clients_browser_friendly(args, exp_config['Model'], client_configs, device)

    # Hyperweights (kept simple; same API, no real DDP)
    hyperweight = {}
    if args.encoder_agg == "conflict_averse":
        hypernet = HyperAggWeight(K=len(all_clients), init_alpha=args.enc_alpha_init).to(device)
        hyperweight['enc'] = IdentityDDP(hypernet)  # keep .module
        hyperweight['enc_optimizer'] = torch.optim.SGD(hypernet.parameters(), **exp_config['Hyperweight'])
    if args.decoder_agg == "cross_attention":
        dummy_decoder = all_clients[0]['model'].module.decoders
        hypernet = HyperCrossAttention(model=dummy_decoder, K=n_decoders, init_beta=args.dec_beta_init).to(device)
        hyperweight['dec'] = IdentityDDP(hypernet)
        hyperweight['dec_optimizer'] = torch.optim.SGD(hypernet.parameters(), **exp_config['Hyperweight'])

    # Go!
    main(
        args=args,
        all_clients=all_clients,
        hyperweight=hyperweight,
        device=device,
    )

    if args.wandb_name is not None:
        import wandb
        wandb.finish()
