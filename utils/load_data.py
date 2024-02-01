import os
import numpy as np
from PIL import Image
from tqdm import trange
import matplotlib.pyplot as plt
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import grad
from torchvision import transforms
from torchvision import datasets
import torchvision.datasets.utils as dataset_utils
from torch.utils.data import Dataset, DataLoader
import torchvision.models as models
from torch.optim.lr_scheduler import CosineAnnealingLR
from wilds import get_dataset
from wilds.common.data_loaders import get_train_loader
import torchvision.transforms as transforms
from wilds.common.data_loaders import get_eval_loader
import os
import sys
import copy
import argparse
import importlib
import copy
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import LinearSegmentedColormap, to_rgb
from tqdm import tqdm

import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn.functional as f
import pandas as pd
import numpy as np
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import torchvision
from PIL import Image
import torch 
from spuco.datasets import WILDSDatasetWrapper
from spuco.datasets import GroupLabeledDatasetWrapper
from spuco.utils import set_seed
from wilds import get_dataset
import torchvision.transforms as transforms
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from spuco.models import model_factory 
from spuco.evaluate import Evaluator
from spuco.last_layer_retrain import DFR, DISPEL

import pickle
# from spuco.invariant_train import ERM 
from torch.optim import SGD
from spuco.group_inference import JTTInference
from spuco.models import model_factory 
from spuco.utils import Trainer
from torch.optim import SGD


from spuco.datasets import SpuCoMNIST, SpuriousFeatureDifficulty
import torchvision.transforms as T


def color_grayscale_arr(arr, red=True):
    """Converts grayscale image to either red or green"""
    assert arr.ndim == 2
    dtype = arr.dtype
    h, w = arr.shape
    arr = np.reshape(arr, [h, w, 1])
    if red:
        arr = np.concatenate([arr, np.zeros((h, w, 2), dtype=dtype)], axis=2)
    else:
        arr = np.concatenate([np.zeros((h, w, 1), dtype=dtype), arr, np.zeros((h, w, 1), dtype=dtype)], axis=2)
    return arr

class ColoredMNIST(datasets.VisionDataset):
    """
    Colored MNIST dataset for testing IRM. Prepared using procedure from https://arxiv.org/pdf/1907.02893.pdf

    Args:
        root (string): Root directory of dataset where ``ColoredMNIST/*.pt`` will exist.
        env (string): Which environment to load. Must be 1 of 'train1', 'val', 'test', or 'all_train'.
        transform (callable, optional): A function/transform that takes in a PIL image and returns a transformed version.
        target_transform (callable, optional): A function/transform that takes in the target and transforms it.
    """
    def __init__(self, root='./data', env='train1', transform=None, target_transform=None):
        super(ColoredMNIST, self).__init__(root, transform=transform, target_transform=target_transform)
        self.prepare_colored_mnist()
        if env in ['train1', 'val', 'test']:
            self.data_label_tuples = torch.load(os.path.join(self.root, 'ColoredMNIST', env) + '.pt')
        elif env == 'all_train':
            self.data_label_tuples = torch.load(os.path.join(self.root, 'ColoredMNIST', 'train1.pt')) + \
                                     torch.load(os.path.join(self.root, 'ColoredMNIST', 'val.pt'))
        else:
            raise RuntimeError(f'{env} env unknown. Valid envs are train1, val, test, and all_train')

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is the index of the target class.
        """
        img, target, c = self.data_label_tuples[index]

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, c

    def __len__(self):
        return len(self.data_label_tuples)

    def prepare_colored_mnist(self):
        colored_mnist_dir = os.path.join(self.root, 'ColoredMNIST')
        # if os.path.exists(os.path.join(colored_mnist_dir, 'train1.pt')) and \
        #    os.path.exists(os.path.join(colored_mnist_dir, 'val.pt')) and \
        #    os.path.exists(os.path.join(colored_mnist_dir, 'test.pt')):
        #     print('Colored MNIST dataset already exists')
        #     return

        print('Preparing Colored MNIST')
        train_mnist = datasets.mnist.MNIST(self.root, train=True, download=True)

        train1_set = []
        val_set = []
        test_set = []
        for idx, (im, label) in enumerate(train_mnist):
            if idx % 10000 == 0:
                print(f'Converting image {idx}/{len(train_mnist)}')
            im_array = np.array(im)

            # Assign a binary label y to the image based on the digit
            binary_label = 0 if label < 5 else 1

            # Flip label with 25% probability
            if np.random.uniform() < 0.25:
                binary_label = binary_label ^ 1

            # Color the image either red or green according to its possibly flipped label
            color_red = binary_label == 0

            # Flip the color with a probability e that depends on the environment
            if idx < 30000:
                # 20% in the first training environment
                if np.random.uniform() < 0.2:
                    color_red = not color_red
            elif idx < 40000:
                # 10% in the first training environment
                if np.random.uniform() < 0.5:
                    color_red = not color_red
            else:
                # 90% in the test environment
                if np.random.uniform() < 0.9:
                    color_red = not color_red

            colored_arr = color_grayscale_arr(im_array, red=color_red)

            if idx < 30000:
                train1_set.append((Image.fromarray(colored_arr), binary_label, color_red))
            elif idx < 40000:
                val_set.append((Image.fromarray(colored_arr), binary_label, color_red))
            else:
                test_set.append((Image.fromarray(colored_arr), binary_label, color_red))
        random.shuffle(train1_set)
        random.shuffle(val_set)
        random.shuffle(test_set)
        if not os.path.exists(colored_mnist_dir):
            os.makedirs(colored_mnist_dir)
        torch.save(train1_set, os.path.join(colored_mnist_dir, 'train1.pt'))
        torch.save(val_set, os.path.join(colored_mnist_dir, 'val.pt'))
        torch.save(test_set, os.path.join(colored_mnist_dir, 'test.pt'))

def load_cmnist():
    
    train_dataset_img = ColoredMNIST(root='./data', env='train1',
                 transform=transforms.Compose([
                     transforms.ToTensor(),
                     transforms.Normalize((0.1307, 0.1307, 0.), (0.3081, 0.3081, 0.3081))
                 ]))
    
    # train_loader_img = DataLoader(
    # train_dataset_img,
    # batch_size=batch_size, shuffle=False)
    
    
    eval_dataset_img = ColoredMNIST(root='./data', env='val',
                 transform=transforms.Compose([
                     transforms.ToTensor(),
                     transforms.Normalize((0.1307, 0.1307, 0.), (0.3081, 0.3081, 0.3081))
                 ]))
    
    # eval_loader_img = DataLoader(
    # eval_dataset_img,
    # batch_size=batch_size, shuffle=False)
    
    
    test_dataset_img = ColoredMNIST(root='./data', env='test',
                 transform=transforms.Compose([
                     transforms.ToTensor(),
                     transforms.Normalize((0.1307, 0.1307, 0.), (0.3081, 0.3081, 0.3081))
                 ]))
    
    # test_loader_img = DataLoader(
    # test_dataset_img,
    # batch_size=batch_size, shuffle=False)
    return train_dataset_img, eval_dataset_img,test_dataset_img




class ColoredMNISTsize(datasets.VisionDataset):
    """
    Colored MNIST dataset for testing IRM. Prepared using procedure from https://arxiv.org/pdf/1907.02893.pdf

    Args:
        root (string): Root directory of dataset where ``ColoredMNIST/*.pt`` will exist.
        env (string): Which environment to load. Must be 1 of 'train1', 'val', 'test', or 'all_train'.
        transform (callable, optional): A function/transform that takes in a PIL image and returns a transformed version.
        target_transform (callable, optional): A function/transform that takes in the target and transforms it.
    """
    def __init__(self, root='./data', env='train1', transform=None, target_transform=None):
        super(ColoredMNISTsize, self).__init__(root, transform=transform, target_transform=target_transform)
        self.prepare_colored_mnist()
        if env in ['train1', 'val', 'test']:
            self.data_label_tuples = torch.load(os.path.join(self.root, 'ColoredMNISTsize', env) + '.pt')
        elif env == 'all_train':
            self.data_label_tuples = torch.load(os.path.join(self.root, 'ColoredMNISTsize', 'train1.pt')) + \
                                     torch.load(os.path.join(self.root, 'ColoredMNISTsize', 'val.pt'))
        else:
            raise RuntimeError(f'{env} env unknown. Valid envs are train1, val, test, and all_train')

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is the index of the target class.
        """
        img, target, c = self.data_label_tuples[index]

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, c

    def __len__(self):
        return len(self.data_label_tuples)

    def prepare_colored_mnist(self):
        colored_mnist_dir = os.path.join(self.root, 'ColoredMNISTsize')
        # if os.path.exists(os.path.join(colored_mnist_dir, 'train1.pt')) and \
        #    os.path.exists(os.path.join(colored_mnist_dir, 'val.pt')) and \
        #    os.path.exists(os.path.join(colored_mnist_dir, 'test.pt')):
        #     print('Colored MNIST dataset already exists')
        #     return

        print('Preparing Colored MNISTsize')
        train_mnist = datasets.mnist.MNIST(self.root, train=True, download=True)

        train1_set = []
        val_set = []
        test_set = []
        for idx, (im, label) in enumerate(train_mnist):
            if idx % 10000 == 0:
                print(f'Converting image {idx}/{len(train_mnist)}')
            im_array = np.array(im)

            # Assign a binary label y to the image based on the digit
            binary_label = 0 if label < 5 else 1

            # Flip label with 25% probability
            if np.random.uniform() < 0.25:
                binary_label = binary_label ^ 1

            # Color the image either red or green according to its possibly flipped label
            color_red = binary_label == 0

            # Flip the color with a probability e that depends on the environment
            if idx < 10000:
                # 20% in the first training environment
                if np.random.uniform() < 0.2:
                    color_red = not color_red
            elif idx < 40000:
                # 10% in the first training environment
                if np.random.uniform() < 0.5:
                    color_red = not color_red
            else:
                # 90% in the test environment
                if np.random.uniform() < 0.9:
                    color_red = not color_red

            colored_arr = color_grayscale_arr(im_array, red=color_red)

            if idx < 10000:
                train1_set.append((Image.fromarray(colored_arr), binary_label, color_red))
            elif idx < 40000:
                val_set.append((Image.fromarray(colored_arr), binary_label, color_red))
            else:
                test_set.append((Image.fromarray(colored_arr), binary_label, color_red))
        random.shuffle(train1_set)
        random.shuffle(val_set)
        random.shuffle(test_set)
        if not os.path.exists(colored_mnist_dir):
            os.makedirs(colored_mnist_dir)
        torch.save(train1_set, os.path.join(colored_mnist_dir, 'train1.pt'))
        torch.save(val_set, os.path.join(colored_mnist_dir, 'val.pt'))
        torch.save(test_set, os.path.join(colored_mnist_dir, 'test.pt'))

def load_cmnist_size():
    
    train_dataset_img = ColoredMNISTsize(root='./data', env='train1',
                 transform=transforms.Compose([
                     transforms.ToTensor(),
                     transforms.Normalize((0.1307, 0.1307, 0.), (0.3081, 0.3081, 0.3081))
                 ]))
    
    # train_loader_img = DataLoader(
    # train_dataset_img,
    # batch_size=batch_size, shuffle=False)
    
    
    eval_dataset_img = ColoredMNISTsize(root='./data', env='val',
                 transform=transforms.Compose([
                     transforms.ToTensor(),
                     transforms.Normalize((0.1307, 0.1307, 0.), (0.3081, 0.3081, 0.3081))
                 ]))
    
    # eval_loader_img = DataLoader(
    # eval_dataset_img,
    # batch_size=batch_size, shuffle=False)
    
    
    test_dataset_img = ColoredMNISTsize(root='./data', env='test',
                 transform=transforms.Compose([
                     transforms.ToTensor(),
                     transforms.Normalize((0.1307, 0.1307, 0.), (0.3081, 0.3081, 0.3081))
                 ]))
    
    # test_loader_img = DataLoader(
    # test_dataset_img,
    # batch_size=batch_size, shuffle=False)
    return train_dataset_img, eval_dataset_img,test_dataset_img




# class MNISTColor(datasets.VisionDataset):
#     """
#     Colored MNIST dataset for testing IRM. Prepared using procedure from https://arxiv.org/pdf/1907.02893.pdf

#     Args:
#         root (string): Root directory of dataset where ``ColoredMNIST/*.pt`` will exist.
#         env (string): Which environment to load. Must be 1 of 'train1', 'val', 'test', or 'all_train'.
#         transform (callable, optional): A function/transform that takes in a PIL image and returns a transformed version.
#         target_transform (callable, optional): A function/transform that takes in the target and transforms it.
#     """
#     def __init__(self, root='./data', env='train1', transform=None, target_transform=None):
#         super(MNISTColor, self).__init__(root, transform=transform, target_transform=target_transform)
#         self.prepare_colored_mnist()
#         if env in ['train1', 'val', 'test']:
#             self.data_label_tuples = torch.load(os.path.join(self.root, 'ColoredMNIST', env) + '.pt')
#         elif env == 'all_train':
#             self.data_label_tuples = torch.load(os.path.join(self.root, 'ColoredMNIST', 'train1.pt')) + \
#                                      torch.load(os.path.join(self.root, 'ColoredMNIST', 'val.pt'))
#         else:
#             raise RuntimeError(f'{env} env unknown. Valid envs are train1, val, test, and all_train')

#     def __getitem__(self, index):
#         """
#         Args:
#             index (int): Index

#         Returns:
#             tuple: (image, target) where target is the index of the target class.
#         """
#         img, target, c = self.data_label_tuples[index]

#         if self.transform is not None:
#             img = self.transform(img)

#         if self.target_transform is not None:
#             target = self.target_transform(target)

#         return img, target, c

#     def __len__(self):
#         return len(self.data_label_tuples)

#     def prepare_colored_mnist(self):
#         colored_mnist_dir = os.path.join(self.root, 'ColoredMNIST')
#         # if os.path.exists(os.path.join(colored_mnist_dir, 'train1.pt')) and \
#         #    os.path.exists(os.path.join(colored_mnist_dir, 'val.pt')) and \
#         #    os.path.exists(os.path.join(colored_mnist_dir, 'test.pt')):
#         #     print('Colored MNIST dataset already exists')
#         #     return

#         print('Preparing MNIST Color')
#         train_mnist = datasets.mnist.MNIST(self.root, train=True, download=True)

#         train1_set = []
#         val_set = []
#         test_set = []
#         for idx, (im, label) in enumerate(train_mnist):
#             if idx % 10000 == 0:
#                 print(f'Converting image {idx}/{len(train_mnist)}')
#             im_array = np.array(im)

#             # Assign a binary label y to the image based on the digit
#             binary_label = 0 if label < 5 else 1

#             # Flip label with 25% probability
#             if np.random.uniform() < 0.25:
#                 binary_label = binary_label ^ 1

#             # Color the image either red or green according to its possibly flipped label
#             color_red = binary_label == 0

#             # Flip the color with a probability e that depends on the environment
#             if idx < 30000:
#                 # 20% in the first training environment
#                 if np.random.uniform() < 0.2:
#                     color_red = not color_red
#             elif idx < 40000:
#                 # 10% in the first training environment
#                 if np.random.uniform() < 0.5:
#                     color_red = not color_red
#             else:
#                 # 90% in the test environment
#                 if np.random.uniform() < 0.9:
#                     color_red = not color_red

#             colored_arr = color_grayscale_arr(im_array, red=color_red)

#             if idx < 30000:
#                 train1_set.append((Image.fromarray(colored_arr), color_red, binary_label))
#             elif idx < 40000:
#                 val_set.append((Image.fromarray(colored_arr), color_red, binary_label))
#             else:
#                 test_set.append((Image.fromarray(colored_arr), color_red, binary_label))
#         random.shuffle(train1_set)
#         random.shuffle(val_set)
#         random.shuffle(test_set)
#         if not os.path.exists(colored_mnist_dir):
#             os.makedirs(colored_mnist_dir)
#         torch.save(train1_set, os.path.join(colored_mnist_dir, 'train1.pt'))
#         torch.save(val_set, os.path.join(colored_mnist_dir, 'val.pt'))
#         torch.save(test_set, os.path.join(colored_mnist_dir, 'test.pt'))

# def load_mcolor():
    
#     train_dataset_img = MNISTColor(root='./data', env='train1',
#                  transform=transforms.Compose([
#                      transforms.ToTensor(),
#                      transforms.Normalize((0.1307, 0.1307, 0.), (0.3081, 0.3081, 0.3081))
#                  ]))
    
#     # train_loader_img = DataLoader(
#     # train_dataset_img,
#     # batch_size=batch_size, shuffle=False)
    
    
#     eval_dataset_img = MNISTColor(root='./data', env='val',
#                  transform=transforms.Compose([
#                      transforms.ToTensor(),
#                      transforms.Normalize((0.1307, 0.1307, 0.), (0.3081, 0.3081, 0.3081))
#                  ]))
    
#     # eval_loader_img = DataLoader(
#     # eval_dataset_img,
#     # batch_size=batch_size, shuffle=False)
    
    
#     test_dataset_img = MNISTColor(root='./data', env='test',
#                  transform=transforms.Compose([
#                      transforms.ToTensor(),
#                      transforms.Normalize((0.1307, 0.1307, 0.), (0.3081, 0.3081, 0.3081))
#                  ]))
    
#     # test_loader_img = DataLoader(
#     # test_dataset_img,
#     # batch_size=batch_size, shuffle=False)
#     return train_dataset_img, eval_dataset_img,test_dataset_img






def load_multipy_cmnist():
    classes = [[0, 1, 2, 3,4], [5, 6, 7, 8, 9]]#[[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]]
    difficulty = SpuriousFeatureDifficulty.MAGNITUDE_LARGE

    trainset = SpuCoMNIST(
    root="/home/yujin/dm/disk/data/mnist",
    spurious_feature_difficulty=difficulty,
    spurious_correlation_strength=0.995,
    classes=classes,
    split="train")
    trainset.initialize()

    valset = SpuCoMNIST(
    root="//home/yujin/dm/disk/data/mnist",
    spurious_feature_difficulty=difficulty,
    classes=classes,
    split="val")
    valset.initialize()
    
    testset = SpuCoMNIST(
    root="//home/yujin/dm/disk/data/mnist",
    spurious_feature_difficulty=difficulty,
    classes=classes,
    split="test")
    testset.initialize()

    return trainset, valset, testset


# class ColoredMNIST(Dataset):
#     """
#     Colored MNIST dataset - labels spuriously correlated with color
#     - We store the label, the spurious attribute, and subclass labels if applicable
#     Args:
#     - data (torch.Tensor): MNIST images
#     - targets (torch.Tensor): MNIST original labels
#     - train_classes (list[]): List of lists describing how to organize labels
#                                 - Each inner list denotes a group, i.e. 
#                                 they all have the same classification label
#                                 - Any labels left out are excluded from training set
#     - train (bool): Training or test dataset
#     - p_correlation (float): Strength of spurious correlation, in [0, 1]
#     - test_shift (str): How to organize test set, from 'random', 'same', 'new'
#     - cmap (str): Colormap for coloring MNIST digits
#     - flipped (bool): If true, color background and keep digit black
#     - transform (torchvision.transforms): Image transformations
#     - args (argparse): Experiment arguments
#     Returns:
#     - __getitem__() returns tuple of image, label, and the index, which can be used for
#                     looking up additional info (e.g. subclass label, spurious attribute)
#     """

#     def __init__(self, data, targets, train_classes=[[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]],
#                  train=True, p_correlation=0.995, test_shift='random', cmap='hsv',
#                  flipped=False, transform=None, args=None):
#         self.args = args
#         # Initialize classes
#         self.class_map = self._init_class_map(train_classes)
#         self.classes = list(self.class_map.keys())
#         self.new_classes = np.unique(list(self.class_map.values()))

#         self.test_classes = [x for x in np.unique(
#             targets) if x not in self.classes]
#         self.p_correlation = p_correlation
#         # Setup spurious correlation ratios per class
#         if args.p_corr_by_class is not None:
#             self.p_correlation = args.p_corr_by_class
#         else:
#             self.p_correlation = [p_correlation] * len(self.new_classes)
#         self.train = train
#         self.test_shift = test_shift
#         self.transform = transform

#         # Filter for train_classes
#         class_filter = torch.stack([(targets == i)
#                                     for i in self.classes]).sum(dim=0)
#         self.targets = targets[class_filter > 0]
#         data = data[class_filter > 0]

#         self.targets_all = {'spurious': np.zeros(len(self.targets), dtype=int),
#                             'sub_target': copy.deepcopy(self.targets)}
#         # Update targets
#         self.targets = torch.tensor([self.class_map[t.item()] for t in self.targets],
#                                     dtype=self.targets.dtype)
#         self.targets_all['target'] = self.targets.numpy()
        
#         # Colors + Data
#         self.colors = self._init_colors(cmap)
#         if flipped:
#             data = 255 - data
#         if data.shape[1] != 3:   # Add RGB channels
#             data = data.unsqueeze(1).repeat(1, 3, 1, 1)
#         self.data = self._init_data(data)
#         self.spurious_group_names = self.colors
#         # Adjust in case data was resampled for class imbalance
#         if self.args.train_class_ratios is not None and self.train is True:
#             self.targets = self.targets[self.selected_indices]
#             for k in self.targets_all:
#                 self.targets_all[k] = self.targets_all[k][self.selected_indices]
                
#         self.n_classes = len(train_classes)
#         self.n_groups = pow(self.n_classes, 2)
#         target_spurious_to_group_ix = np.arange(self.n_groups).reshape((self.n_classes, self.n_classes)).astype('int')
        
#         # Access datapoint's subgroup idx, i.e. 1 of 25 diff values if we have 5 classes, 5 colors
#         group_array = []
#         for ix in range(len(self.targets_all['target'])):
#             y = self.targets_all['target'][ix]
#             a = self.targets_all['spurious'][ix]
#             group_array.append(target_spurious_to_group_ix[y][a])
#         group_array = np.array(group_array)
#         self.group_array = torch.LongTensor(group_array)
        
#         # Index for (y, a) group
#         all_group_labels = []
#         for n in range(self.n_classes):
#             for m in range(self.n_classes):
#                 all_group_labels.append(str((n, m)))
#         self.targets_all['group_idx'] = self.group_array.numpy()
#         self.group_labels = all_group_labels

#     def __len__(self):
#         return len(self.targets)

#     def __getitem__(self, idx):
#         sample = self.data[idx]
#         if self.transform:
#             sample = self.transform(sample)
#         return (sample, self.targets[idx], idx)

#     def _init_class_map(self, classes):
#         class_map = {}
#         for c_ix, targets in enumerate(classes):
#             for t in targets:
#                 class_map[t] = c_ix
#         return class_map

#     def _init_colors(self, cmap):
#         # Initialize list of RGB color values
#         try:
#             cmap = cm.get_cmap(cmap)
#         except ValueError:  # single color
#             cmap = self._get_single_color_cmap(cmap)
#         cmap_vals = np.arange(0, 1, step=1 / len(self.new_classes))
#         colors = []
#         for ix, c in enumerate(self.new_classes):
#             rgb = cmap(cmap_vals[ix])[:3]
#             rgb = [int(np.float(x)) for x in np.array(rgb) * 255]
#             colors.append(rgb)
#         return colors

#     def _get_single_color_cmap(self, c):
#         rgb = to_rgb(c)
#         r1, g1, b1 = rgb
#         cdict = {'red':   ((0, r1, r1),
#                            (1, r1, r1)),
#                  'green': ((0, g1, g1),
#                            (1, g1, g1)),
#                  'blue':  ((0, b1, b1),
#                            (1, b1, b1))}
#         cmap = LinearSegmentedColormap('custom_cmap', cdict)
#         return cmap

#     def _init_data(self, data):
#         np.random.seed(self.args.seed)
#         self.selected_indices = []
#         pbar = tqdm(total=len(self.targets), desc='Initializing data')
#         for ix, c in enumerate(self.new_classes):
#             class_ix = np.where(self.targets == c)[0]
#             # Introduce class imbalance
#             if self.args.train_class_ratios is not None and self.train is True:
#                 class_size = int(np.round(
#                     len(class_ix) * self.args.train_class_ratios[ix][0]))
#                 class_ix = np.random.choice(
#                     class_ix, size=class_size, replace=False)
#                 self.selected_indices.append(class_ix)
#             is_spurious = np.random.binomial(1, self.p_correlation[ix],
#                                              size=len(class_ix))
#             for cix_, cix in enumerate(class_ix):
#                 # Replace pixels
#                 pixels_r = np.where(
#                     np.logical_and(data[cix, 0, :, :] >= 120,
#                                    data[cix, 0, :, :] <= 255))
#                 # May refactor this out as a separate function later
#                 if self.train or self.test_shift == 'iid':
#                     color_ix = (ix if is_spurious[cix_] else
#                                 np.random.choice([
#                                     x for x in np.arange(len(self.colors)) if x != ix]))
#                 elif 'shift' in self.test_shift:
#                     n = int(self.test_shift.split('_')[-1])
#                     color_ix = (ix + n) % len(self.new_classes)
#                 else:
#                     color_ix = np.random.randint(len(self.colors))
#                 color = self.colors[color_ix]
#                 data[cix, :, pixels_r[0], pixels_r[1]] = (
#                     torch.tensor(color, dtype=torch.uint8).unsqueeze(1).repeat(1, len(pixels_r[0])))
#                 self.targets_all['spurious'][cix] = int(color_ix)
#                 pbar.update(1)
#         if self.args.train_class_ratios is not None and self.train is True:
#             self.selected_indices = np.concatenate(self.selected_indices)
#             return data[self.selected_indices].float() / 255
#         return data.float() / 255  # For normalization

#     def get_dataloader(self, batch_size, shuffle, num_workers):
#         return DataLoader(self, batch_size=batch_size, shuffle=shuffle,
#                           num_workers=num_workers)


# def load_cmnist(train_shuffle=False, transform=None):
#     """
#     Default dataloader setup for Colored MNIST
#     Args:
#     - args (argparse): Experiment arguments
#     - transform (torchvision.transforms): Image transformations
#     Returns:
#     - (train_loader, test_loader): Tuple of dataloaders for train and test
#     """
#     mnist_train = torchvision.datasets.MNIST(root=args.data_path, 
#                                              train=True, download=True)
#     mnist_test = torchvision.datasets.MNIST(root=args.data_path, 
#                                             train=False, download=True)

#     transform = (transforms.Compose([transforms.Resize(40),
#                                      transforms.RandomCrop(32, padding=0),
#                                      transforms.Normalize((0.5, 0.5, 0.5),
#                                                           (0.5, 0.5, 0.5))])
#                  if transform is None else transform)
    
#     # Split original train set into train and val
#     train_indices, val_indices = train_val_split(mnist_train, 
#                                                  args.val_split,
#                                                  args.seed)
#     train_data = mnist_train.data[train_indices]
#     train_targets = mnist_train.targets[train_indices]
#     val_data = mnist_train.data[val_indices]
#     val_targets = mnist_train.targets[val_indices]
    
#     colored_mnist_train = ColoredMNIST(data=train_data,
#                                        targets=train_targets,
#                                        train_classes=args.train_classes,
#                                        train=True,
#                                        p_correlation=args.p_correlation,
#                                        test_shift=args.test_shift,
#                                        cmap=args.data_cmap,
#                                        transform=transform,
#                                        flipped=args.flipped,
#                                        args=args)
#     # Val set is setup with same data distribution as test set by convention.
#     colored_mnist_val = None
#     if len(val_data) > 0:
#         colored_mnist_val = ColoredMNIST(data=val_data, targets=val_targets,
#                                          train_classes=args.train_classes,
#                                          train=False,
#                                          p_correlation=args.p_correlation,
#                                          test_shift=args.test_shift,
#                                          cmap=args.data_cmap,
#                                          transform=transform,
#                                          flipped=args.flipped,
#                                          args=args)
        
#     test_cmap = args.data_cmap if args.test_cmap == '' else args.test_cmap
#     test_p_corr = args.p_correlation if args.test_cmap == '' else 1.0
#     colored_mnist_test = ColoredMNIST(data=mnist_test.data,
#                                       targets=mnist_test.targets,
#                                       train_classes=args.train_classes,
#                                       train=False,
#                                       p_correlation=test_p_corr,
#                                       test_shift=args.test_shift,
#                                       cmap=test_cmap,
#                                       transform=transform,
#                                       flipped=args.flipped,
#                                       args=args)
#     train_loader = DataLoader(colored_mnist_train, batch_size=args.bs_trn,
#                               shuffle=train_shuffle,
#                               num_workers=args.num_workers)
#     val_loader   = DataLoader(colored_mnist_val, batch_size=args.bs_val,
#                               shuffle=False, num_workers=args.num_workers)
#     test_loader  = DataLoader(colored_mnist_test, batch_size=args.bs_val,
#                               shuffle=False, num_workers=args.num_workers)
#     # Update args.num_classes
#     args.num_classes = len(colored_mnist_train.new_classes)
#     return train_loader, val_loader, test_loader


    
    


def load_waterbirds():
    dataset = get_dataset(dataset="waterbirds", 
                      root_dir="./data/waterbirds/",
                      download=True)
    scale = 256.0/224.0
    target_resolution=(224, 224)
    train_dataset_img = dataset.get_subset(
    "train",
    transform= transforms.Compose(
         [
         transforms.Resize((int(target_resolution[0]*scale), int(target_resolution[1]*scale))),
          transforms.ToTensor(),
        transforms.RandomResizedCrop(224, scale=(0.7, 1.0), ratio=(0.75, 4./3.),interpolation=2),
         transforms.RandomHorizontalFlip(),
          transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]
     ))
    eval_dataset_img = dataset.get_subset(
    "val",
    transform=transforms.Compose(
        [
         transforms.Resize((int(target_resolution[0]*scale), int(target_resolution[1]*scale))),
         transforms.ToTensor(),
         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]
    ))
    test_dataset_img = dataset.get_subset(
    "test",
    transform=transforms.Compose(
        [transforms.Resize((int(target_resolution[0]*scale), int(target_resolution[1]*scale))),
         transforms.ToTensor(),
         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]
    ))
    return train_dataset_img, eval_dataset_img,test_dataset_img

def load_celebA():
    dataset = get_dataset(dataset="celebA", 
                      root_dir="./data/celebA/",
                      download=True)
    train_dataset_img = dataset.get_subset(
    "train",
    transform=transforms.Compose(
        [transforms.Resize((224, 224)),
         transforms.ToTensor(),
         transforms.RandomResizedCrop(224, scale=(0.7, 1.0), ratio=(0.75, 4./3.),interpolation=2),
         transforms.RandomHorizontalFlip(),
         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]
    ))
    eval_dataset_img = dataset.get_subset(
    "val",
    transform=transforms.Compose(
        [transforms.Resize((224, 224)),
         transforms.ToTensor(),
         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]
    ))
    test_dataset_img = dataset.get_subset(
    "test",
    transform=transforms.Compose(
        [transforms.Resize((224, 224)),
         transforms.ToTensor(),
         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]
    ))
    return train_dataset_img, eval_dataset_img,test_dataset_img


import torchvision.transforms as transforms
import einops
import torch
import math
from transformers import BertTokenizer
from transformers import DebertaV2Tokenizer
from transformers import AlbertTokenizer
from timm.data.random_erasing import RandomErasing

class TokenizeTransform:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, text):
        tokens = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=220,
            return_tensors="pt",
        )

        return torch.squeeze(torch.stack((
            tokens["input_ids"], tokens["attention_mask"], 
            tokens["token_type_ids"]), dim=2), dim=0)

class BertTokenizeTransform(TokenizeTransform):
    def __init__(self, train):
        super().__init__(
                tokenizer=BertTokenizer.from_pretrained("bert-base-uncased"))
        del train


from typing import Dict, List, Tuple, Optional

from tqdm import tqdm
from wilds.datasets.wilds_dataset import WILDSDataset
from torch.utils.data import Subset,Dataset, DataLoader
from spuco.datasets import BaseSpuCoCompatibleDataset

class WILDSDatasetWrapper(BaseSpuCoCompatibleDataset):
    """
    Wrapper class that wraps WILDSDataset into a Dataset to be compatible with SpuCo.
    """
    def __init__(
        self,
        dataset: WILDSDataset,
        metadata_spurious_label: str,
        verbose=False,
        subset_indices: Optional[List[int]] = None
    ):
        """
        Wraps  WILDS Dataset into a Dataset object. 

        :param dataset: The source WILDS dataset
        :type dataset: WILDDataset
        :param metadata_spurious_label: String name of property in metadata_map corresponding to spurious target
        :type metadata_spurious_label: str 
        :param verbose: Show logs
        :type verbose: bool
        """
        super().__init__()

        self.dataset = dataset
        self._num_classes = dataset.n_classes 

        # Get index in meta data array corresponding to spurious target 
        spurious_target_idx = dataset.metadata_fields.index(metadata_spurious_label)

        # Get labels 
        self._labels = dataset.y_array.long().tolist()

        # Get spurious labels
        self._spurious = dataset.metadata_array[:, spurious_target_idx].long().tolist()

        # Create group partition using labels and spurious labels
        self._group_partition = {}
        for i, group_label in tqdm(
            enumerate(zip(self._labels, self._spurious)),
            desc="Partitioning data indices into groups",
            disable=not verbose,
            total=len(self.dataset)
        ):
            if group_label not in self._group_partition:
                self._group_partition[group_label] = []
            self._group_partition[group_label].append(i)
        
        # Set group weights based on group sizes
        self._group_weights = {}
        for group_label in self._group_partition.keys():
            self._group_weights[group_label] = len(self._group_partition[group_label]) / len(self.dataset)
        
        # Subset if needed
        self.indices = range(len(dataset))
        if subset_indices is not None:
            self.indices = subset_indices

    @property
    def group_partition(self) -> Dict[Tuple[int, int], List[int]]:
        """
        Dictionary partitioning indices into groups
        """
        return self._group_partition 
    
    @property
    def group_weights(self) -> Dict[Tuple[int, int], float]:
        """
        Dictionary containing the fractional weights of each group
        """
        return self._group_weights
    
    @property
    def spurious(self) -> List[int]:
        """
        List containing spurious labels for each example
        """
        return self._spurious

    @property
    def labels(self) -> List[int]:
        """
        List containing class labels for each example
        """
        return self._labels
    
    @property
    def num_classes(self) -> int:
        """
        Number of classes
        """
        return self._num_classes
    
    def __getitem__(self, index):
        """
        Retrieves an item from the dataset.

        :param index: The index of the item.
        :type index: int
        :return: The item at the given index.
        """
        index = self.indices[index]
        source_tuple = self.dataset.__getitem__(index)
        # return (source_tuple[0], source_tuple[1])
        data, label = source_tuple[0], source_tuple[1]
        spurious_label = self._spurious[index]
        return data, label, torch.tensor(spurious_label)
    
    def __len__(self):
        """
        Returns the length of the dataset.

        :return: The length of the dataset.
        :rtype: int
        """
        return len(self.indices)




from wilds import get_dataset
from spuco.utils import initialize_bert_transform


def load_civilcomments(spurious_labels = "identity_any"):
    dataset = get_dataset(dataset="civilcomments", download=True, root_dir="./data/civilcomments/")
    transform = initialize_bert_transform(model="bert-base-uncased", max_token_length=300)
    # BertTokenizeTransform(train="bert-base-uncased")#initialize_bert_transform(model="distilbert-base-uncased", max_token_length=300)
    
    train_data = dataset.get_subset(
    "train",
    transform=transform)
    
    eval_data = dataset.get_subset(
    "val",
    transform=transform)
    
    test_data = dataset.get_subset(
    "test",
    transform=transform)
    
    train_dataset = WILDSDatasetWrapper(dataset=train_data, metadata_spurious_label=spurious_labels , verbose=True)
    # train_loader_img = DataLoader(train_dataset, batch_size=16, shuffle=False)
    
    eval_dataset = WILDSDatasetWrapper(dataset=eval_data, metadata_spurious_label=spurious_labels , verbose=True)
    # eval_loader_img = DataLoader(eval_dataset, batch_size=16, shuffle=False)
    
    test_dataset = WILDSDatasetWrapper(dataset=test_data, metadata_spurious_label=spurious_labels, verbose=True)
    # test_loader_img = DataLoader(test_dataset, batch_size=16, shuffle=False)
    return train_dataset, eval_dataset,test_dataset


import copy
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import LinearSegmentedColormap, to_rgb
from tqdm import tqdm

import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader

def train_val_split(dataset, val_split, seed):
    """
    Compute indices for train and val splits
    
    Args:
    - dataset (torch.utils.data.Dataset): Pytorch dataset
    - val_split (float): Fraction of dataset allocated to validation split
    - seed (int): Reproducibility seed
    Returns:
    - train_indices, val_indices (np.array, np.array): Dataset indices
    """
    train_ix = int(np.round(val_split * len(dataset)))
    all_indices = np.arange(len(dataset))
    np.random.seed(seed)
    np.random.shuffle(all_indices)
    train_indices = all_indices[train_ix:]
    val_indices = all_indices[:train_ix]
    return train_indices, val_indices


class ColoredMNIST_(Dataset):
    """
    Colored MNIST dataset - labels spuriously correlated with color
    - We store the label, the spurious attribute, and subclass labels if applicable
    Args:
    - data (torch.Tensor): MNIST images
    - targets (torch.Tensor): MNIST original labels
    - train_classes (list[]): List of lists describing how to organize labels
                                - Each inner list denotes a group, i.e. 
                                they all have the same classification label
                                - Any labels left out are excluded from training set
    - train (bool): Training or test dataset
    - p_correlation (float): Strength of spurious correlation, in [0, 1]
    - test_shift (str): How to organize test set, from 'random', 'same', 'new'
    - cmap (str): Colormap for coloring MNIST digits
    - flipped (bool): If true, color background and keep digit black
    - transform (torchvision.transforms): Image transformations
    - args (argparse): Experiment arguments
    Returns:
    - __getitem__() returns tuple of image, label, and the index, which can be used for
                    looking up additional info (e.g. subclass label, spurious attribute)
    """

    def __init__(self, data, targets, train_classes=[[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]],
                 train=True, p_correlation=0.995, test_shift='random', cmap='hsv',
                 flipped=False, transform=None, args=None):
        self.args = args
        # Initialize classes
        self.class_map = self._init_class_map(train_classes)
        self.classes = list(self.class_map.keys())
        self.new_classes = np.unique(list(self.class_map.values()))

        self.test_classes = [x for x in np.unique(
            targets) if x not in self.classes]
        self.p_correlation = p_correlation
        # Setup spurious correlation ratios per class
        self.p_correlation = [p_correlation] * len(self.new_classes)
        self.train = train
        self.test_shift = test_shift
        self.transform = transform

        # Filter for train_classes
        class_filter = torch.stack([(targets == i)
                                    for i in self.classes]).sum(dim=0)
        self.targets = targets[class_filter > 0]
        data = data[class_filter > 0]

        self.targets_all = {'spurious': np.zeros(len(self.targets), dtype=int),
                            'sub_target': copy.deepcopy(self.targets)}
        # Update targets
        self.targets = torch.tensor([self.class_map[t.item()] for t in self.targets],
                                    dtype=self.targets.dtype)
        self.targets_all['target'] = self.targets.numpy()
        
        # Colors + Data
        self.colors = self._init_colors(cmap)
        if flipped:
            data = 255 - data
        if data.shape[1] != 3:   # Add RGB channels
            data = data.unsqueeze(1).repeat(1, 3, 1, 1)
        self.data = self._init_data(data)
        self.spurious_group_names = self.colors
        # Adjust in case data was resampled for class imbalance
        if self.train_class_ratios is not None and self.train is True:
            self.targets = self.targets[self.selected_indices]
            for k in self.targets_all:
                self.targets_all[k] = self.targets_all[k][self.selected_indices]
                
        self.n_classes = len(train_classes)
        self.n_groups = pow(self.n_classes, 2)
        target_spurious_to_group_ix = np.arange(self.n_groups).reshape((self.n_classes, self.n_classes)).astype('int')
        
        # Access datapoint's subgroup idx, i.e. 1 of 25 diff values if we have 5 classes, 5 colors
        group_array = []
        for ix in range(len(self.targets_all['target'])):
            y = self.targets_all['target'][ix]
            a = self.targets_all['spurious'][ix]
            group_array.append(target_spurious_to_group_ix[y][a])
        group_array = np.array(group_array)
        self.group_array = torch.LongTensor(group_array)
        
        # Index for (y, a) group
        all_group_labels = []
        for n in range(self.n_classes):
            for m in range(self.n_classes):
                all_group_labels.append(str((n, m)))
        self.targets_all['group_idx'] = self.group_array.numpy()
        self.group_labels = all_group_labels

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        sample = self.data[idx]
        # self.targets = torch.tensor([self.class_map[t.item()] for t in self.targets],
        #                             dtype=self.targets.dtype)
        self.spu = torch.tensor([t.item() for t in self.targets_all['spurious']],dtype=self.targets.dtype)
        if self.transform:
            sample = self.transform(sample)
        return (sample, self.targets[idx], self.spu[idx])

    def _init_class_map(self, classes):
        class_map = {}
        for c_ix, targets in enumerate(classes):
            for t in targets:
                class_map[t] = c_ix
        return class_map

    def _init_colors(self, cmap):
        # Initialize list of RGB color values
        try:
            cmap = cm.get_cmap(cmap)
        except ValueError:  # single color
            cmap = self._get_single_color_cmap(cmap)
        cmap_vals = np.arange(0, 1, step=1 / len(self.new_classes))
        colors = []
        for ix, c in enumerate(self.new_classes):
            rgb = cmap(cmap_vals[ix])[:3]
            rgb = [int(float(x)) for x in np.array(rgb) * 255]
            colors.append(rgb)
        return colors

    def _get_single_color_cmap(self, c):
        rgb = to_rgb(c)
        r1, g1, b1 = rgb
        cdict = {'red':   ((0, r1, r1),
                           (1, r1, r1)),
                 'green': ((0, g1, g1),
                           (1, g1, g1)),
                 'blue':  ((0, b1, b1),
                           (1, b1, b1))}
        cmap = LinearSegmentedColormap('custom_cmap', cdict)
        return cmap

    def _init_data(self, data):
        np.random.seed(self.args.seed)
        self.selected_indices = []
        self.train_class_ratios = 1.0
        pbar = tqdm(total=len(self.targets), desc='Initializing data')
        for ix, c in enumerate(self.new_classes):
            class_ix = np.where(self.targets == c)[0]
            # Introduce class imbalance
            if self.train_class_ratios is not None and self.train is True:
                class_size = int(np.round(
                    len(class_ix) * self.train_class_ratios))
                class_ix = np.random.choice(
                    class_ix, size=class_size, replace=False)
                self.selected_indices.append(class_ix)
            is_spurious = np.random.binomial(1, self.p_correlation[ix],
                                             size=len(class_ix))
            for cix_, cix in enumerate(class_ix):
                # Replace pixels
                pixels_r = np.where(
                    np.logical_and(data[cix, 0, :, :] >= 120,
                                   data[cix, 0, :, :] <= 255))
                # May refactor this out as a separate function later
                if self.train or self.test_shift == 'iid':
                    color_ix = (ix if is_spurious[cix_] else
                                np.random.choice([
                                    x for x in np.arange(len(self.colors)) if x != ix]))
                elif 'shift' in self.test_shift:
                    n = int(self.test_shift.split('_')[-1])
                    color_ix = (ix + n) % len(self.new_classes)
                else:
                    color_ix = np.random.randint(len(self.colors))
                color = self.colors[color_ix]
                data[cix, :, pixels_r[0], pixels_r[1]] = (
                    torch.tensor(color, dtype=torch.uint8).unsqueeze(1).repeat(1, len(pixels_r[0])))
                self.targets_all['spurious'][cix] = int(color_ix)
                pbar.update(1)
        if self.train_class_ratios is not None and self.train is True:
            self.selected_indices = np.concatenate(self.selected_indices)
            return data[self.selected_indices].float() / 255
        return data.float() / 255  # For normalization

    def get_dataloader(self, batch_size, shuffle, num_workers):
        return DataLoader(self, batch_size=batch_size, shuffle=shuffle,
                          num_workers=num_workers)


def load_colored_mnist_cnc(args,transform = None):
    """
    Default dataloader setup for Colored MNIST
    Args:
    - args (argparse): Experiment arguments
    - transform (torchvision.transforms): Image transformations
    Returns:
    - (train_loader, test_loader): Tuple of dataloaders for train and test
    """
    mnist_train = torchvision.datasets.MNIST(root='./data', 
                                             train=True, download=True)
    mnist_test = torchvision.datasets.MNIST(root='./data', 
                                            train=False, download=True)

    transform = (transforms.Compose([transforms.Resize(40),
                                     transforms.RandomCrop(32, padding=0),
                                     transforms.Normalize((0.5, 0.5, 0.5),
                                                          (0.5, 0.5, 0.5))])
                 if transform is None else transform)
    
    # Split original train set into train and val
    train_indices, val_indices = train_val_split(mnist_train, 
                                                 0.2,
                                                 args.seed)
    train_data = mnist_train.data[train_indices]
    train_targets = mnist_train.targets[train_indices]
    val_data = mnist_train.data[val_indices]
    val_targets = mnist_train.targets[val_indices]
    
    colored_mnist_train = ColoredMNIST_(data=train_data,
                                       targets=train_targets,
                                       train_classes=[[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]],
                                       train=True,
                                       p_correlation=0.995,
                                       test_shift='random',
                                       cmap='hsv',
                                       transform=transform,
                                       flipped=False,
                                       args=args)
    # Val set is setup with same data distribution as test set by convention.
    colored_mnist_val = None
    if len(val_data) > 0:
        colored_mnist_val = ColoredMNIST_(data=val_data, targets=val_targets,
                                         train_classes=[[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]],
                                         train=False,
                                         p_correlation=0.995,
                                         test_shift='random',
                                         cmap='hsv',
                                         transform=transform,
                                         flipped=False,
                                         args=args)
        
    test_cmap = 'hsv'#args.data_cmap #if args.test_cmap == '' else args.test_cmap
    test_p_corr = 0.995#args.p_correlation if args.test_cmap == '' else 1.0
    colored_mnist_test = ColoredMNIST_(data=mnist_test.data,
                                      targets=mnist_test.targets,
                                      train_classes=[[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]],
                                      train=False,
                                      p_correlation=test_p_corr,
                                      test_shift='random',
                                      cmap=test_cmap,
                                      transform=transform,
                                      flipped=False,
                                      args=args)
    
    return colored_mnist_train, colored_mnist_val, colored_mnist_test


"""Build environments."""
import attr
import numpy as np
import torch
from torchvision import datasets


def get_envs(cuda=True, flags=None):

    if flags is None:  # configure data generation like in original IRM paper
        @attr.s
        class DefaultFlags(object):
            # train_env_1__label_noise = attr.ib(default=0.0)
            # train_env_2__label_noise = attr.ib(default=0.0)
            # test_env__label_noise = attr.ib(default=0.0)
            # label_noise = attr.ib(default=0.0)
            train_env_1__color_noise= attr.ib(default=0.2)
            train_env_2__color_noise = attr.ib(default=0.5)
            test_env__color_noise = attr.ib(default=0.9)
            color_noise = attr.ib(default=0.2)
            label_noise  = attr.ib(default=0.0)
        flags = DefaultFlags()
        
    def color_grayscale_arr(images, colors):
        assert images.ndim == 3 and images.shape[1:] == (28, 28), "Input must be of shape (N, 28, 28)"
        assert colors.ndim == 1 and colors.shape[0] == images.shape[0], "Colors tensor must have the same number of elements as the input array"
        dtype = images.dtype
        num_images = images.shape[0]
        colorized_images = torch.zeros((num_images, 3, 28, 28), dtype=dtype)
        for i in range(num_images):
            image = images[i]
            h, w = image.shape
            image = torch.reshape(image, [h, w, 1])
            red = colors[i]
            if red:
                colored_image = torch.cat([image, torch.zeros((h, w, 2), dtype=dtype)], dim=2)
            else:
                colored_image = torch.cat([torch.zeros((h, w, 1), dtype=dtype), image, torch.zeros((h, w, 1), dtype=dtype)], dim=2)
            colorized_images[i] = colored_image.permute(2, 0, 1)
        return colorized_images


    def _make_environment(images, e, labels):

        # NOTE: low e indicates label_noise
        # selected_indices = torch.nonzero((labels == 0) | (labels == 1)).squeeze()
        # labels = labels[(labels == 0) | (labels == 1)]
        # images = images[selected_indices]

        def torch_bernoulli(p, size):
            return (torch.rand(size) < p).float()

        def torch_xor(a, b):
            return (a - b).abs()  # Assumes both inputs are either 0 or 1

        samples = dict()
        # print(images.shape)
        # 2x subsample for computational convenience
        images = images.reshape((-1, 28, 28))#[:, ::2, ::2]
        # print(images.shape)
        # Assign a binary label based on the digit; flip label with probability 0.25
        labels = (labels >= 5).float()
        # samples.update(preliminary_labels=labels)
        # print(e, len(labels))
        label_noise = torch_bernoulli(flags.label_noise, len(labels))  # assign label noise
        labels = torch_xor(labels, label_noise)
        samples.update(final_labels=labels)
        # samples.update(label_noise=label_noise)
        # Assign a color based on the label; flip the color with probability e
        color_noise = torch_bernoulli(e, len(labels))  # assign color noise
        colors = torch_xor(labels, color_noise)

        # samples.update(color_noise=color_noise)
        # Apply the color to the image by zeroing out the other color channel
        # images = torch.stack([images, images], dim=1)
        # images[torch.tensor(range(len(images))), (1 - colors).long(), :, :] *= 0
        # images = (images.float() / 255.)
        # print(colors.shape)
        images = color_grayscale_arr(images, colors)
        colors = torch_xor(colors, torch_bernoulli(flags.color_noise, len(colors)))
        labels = labels[:, None]
        # colors = colors[:, None]
        images = images.float()
        # if cuda and torch.cuda.is_available():
        #     images = images.cuda()
        #     labels = labels.cuda()
        samples.update(colors=colors)
        samples.update(images=images, labels=labels)
        return samples

    mnist = datasets.MNIST('./data', train=True, download=True)
    mnist_train = (mnist.data[:40000], mnist.targets[:40000])
    mnist_val = (mnist.data[40000:], mnist.targets[40000:])

    rng_state = np.random.get_state()
    np.random.shuffle(mnist_train[0].numpy())
    np.random.set_state(rng_state)
    np.random.shuffle(mnist_train[1].numpy())

    # envs = [
    #     _make_environment(mnist_train[0][:30000], mnist_train[1][:30000], flags.train_env_1__color_noise),
    #     _make_environment(mnist_train[0][30000:], mnist_train[1][30000:], flags.train_env_2__color_noise),
    #     _make_environment(mnist_val[0], mnist_val[1], flags.test_env__color_noise)
    # ]
    envs = [
        _make_environment(mnist_train[0][:30000], flags.train_env_1__color_noise, mnist_train[1][:30000]),
        _make_environment(mnist_train[0][30000:], flags.train_env_2__color_noise, mnist_train[1][30000:]),
        _make_environment(mnist_val[0], flags.test_env__color_noise, mnist_val[1])
    ]
    return envs


def get_envs_with_indices():
    """Return IRM envs but with indices and environment indicators."""
    envs = get_envs()
    examples_so_far = 0
    for i, env in enumerate(envs):
        num_examples = len(env['images'])
        env['idx'] = idx = torch.tensor(
            np.arange(examples_so_far, examples_so_far + num_examples),
            dtype=torch.int32
        )
        examples_so_far += num_examples
        # here "env" is a label indicating which env each example belongs to
        env['env'] = torch.tensor(i * np.ones_like(env['idx']), dtype=torch.uint8)
    return envs


def split_by_noise(env, noise_var='label'):
    assert noise_var in ('label', 'color'), 'Unexpected noise variable.'
    noise_name = '%s_noise' % noise_var
    clean_idx = (env[noise_name] == 0.)
    noisy_idx = (env[noise_name] == 1.)
    from copy import deepcopy
    clean_env, noisy_env = deepcopy(env), deepcopy(env)
    for k, v in clean_env.items():
        if v.numel() > 1:
            clean_env[k] = v[clean_idx]
    for k, v in noisy_env.items():
        if v.numel() > 1:
            noisy_env[k] = v[noisy_idx]
    return clean_env, noisy_env



"""Build environments."""
import attr
import numpy as np
import torch
from torchvision import datasets


# def get_envs(cuda=True, flags=None):
#     if flags is None:  # configure data generation like in original IRM paper
#         @attr.s
#         class DefaultFlags(object):
#             train_env_1__color_noise = attr.ib(default=0.2)
#             train_env_2__color_noise = attr.ib(default=0.1)
#             test_env__color_noise = attr.ib(default=0.9)
#             color_noise = attr.ib(default=0.3)
#             label_noise = attr.ib(default=0.25)
#         flags = DefaultFlags()
        
#     def _make_environment(images, e, labels):
#         def torch_bernoulli(p, size):
#             return (torch.rand(size) < p).float()
        
#         def torch_xor(a, b):
#             return (a-b).abs() # Assumes both inputs are either 0 or 1

#         samples = dict()
#         images = images.reshape((-1, 28, 28))[:, ::2, ::2]
#         labels = (labels < 5).float()
#         samples.update(preliminary_labels=labels)
#         label_noise = torch_bernoulli(e, len(labels)) # assign label noise
#         labels = torch_xor(labels, label_noise)
#         samples.update(final_labels=labels)
#         samples.update(label_noise=label_noise)
#         color_noise = torch_bernoulli(flags.color_noise, len(labels)) # assign color noise
#         colors = torch_xor(labels, color_noise)
#         samples.update(colors=colors)
#         samples.update(color_noise=color_noise)
#         images = torch.stack([images, images], dim=1)
#         images[torch.tensor(range(len(images))), (1-colors).long(), :, :] *= 0
#         images = (images.float() / 255.)
#         labels = labels[:, None]
#         if cuda and torch.cuda.is_available():
#             images = images.cuda()
#             labels = labels.cuda()
#         samples.update(images=images, labels=labels)
#         return samples

#     mnist = datasets.MNIST('./data/mnist', train=True, download=True)
#     mnist_train = (mnist.data[:50000], mnist.targets[:50000])
#     mnist_val = (mnist.data[50000:], mnist.targets[50000:])
    
#     rng_state = np.random.get_state()
#     np.random.shuffle(mnist_train[0].numpy())
#     np.random.set_state(rng_state)
#     np.random.shuffle(mnist_train[1].numpy())
    
#     envs = [
#     _make_environment(mnist_train[0][::2] , flags.train_env_1__color_noise, mnist_train[1][::2]),#, flags.train_env_1__label_noise),
#     _make_environment(mnist_train[0][1::2], flags.train_env_1__color_noise, mnist_train[1][1::2]),#, flags.train_env_2__label_noise),
#     _make_environment(mnist_val[0],  flags.test_env__color_noise, mnist_val[1])]#, flags.test_env__label_noise)]
    
#     return envs


# def get_envs_with_indices():
#     envs = get_envs()
#     examples_so_far = 0
#     for i, env in enumerate(envs):
#         num_examples = len(env['images'])
#         env['idx'] = idx = torch.tensor(
#             np.arange(examples_so_far, examples_so_far + num_examples),
#             dtype=torch.int32)
        
#         examples_so_far += num_examples
#         env['env'] = torch.tensor(i * np.ones_like(env['idx']), dtype=torch.uint8)
#     return envs


# def split_by_noise(env, noise_var='label'):
#     assert noise_var in ('label', 'color'), 'Unexpected noise variable.'
#     noise_name = '%s_noise' % noise_var
#     clean_idx = (env[noise_name] == 0.)
#     noisy_idx = (env[noise_name] == 1.)
#     from copy import deepcopy
#     clean_env, noisy_env = deepcopy(env), deepcopy(env)
#     for k, v in clean_env.items():
#         if v.numel() > 1:
#             clean_env[k] = v[clean_idx]
#     for k, v in noisy_env.items():
#         if v.numel() > 1:
#             noisy_env[k] = v[noisy_idx]
#     return clean_env, noisy_env

import torch
from torch.utils.data import Dataset, DataLoader

# 定义一个自定义的Dataset类
class EnvLoaderDataset(Dataset):
    def __init__(self, images, colors, final_labels):
        self.images = images
        self.colors = colors
        self.labels = final_labels

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        return self.images[index], int(self.colors[index]), self.labels[index]
    
def load_mcolor():
    envs = get_envs(flags=None)
    train_dataset = EnvLoaderDataset(envs[0]['images'], envs[0]['colors'], envs[0]['final_labels'])
    eval_dataset = EnvLoaderDataset(envs[1]['images'], envs[1]['colors'], envs[1]['final_labels'])
    test_dataset = EnvLoaderDataset(envs[2]['images'], envs[2]['colors'], envs[2]['final_labels'])
    return train_dataset, eval_dataset, test_dataset


