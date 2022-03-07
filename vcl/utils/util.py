import os
import sys
import time
import math
from datetime import datetime
import random
import logging
from collections import OrderedDict
import numpy as np
import cv2
import torch
from torchvision.utils import make_grid
from shutil import get_terminal_size
import mmcv
from PIL import Image
import copy
import matplotlib.pyplot as plt
from mmcv.runner import load_state_dict


def show_cam_on_image(img, mask):
    heatmap = cv2.applyColorMap(np.uint8(mask), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    cam = heatmap + np.float32(img)
    cam = cam / np.max(cam)
    return np.uint8(255 * cam)

def tensor2img(tensor, out_type=np.uint8, min_max=(0, 1), norm_mode='0-1'):
    '''
    Converts a torch Tensor into an image Numpy array
    Input: 4D(B,(3/1),H,W), 3D(C,H,W), or 2D(H,W), any range, RGB channel order
    Output: 3D(H,W,C) or 2D(H,W), [0,255], np.uint8 (default)
    '''
    if norm_mode == '0-1':
        tensor = tensor.squeeze().float().cpu() # clamp
        tensor = (tensor - min_max[0]) / (min_max[1] - min_max[0])  # to range [0,1]
    elif norm_mode == 'mean-std':
        tensor = tensor.squeeze().float().cpu() # clamp
        mean=torch.tensor([123.675, 116.28, 103.53]).reshape(3,1,1)
        std=torch.tensor([58.395, 57.12, 57.375]).reshape(3,1,1)
        tensor = (tensor * std) + mean
        tensor = tensor.clamp(0,255)
    else:
        tensor = tensor.squeeze().float().cpu().clamp(0, 1)
        tensor = tensor * 255

    n_dim = tensor.dim()
    if n_dim == 4:
        n_img = len(tensor)
        img_np = make_grid(tensor, nrow=int(math.sqrt(n_img)), normalize=False).numpy()
        img_np = np.transpose(img_np[[2, 1, 0], :, :], (1, 2, 0))  # HWC, BGR
    elif n_dim == 3:
        img_np = tensor.numpy()
        img_np = np.transpose(img_np[[2, 1, 0], :, :], (1, 2, 0))  # HWC, BGR
    elif n_dim == 2:
        img_np = tensor.numpy()
    else:
        raise TypeError(
            'Only support 4D, 3D and 2D tensor. But received with dimension: {:d}'.format(n_dim))
    if out_type == np.uint8:
        img_np = (img_np * 255).round() if norm_mode == '0-1' else (img_np ).round()
        # Important. Unlike matlab, numpy.unit8() WILL NOT round by default.
    return img_np.astype(out_type)



