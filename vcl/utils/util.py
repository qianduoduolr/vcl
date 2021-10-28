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

def tensor2img(tensor, out_type=np.uint8, min_max=(0, 1)):
    '''
    Converts a torch Tensor into an image Numpy array
    Input: 4D(B,(3/1),H,W), 3D(C,H,W), or 2D(H,W), any range, RGB channel order
    Output: 3D(H,W,C) or 2D(H,W), [0,255], np.uint8 (default)
    '''
    tensor = tensor.squeeze().float().cpu() # clamp
    # tensor = (tensor - min_max[0]) / (min_max[1] - min_max[0])  # to range [0,1]

    mean=torch.tensor([123.675, 116.28, 103.53]).reshape(3,1,1)
    std=torch.tensor([58.395, 57.12, 57.375]).reshape(3,1,1)
    tensor = (tensor * std) + mean

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
        img_np = (img_np).round()
        # Important. Unlike matlab, numpy.unit8() WILL NOT round by default.
    return img_np.astype(out_type)

def visualize_att(imgs, att, idx, feat_size=25, patch_size=5, dst_path='/home/lr/project/vcl/output/vis'):
    bsz, num_clips, t, c, w, h = imgs.shape

    dst_path = os.path.join(dst_path, str(idx))
    mmcv.mkdir_or_exist(dst_path)

    for i in range(t):
        img = imgs[:,:,i].reshape(c,w,h)
        out = tensor2img(img)
        # out = cv2.resize(out, (feat_size,feat_size))
        cv2.imwrite(os.path.join(dst_path,f'{i}.jpg'), out)

    att = att.cpu().detach().numpy()[0,0]
    x = random.randint(patch_size // 2+1, feat_size - patch_size // 2 -1)
    y = random.randint(patch_size // 2+1, feat_size - patch_size // 2 -1)


    attr = (att[:, x, y] * 255).astype(np.uint8).reshape(patch_size, patch_size)
    att_out = np.zeros((feat_size, feat_size)).astype(np.uint8)
    att_out_query = np.zeros((feat_size, feat_size)).astype(np.uint8)
    att_out_combine = np.zeros((feat_size, feat_size)).astype(np.uint8)


    att_out[x-patch_size // 2:x+patch_size // 2+1, y-patch_size // 2:y+patch_size // 2+1] = attr
    att_out_combine[x-patch_size // 2:x+patch_size // 2+1, y-patch_size // 2:y+patch_size // 2+1] = attr

    att_out_query[x,y] = 255
    att_out_combine[x,y] = 255

    att_map = cv2.applyColorMap(att_out, cv2.COLORMAP_BONE)
    attr = cv2.applyColorMap(attr, cv2.COLORMAP_BONE)
    att_query_map = cv2.applyColorMap(att_out_query, cv2.COLORMAP_BONE)
    att_combine = cv2.applyColorMap(att_out_combine, cv2.COLORMAP_BONE)

    cv2.imwrite(os.path.join(dst_path,'att_all.jpg'), att_map)
    cv2.imwrite(os.path.join(dst_path,'att.jpg'), attr)
    cv2.imwrite(os.path.join(dst_path,'att_all_query.jpg'), att_query_map)
    cv2.imwrite(os.path.join(dst_path,'att_all_combine.jpg'), att_combine)






