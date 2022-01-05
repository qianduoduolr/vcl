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

def visualize_att(imgs, att, idx, query_idx=None, feat_size=25, patch_size=5, dst_path='/home/lr/project/vcl/output/vis', norm_mode='mean-std', color='bone'):
    bsz, num_clips, t, c, w, h = imgs.shape

    dst_path = os.path.join(dst_path, str(idx))
    mmcv.mkdir_or_exist(dst_path)

    outs = []
    for i in range(t):
        img = imgs[:,:,i].reshape(c,w,h)
        out = tensor2img(img, norm_mode=norm_mode)
        # out = cv2.resize(out, (feat_size,feat_size))
        cv2.imwrite(os.path.join(dst_path,f'{i}.jpg'), out)
        outs.append(cv2.cvtColor(out, cv2.COLOR_BGR2RGB))

    if query_idx == None:
        if patch_size != -1:
            att = att.cpu().detach().numpy()[0,0]

            x = random.randint(patch_size // 2+1, feat_size - patch_size // 2 -1)
            y = random.randint(patch_size // 2+1, feat_size - patch_size // 2 -1)
            attr = (att[:, x, y]).reshape(patch_size, patch_size)
            attr = ((attr - attr.min()) * 255 / (attr.max() - attr.min())).astype(np.uint8)

            att_out = np.zeros((feat_size, feat_size)).astype(np.uint8)
            att_out_query = np.zeros((feat_size, feat_size)).astype(np.uint8)
            att_out[x-patch_size // 2:x+patch_size // 2+1, y-patch_size // 2:y+patch_size // 2+1] = attr


            att_out_query[x,y] = 255

        else:
            att = att.cpu().detach().numpy()
            query_idx = random.randint(feat_size * feat_size -1)

            attr = (att[:, query_idx.item()]).reshape(feat_size, feat_size)
            attr = ((attr - attr.min()) * 255 / (attr.max() - attr.min())).astype(np.uint8)

            att_out = attr
            att_out_query = np.zeros(feat_size * feat_size).astype(np.uint8)
            att_out_query[query_idx.item()] = 255
            att_out_query = att_out_query.reshape(feat_size, feat_size)

    else:
        if patch_size == -1:
            att = att.cpu().detach().numpy()
            # x = query_idx.item() // feat_size
            # y = query_idx.item() % feat_size

            attr = (att[:, query_idx.item()]).reshape(feat_size, feat_size)
            attr = ((attr - attr.min()) * 255 / (attr.max() - attr.min())).astype(np.uint8)

            att_out = attr
            att_out_query = np.zeros(feat_size * feat_size).astype(np.uint8)
            att_out_query[query_idx.item()] = 255
            att_out_query = att_out_query.reshape(feat_size, feat_size)
        else:
            att = att.cpu().detach().numpy()
            x = query_idx.item() // feat_size
            y = query_idx.item() % feat_size

            att_out = np.zeros((feat_size, feat_size))
            attr = (att[:, query_idx.item()]).reshape(feat_size, feat_size)

            att_out[max(x-patch_size // 2,0):min(x+patch_size // 2+1,feat_size), max(y-patch_size // 2, 0):min(y+patch_size // 2+1, feat_size)] = attr[max(x-patch_size // 2,0):min(x+patch_size // 2+1,feat_size), max(y-patch_size // 2, 0):min(y+patch_size // 2+1, feat_size)]

            att_out = ((att_out - att_out.min()) * 255 / (att_out.max() - att_out.min())).astype(np.uint8)
            attr = ((attr - attr.min()) * 255 / (attr.max() - attr.min())).astype(np.uint8)

            att_out_query = np.zeros(feat_size * feat_size).astype(np.uint8)
            att_out_query[query_idx.item()] = 255
            att_out_query = att_out_query.reshape(feat_size, feat_size)


    
    resized_1 = cv2.resize(att_out, (w,h))
    resized_2 = cv2.resize(att_out_query, (w,h), cv2.INTER_NEAREST)

    if color == 'bone':
        att_map = cv2.applyColorMap(att_out, cv2.COLORMAP_BONE)
        attr = cv2.applyColorMap(attr, cv2.COLORMAP_BONE)
        att_query_map = cv2.applyColorMap(att_out_query, cv2.COLORMAP_BONE)

        img_ = Image.fromarray(copy.deepcopy(outs[0])).convert('RGBA')
        resized_1 = Image.fromarray(resized_1).convert('RGBA')
        blend_out1 = Image.blend(img_, resized_1, 0.8)
        blend_out1.save(os.path.join(dst_path,'blend_ref.png'))

        img_ = Image.fromarray(copy.deepcopy(outs[1])).convert('RGBA')
        resized_1 = Image.fromarray(resized_2).convert('RGBA')
        blend_out2 = Image.blend(img_, resized_1, 0.8)
        blend_out2.save(os.path.join(dst_path,'blend_tar.png'))
    
    else:
        blend_out1 = show_cam_on_image(outs[0], resized_1)
        cv2.imwrite(os.path.join(dst_path,'blend_ref.png'), blend_out1)

        blend_out2 = show_cam_on_image(outs[1], resized_2)
        cv2.imwrite(os.path.join(dst_path,'blend_tar.png'), blend_out2)


def moment_update(model, model_ema, m):
    """ model_ema = m * model_ema + (1 - m) model """
    for p1, p2 in zip(model.parameters(), model_ema.parameters()):
        p2.data.mul_(m).add_(1 - m, p1.detach().data)


def make_pbs(exp_name, docker_name):
    pbs_data = ""
    with open('configs/pbs/template.pbs', 'r') as f:
        for line in f:
            line = line.replace('exp_name',f'{exp_name}')
            line = line.replace('docker_name', f'{docker_name}')
            pbs_data += line

    with open(f'configs/pbs/{exp_name}.pbs',"w") as f:
        f.write(pbs_data)

def make_local_config(exp_name):
    config_data = ""
    with open(f'configs/train/local/{exp_name}.py', 'r') as f:
        for line in f:
            line = line.replace('/home/lr','/gdata/lirui')
            line = line.replace('/gdata/lirui/dataset/YouTube-VOS','/dev/shm')
            # line = line.replace('/home/lr/dataset','/home/lr/dataset')
            config_data += line

    with open(f'configs/train/ypb/{exp_name}.py',"w") as f:
        f.write(config_data)


@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [torch.ones_like(tensor)
        for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output

def shift_dim(x, src_dim=-1, dest_dim=-1, make_contiguous=True):
    n_dims = len(x.shape)
    if src_dim < 0:
        src_dim = n_dims + src_dim
    if dest_dim < 0:
        dest_dim = n_dims + dest_dim

    assert 0 <= src_dim < n_dims and 0 <= dest_dim < n_dims

    dims = list(range(n_dims))
    del dims[src_dim]

    permutation = []
    ctr = 0
    for i in range(n_dims):
        if i == dest_dim:
            permutation.append(src_dim)
        else:
            permutation.append(dims[ctr])
            ctr += 1
    x = x.permute(permutation)
    if make_contiguous:
        x = x.contiguous()
    return x


def view_range(x, i, j, shape):
    shape = tuple(shape)

    n_dims = len(x.shape)
    if i < 0:
        i = n_dims + i

    if j is None:
        j = n_dims
    elif j < 0:
        j = n_dims + j

    assert 0 <= i < j <= n_dims

    x_shape = x.shape
    target_shape = x_shape[:i] + shape + x_shape[j:]
    return x.view(target_shape)

    
def tensor_slice(x, begin, size):
    assert all([b >= 0 for b in begin])
    size = [l - b if s == -1 else s
            for s, b, l in zip(size, begin, x.shape)]
    assert all([s >= 0 for s in size])

    slices = [slice(b, b + s) for b, s in zip(begin, size)]
    return x[slices]


def PCA_numpy(X, embed_dim):
    bsz, c, h, w = X.shape
    X = X.permute(0, 2, 3, 1).flatten(0,2).detach().cpu().numpy()
    X_new = X - np.mean(X, axis=0)
    # SVD
    U, Sigma, Vh = np.linalg.svd(X_new, full_matrices=False, compute_uv=True)
    X_pca_svd = np.dot(X_new, (Vh.T)[:,:embed_dim])
    X_out = torch.from_numpy(X_pca_svd).cuda()
    out = X_out.reshape(bsz, h, w, embed_dim)

    return out.permute(0, 3, 1, 2)


def PCA_torch_v1(X, embed_dim):
    bsz, c, h, w = X.shape
    X = X.permute(0, 2, 3, 1).flatten(1,2)
    U, S, V = torch.pca_lowrank(X, q=embed_dim, center=True, niter=2)
    X = torch.matmul(X, V)
    
    return X.permute(0, 2, 1).reshape(bsz, embed_dim, h, w)


def PCA_torch_v2(X, embed_dim):
    bsz, c, h, w = X.shape
    X = X.permute(0, 2, 3, 1).flatten(1,2)
    X = X - X.mean(dim=1, keepdim=True)
    U, S, V = torch.svd(X)
    X = torch.matmul(X, V[:,:,:embed_dim])
    
    return X.permute(0, 2, 1).reshape(bsz, embed_dim, h, w)


def make_mask(size, t_size, eq=True):
    masks = []
    for i in range(size):
        for j in range(size):
            mask = torch.zeros((size, size)).cuda()
            if eq:
                mask[max(0, i-t_size):min(size, i+t_size+1), max(0, j-t_size):min(size, j+t_size+1)] = 1
            else:
                mask[max(0, i-t_size):min(size, i+t_size+1), max(0, j-t_size):min(size, j+t_size+1)] = 0.7
                mask[i,j] = 1
                
            masks.append(mask.reshape(-1))
    return torch.stack(masks)