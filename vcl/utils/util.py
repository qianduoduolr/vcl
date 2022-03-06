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
from vcl.models.common.correlation import *
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

def visualize_att(imgs, atts, idx, abs_att=False, query_idx=None, feat_size=25, patch_size=5, dst_path='/home/lr/project/vcl/output/vis', norm_mode='mean-std', color='bone'):
    bsz, num_clips, t, c, w, h = imgs.shape

    dst_path = os.path.join(dst_path, f'{idx}')
    mmcv.mkdir_or_exist(dst_path)


    atts = list([att.cpu().numpy() for att in atts])
    atts = np.stack(atts, 2)
    atts = atts[0, query_idx.item()]
    if abs_att:
        atts = ((atts - atts.min()) * 255 / (atts.max() - atts.min())).astype(np.uint8)
    else:
        atts = ((atts - atts.min(-1)) * 255 / (atts.max(-1) - atts.min(-1))).astype(np.uint8)
        

    outs = []
    for i in range(t):
        img = imgs[:,:,i].reshape(c,w,h)
        out = tensor2img(img, norm_mode=norm_mode)
        outs.append(out)

    blend_outs = []
    for index, i in enumerate(range(atts.shape[0])):
        
        if patch_size == -1:
            attr = atts[i].reshape(feat_size, feat_size)
            # attr = (att[:, query_idx.item()]).reshape(feat_size, feat_size)
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


        
        resized_ = cv2.resize(att_out, (w,h))
        resized_query = cv2.resize(att_out_query, (w,h), cv2.INTER_NEAREST)


        att_map = cv2.applyColorMap(att_out, cv2.COLORMAP_BONE)
        attr = cv2.applyColorMap(attr, cv2.COLORMAP_BONE)
        att_query_map = cv2.applyColorMap(att_out_query, cv2.COLORMAP_BONE)

        if index == 0:
            img_ = Image.fromarray(copy.deepcopy(outs[index])).convert('RGBA')
            resized_out = Image.fromarray(resized_query).convert('RGBA')
            blend_out_query = Image.blend(img_, resized_out, 0.7)
            blend_out_query = np.array(blend_out_query)
            blend_out_query = np.concatenate([blend_out_query, np.ones((h,7,c+1))*255], 1)
            blend_outs.append(np.array(blend_out_query))
            
        img_ = Image.fromarray(copy.deepcopy(outs[index+1])).convert('RGBA')
        resized_ = Image.fromarray(resized_).convert('RGBA')
        blend_out = Image.blend(img_, resized_, 0.7)
        blend_out = np.array(blend_out)
        blend_out = np.concatenate([blend_out, np.ones((h,7,c+1))*255], 1)
        blend_outs.append(np.array(blend_out))
    
    blend_outs = np.concatenate(blend_outs, 1)
    outs = np.concatenate(outs, 1)
    
    
    cv2.imwrite(os.path.join(dst_path,'out_blend.png'), blend_outs)
    cv2.imwrite(os.path.join(dst_path,'out_img.png'), outs)
    

def moment_update(model, model_ema, m):
    """ model_ema = m * model_ema + (1 - m) model """
    for p1, p2 in zip(model.parameters(), model_ema.parameters()):
        p2.data.mul_(m).add_(1 - m, p1.detach().data)

def copy_params(model, model_test):
    origin_params = {}
    for name, param in model.state_dict().items():
        if name in model_test.state_dict().keys():
            origin_params[name.replace('module.','')] = param.data.detach().cpu()
    load_state_dict(model_test,origin_params, strict=False)

def make_pbs(exp_name, docker_name):
    pbs_data = ""
    with open('/home/lr/project/vcl/configs/pbs/template.pbs', 'r') as f:
        for line in f:
            line = line.replace('exp_name',f'{exp_name}')
            line = line.replace('docker_name', f'{docker_name}')
            pbs_data += line

    with open(f'/home/lr/project/vcl/configs/pbs/{exp_name}.pbs',"w") as f:
        f.write(pbs_data)

def make_local_config(exp_name):
    config_data = ""
    with open(f'/home/lr/project/vcl/configs/train/local/{exp_name}.py', 'r') as f:
        for line in f:
            line = line.replace('/home/lr','/gdata/lirui')
            line = line.replace('/gdata/lirui/dataset/YouTube-VOS','/dev/shm')
            # line = line.replace('/home/lr/dataset','/home/lr/dataset')
            config_data += line

    with open(f'/home/lr/project/vcl/configs/train/ypb/{exp_name}.py',"w") as f:
        f.write(config_data)



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


def video2images(imgs):
    batches, channels, clip_len = imgs.shape[:3]
    if clip_len == 1:
        new_imgs = imgs.squeeze(2).reshape(batches, channels, *imgs.shape[3:])
    else:
        new_imgs = imgs.transpose(1, 2).contiguous().reshape(
            batches * clip_len, channels, *imgs.shape[3:])

    return new_imgs


def images2video(imgs, clip_len):
    batches, channels = imgs.shape[:2]
    if clip_len == 1:
        new_imgs = imgs.unsqueeze(2)
    else:
        new_imgs = imgs.reshape(batches // clip_len, clip_len, channels,
                                *imgs.shape[2:]).transpose(1, 2).contiguous()

    return new_imgs


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def visualize_vqvae(z1_q, z2_q, frame1, frame2, x_rec1, x_rec2, nembed=2048, rescale=True, mask=None):
    frame_vq_1 = z1_q.permute(1,2,0).numpy()
    frame_vq_2 = z2_q.permute(1,2,0).numpy()

    print(len(np.unique(frame_vq_1)))

    if rescale:
        frame_vq_1 = (frame_vq_1 * 255 / nembed).astype(np.uint8)
        frame_vq_2 = (frame_vq_2 * 255 / nembed).astype(np.uint8)
    else:
        frame_vq_1 = (frame_vq_1).astype(np.uint8)
        frame_vq_2 = (frame_vq_2).astype(np.uint8)

    if mask is not None:

        frame_vq_2 = frame_vq_2 * mask

    plt.rcParams['figure.dpi'] = 200

    plt.figure()

    if x_rec1 is not None:
        plt.subplot(3,2,1)
        plt.imshow(frame_vq_1, cmap=plt.get_cmap('jet'))
        plt.subplot(3,2,2)
        plt.imshow(frame_vq_2, cmap=plt.get_cmap('jet'))

        plt.subplot(3,2,3)
        plt.imshow(np.array(frame1))

        plt.subplot(3,2,4)
        plt.imshow(np.array(frame2))

        plt.subplot(3,2,5)
        plt.imshow(np.array(x_rec1))

        plt.subplot(3,2,6)
        plt.imshow(np.array(x_rec2))

        plt.show()
    else:
        plt.subplot(2,2,1)
        plt.imshow(frame_vq_1, cmap=plt.get_cmap('jet'))
        plt.subplot(2,2,2)
        plt.imshow(frame_vq_2, cmap=plt.get_cmap('jet'))

        plt.subplot(2,2,3)
        plt.imshow(np.array(frame1))

        plt.subplot(2,2,4)
        plt.imshow(np.array(frame2))


def visualize_correspondence(z1_q, z2_q, sample_idx, frame1, frame2, scale=32):
    plt.rcParams['figure.dpi'] = 200

    z1_q = z1_q[0].numpy()
    z2_q = z2_q[0].numpy()
    find = False
    count = 0

    while not find:
        x, y = sample_idx % scale, sample_idx // scale

        query = z1_q[y,x]
        m = (z2_q == query).astype(np.uint8) * 255
        count += 1

        if m.max() > 1:
            find = True
        else:
            # sample_idx = random.randint(0, scale*scale -1)
            sample_idx = random.randint(210, 250)
            print('not find, change query')
        
    print(f"find correspodence at {count}")   

    querys_map = np.zeros((scale,scale))
    querys_map[y,x] = 255
    querys_map = querys_map.astype(np.uint8)
    

    plt.figure()
    plt.subplot(2,2,1)
    plt.imshow(querys_map, cmap=plt.get_cmap('jet'))
    plt.subplot(2,2,2)
    plt.imshow(m, cmap=plt.get_cmap('jet'))
    plt.subplot(2,2,3)
    plt.imshow(np.array(frame1))

    plt.subplot(2,2,4)
    plt.imshow(np.array(frame2))
    
def visualize_correspondence_quant(z1_q, z2_q, sample_idx, frame1, frame2, scale=32):
    plt.rcParams['figure.dpi'] = 200
    
    x, y = sample_idx % scale, sample_idx // scale

    querys_map = np.zeros((scale,scale))
    querys_map[y,x] = 255
    querys_map = querys_map.astype(np.uint8)
    
    _, att = non_local_attention(z1_q, [z2_q], flatten=False)
    
    # print(att[0,0, sample_idx].max(), att[0,0,sample_idx].sum())
    
    att = att[0, 0, sample_idx].reshape(scale,scale).detach().cpu().numpy() * 255
    
    # print(att.max())
    att = ((att - att.min()) * 255 / (att.max() - att.min())).astype(np.uint8)
    
    att = cv2.resize(att, (256,256))
    querys_map = cv2.resize(querys_map, (256,256))
    
    att_map = cv2.applyColorMap(att.astype(np.uint8), cv2.COLORMAP_BONE)
    att_query_map = cv2.applyColorMap(querys_map, cv2.COLORMAP_BONE)


    tar = Image.fromarray(frame1).convert('RGBA')
    query = Image.fromarray(att_query_map).convert('RGBA')
    blend_out_query = Image.blend(tar, query, 0.7)
    blend_out_query = np.array(blend_out_query)

    ref = Image.fromarray(frame2).convert('RGBA')
    result = Image.fromarray(att_map).convert('RGBA')
    blend_out_result = Image.blend(ref, result, 0.7)
    blend_out_result = np.array(blend_out_result)


    plt.figure()
    plt.subplot(2,2,1)
    plt.imshow(blend_out_query)
    plt.subplot(2,2,2)
    plt.imshow(blend_out_result)
    plt.subplot(2,2,3)
    plt.imshow(querys_map)
    plt.subplot(2,2,4)
    plt.imshow(att)
    
    # plt.figure()
    # plt.subplot(1,2,1)
    # plt.imshow(querys_map)
    # plt.subplot(1,2,2)
    # plt.imshow(att)

def preprocess_(img, mode='rgb'):
    
    if mode == 'rgb':
        mean=np.array([123.675, 116.28, 103.53])
        std=np.array([58.395, 57.12, 57.375])
    else:
        mean=np.array([50, 0, 0])
        std=np.array([50, 127, 127])
        img = (img / 255.0).astype(np.float32)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2Lab)
    
    # resize
    img = img.astype(np.float32)
    
    mmcv.imnormalize_(img, mean, std, False)
    
    out = torch.from_numpy(img).unsqueeze(0).permute(0, 3, 1, 2)
    return out




