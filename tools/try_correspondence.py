import _init_paths
import cv2

# OpenCV setup
cv2.namedWindow('Source')
cv2.namedWindow('Target')


from argparse import ArgumentParser
from vcl.models.common.utils import make_mask

import numpy as np
import torch.nn.functional as F

import mmcv
import torch


from vcl.models.builder import build_backbone, build_model
from mmcv.runner import load_checkpoint
from vcl.models.common import pad_divide_by, unpad
from mmcv import Config
from vcl.datasets.pipelines import Compose


parser = ArgumentParser()
parser.add_argument('--config', default='/home/lr/project/vcl/configs/train/local/eval/try_corr.py')
parser.add_argument('--src_image', default='/home/lr/dataset/DAVIS/JPEGImages/480p/monkeys/00001.jpg')
parser.add_argument('--tar_image', default='/home/lr/dataset/DAVIS/JPEGImages/480p/monkeys/00001.jpg')
parser.add_argument('--resize', type=bool, default=True)
parser.add_argument('--mask', type=bool, default=False)


parser.add_argument('--model', default='saves/propagation_model.pth')
args = parser.parse_args()

cfg = Config.fromfile(args.config)

# Reading stuff
src_image = mmcv.imread(args.src_image, channel_order='rgb')
tar_image = mmcv.imread(args.tar_image, channel_order='rgb')


if args.resize:
    src_image = cv2.resize(src_image, (384,384))
    tar_image = cv2.resize(tar_image, (384,384))


show_src = cv2.cvtColor(src_image, cv2.COLOR_RGB2BGR)
show_tar = cv2.cvtColor(tar_image, cv2.COLOR_RGB2BGR)

pad_size = 8

# remember the normalization!
data_pipeline = Compose(cfg.val_pipeline)

data = {
        'imgs': [src_image, tar_image],
        'modality': 'RGB',
        'num_clips': 1,
        'num_proposals':1,
        'clip_len': 2
        } 

data = data_pipeline(data)

src_im_th = data['imgs'][:,:,0].cuda()
tar_im_th = data['imgs'][:,:,1].cuda()

# Inputs need to have dimensions as multiples of pad_size
src_im_th, pads = pad_divide_by(src_im_th, pad_size)
tar_im_th, _ = pad_divide_by(tar_im_th, pad_size)

# Mask input is not crucial to getting a good correspondence
# we are just using an empty mask here
b, _, h, w = src_im_th.shape
empty_mask = torch.zeros((b, 1, h, w)).cuda()

changed = True
click = (0, 0)
nh, nw = h//pad_size, w//pad_size
transfer_mask = torch.zeros((b, 1, nh, nw)).cuda()

def mouse_callback(event, x, y, flags, param):
    global changed, click
    # Changing modes
    if event == cv2.EVENT_LBUTTONDOWN:
        changed = True
        click = (x, y)
        transfer_mask.zero_()
        transfer_mask[0,0,y//pad_size,x//pad_size] = 1

cv2.setMouseCallback('Source', mouse_callback)


def comp_binary(image, mask):
    # Increase the brightness a bit
    mask = (mask*2).clip(0, 1)
    color_mask = np.zeros_like(image, dtype=np.uint8)
    color_mask[:,:,2] = 255
    if len(mask.shape) == 2:
        mask = mask[:,:,None]
    image_dim = image*(1-mask)*0.7 + mask*image*0.3
    comp = (image_dim + color_mask*mask*0.7).astype(np.uint8)

    return comp

# loading a pretrained propagation network as correspondence network
model = build_model(cfg.model).cuda()
model.init_weights()
model.eval()

ckpt = cfg.eval_config.checkpoint_path
if ckpt is not None:
    _ = load_checkpoint(model, ckpt, map_location='cpu')

# We can precompute the affinity matrix (H/pad_size * W/pad_size) * (H/pad_size * W/pad_size)
# pad_size is the encoder stride
if args.mask:
    mask = make_mask(48, 9)
else:
    mask = None
corr = model.get_corrspondence(src_im_th, tar_im_th, mask=mask).transpose(1,2)


# Generate the transfer mask
# This mask is considered as our "feature" to be transferred using the affinity matrix
# A feature vectors can also be used (i.e. channel size > 1)


def match(W, transfer_feat):
    # This is mostly just torch.bmm(features, affinity)
    transferred = torch.bmm(transfer_feat.flatten(-2), W).view(b, 1, nh, nw)
    # Upsample pad_size stride image to original size
    transferred = F.interpolate(transferred, scale_factor=pad_size, mode='bilinear', align_corners=False)
    # Remove padding introduced at the beginning
    transferred = unpad(transferred, pads)
    return transferred



while 1:
    if changed:
        click_map_vis = F.interpolate(transfer_mask, scale_factor=pad_size, mode='bilinear', align_corners=False)
        click_map_vis = unpad(click_map_vis, pads)
        click_map_vis = click_map_vis[0,0].cpu().numpy()
        attn_map = match(corr, transfer_mask)
        attn_map = attn_map/(attn_map.max()+1e-6)
        # Scaling for visualization
        attn_map = attn_map.detach()[0,0].cpu().numpy()

        tar_vis = comp_binary(show_tar, attn_map)
        src_vis = comp_binary(show_src, click_map_vis)

        cv2.imshow('Source', src_vis)
        cv2.imshow('Target', tar_vis)
        changed = False

    k = cv2.waitKey(1) & 0xFF
    if k == 27:
        break

cv2.destroyAllWindows()