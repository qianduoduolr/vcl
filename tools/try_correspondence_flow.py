import _init_paths
import cv2
import os


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
from vcl.models.common import pad_divide_by, unpad, coords_grid, bilinear_sample
from mmcv import Config
from vcl.datasets.pipelines import Compose


parser = ArgumentParser()
parser.add_argument('--config', default='/home/lr/project/vcl/configs/train/local/eval/try_corr_flow.py')
parser.add_argument('--src_image', default='/home/lr/dataset/DAVIS/JPEGImages/480p/horsejump-stick/00001.jpg')
parser.add_argument('--tar_image_list', default=['/home/lr/dataset/DAVIS/JPEGImages/480p/horsejump-stick/00001.jpg', '/home/lr/dataset/DAVIS/JPEGImages/480p/horsejump-stick/00010.jpg','/home/lr/dataset/DAVIS/JPEGImages/480p/horsejump-stick/00020.jpg'])
parser.add_argument('--resize', type=bool, default=True)
parser.add_argument('--mask', type=bool, default=False)


parser.add_argument('--model', default='saves/propagation_model.pth')
args = parser.parse_args()

cfg = Config.fromfile(args.config)

# temp -> mp -> detco -> st(res50) -> st(res18)

# Reading stuff
vname = args.src_image.split('/')[-2]
src_image = mmcv.imread(args.src_image, channel_order='rgb')
tar_images = [mmcv.imread(i, channel_order='rgb') for i in args.tar_image_list]


if args.resize:
    src_image = cv2.resize(src_image, (384,384))
    tar_images = [cv2.resize(i, (384,384)) for i in tar_images]

show_src = cv2.cvtColor(src_image, cv2.COLOR_RGB2BGR)
show_tars = np.concatenate([cv2.cvtColor(i, cv2.COLOR_RGB2BGR) for i in tar_images], 1)

tar_images.insert(0, src_image)

pad_size = 8

# remember the normalization!
data_pipeline = Compose(cfg.val_pipeline)

data = {
        'imgs': tar_images,
        'modality': 'RGB',
        'num_clips': 1,
        'num_proposals':1,
        'clip_len': len(tar_images)
        } 

data = data_pipeline(data)

src_im_th = data['imgs'][:,:,0].cuda()
tar_im_ths = [data['imgs'][:,:,i].cuda() for i in range(1, data['imgs'].shape[2])]

# Inputs need to have dimensions as multiples of pad_size
src_im_th, pads = pad_divide_by(src_im_th, pad_size)
tar_im_ths = [pad_divide_by(i, pad_size)[0] for i in tar_im_ths ]
tar_im_ths = torch.stack(tar_im_ths, 2)


src_im_ths = src_im_th.unsqueeze(2).repeat(1, 1, tar_im_ths.shape[2], 1, 1)
imgs = torch.cat([tar_im_ths, src_im_ths], 0).permute(2,1,0,3,4).unsqueeze(1)
    


# Mask input is not crucial to getting a good correspondence
# we are just using an empty mask here
b, p, c, _, h, w = imgs.shape

changed = False
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
        transfer_mask[:,0,y//pad_size,x//pad_size] = 1
        

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

ckpt = cfg.eval_config.checkpoint_path
if ckpt is not None:
    _ = load_checkpoint(model, ckpt, map_location='cpu')

model.eval()


# We can precompute the affinity matrix (H/pad_size * W/pad_size) * (H/pad_size * W/pad_size)
# pad_size is the encoder stride
if args.mask:
    mask = make_mask(48, 9)
else:
    mask = None

# b2hw
flow = model(True, imgs=imgs, return_lr=True)[0]

# Generate the transfer mask
# This mask is considered as our "feature" to be transferred using the affinity matrix
# A feature vectors can also be used (i.e. channel size > 1)


def match(flow, transfer_feat):
    B, _ , H, W = flow.shape
    # This is mostly just torch.bmm(features, affinity)
    xx = torch.arange(0, W, device=flow.device)
    yy = torch.arange(0, H, device=flow.device)
    grid = coords_grid(B, xx, yy) + flow  # shape N, 2, H, W
    grid = grid.permute(0, 2, 3, 1)  # shape N, H, W, 2
    
    centroid_lvl = grid.reshape(B * H * W, 1, 1, 2)

    ref_v = transfer_feat.unsqueeze(1).repeat(1, H*W, 1, 1, 1).flatten(0,1)
    v = bilinear_sample(ref_v, centroid_lvl, align_corners=True)
    v = v.view(B, H, W, -1).permute(0, 3, 1, 2)

    # Upsample pad_size stride image to original size
    transferred = F.interpolate(v, scale_factor=pad_size, mode='bilinear', align_corners=False)
    # Remove padding introduced at the beginning
    transferred = unpad(transferred, pads)
    return transferred


while 1:
    if changed:
        click_map_vis = F.interpolate(transfer_mask, scale_factor=pad_size, mode='bilinear', align_corners=False)
        click_map_vis = unpad(click_map_vis, pads)
        click_map_vis = click_map_vis[0,0].cpu().numpy()

        # btij
        attn_maps = match(flow, transfer_mask).detach().cpu()
        attn_maps = attn_maps/(attn_maps.max()+1e-6)
        # Scaling for visualization
        attn_maps = [attn_maps[i,0].numpy() for i in range(attn_maps.shape[0])]
        attn_maps = np.concatenate(attn_maps, 1)

        tar_vis = comp_binary(show_tars, attn_maps)
        src_vis = comp_binary(show_src, click_map_vis)

        out = np.concatenate([src_vis, tar_vis], 1)
        out_file = os.path.join('/home/lr/expdir/VCL/corr_output/flow_raft', f'click_{click[0]}_{click[1]}_{vname}.jpg')
        cv2.imwrite(out_file, out)

        cv2.imshow('Source', src_vis)
        cv2.imshow('Target', tar_vis)
        changed = False

    k = cv2.waitKey(1) & 0xFF
    if k == 27:
        break

cv2.destroyAllWindows()