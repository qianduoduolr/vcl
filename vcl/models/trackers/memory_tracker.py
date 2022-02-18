import os.path as osp
from collections import *

import mmcv
import tempfile
from mmcv.runner import auto_fp16, load_checkpoint
from dall_e  import map_pixels, unmap_pixels, load_model

from vcl.models.common.correlation import *

from ..base import BaseModel
from ..builder import build_backbone
from ..registry import MODELS
from vcl.utils import *

import torch.nn as nn
import torch.nn.functional as F
from .modules import *

@MODELS.register_module()
class Memory_Tracker(BaseModel):
    def __init__(self,
                 downsample_rate=4,
                 test_cfg=None,
                 train_cfg=None
                 ):
        """ MAST  (CVPR2020)

        Args:
            backbone ([type]): [description]
            test_cfg ([type], optional): [description]. Defaults to None.
            train_cfg ([type], optional): [description]. Defaults to None.
        """
        super().__init__()
        # Model options
        self.p = 0.3
        self.C = 7

        # self.backbone = build_backbone(backbone)
        self.backbone = ResNet18(3)
        self.post_convolution = nn.Conv2d(256, 64, 3, 1, 1)
        self.D = downsample_rate

        self.test_cfg = test_cfg
        self.train_cfg = train_cfg
        
        self.test_cfg.ref = 2

        self.R = 12 # radius

        self.colorizer = Colorizer(self.D, self.R, self.C)
        
    def forward_colorization(self, rgb_r, quantized_r, rgb_t, ref_index=None,current_ind=None):
        feats_r = [self.post_convolution(self.backbone(rgb)) for rgb in rgb_r]
        feats_t = self.post_convolution(self.backbone(rgb_t))

        quantized_t = self.colorizer(feats_r, feats_t, quantized_r, ref_index, current_ind)
        return quantized_t
    

    def forward_test(self, imgs, ref_seg_map, img_meta,
                save_image=False,
                save_path=None,
                iteration=None):
        num_frame = imgs.shape[3]
        
        imgs = [imgs[:,0,:,i] for i in range(num_frame)]
        annotations = [ref_seg_map[:,None,:]]

        all_seg_preds = []
        seg_preds = []
        
        N = len(imgs)
        outputs = [annotations[0].contiguous()]

        for i in range(N-1):
            mem_gap = 2
            # ref_index = [i]
            if self.test_cfg.ref == 0:
                ref_index = list(filter(lambda x: x <= i, [0, 5])) + list(filter(lambda x:x>0,range(i,i-mem_gap*3,-mem_gap)))[::-1]
                ref_index = sorted(list(set(ref_index)))
            elif self.test_cfg.ref == 1:
                ref_index = [0] + list(filter(lambda x: x > 0, range(i, i - mem_gap * 3, -mem_gap)))[::-1]
            elif self.test_cfg.ref == 2:
                ref_index = [i]
            else:
                raise NotImplementedError

            rgb_0 = [imgs[ind] for ind in ref_index]
            rgb_1 = imgs[i+1]

            anno_0 = [outputs[ind] for ind in ref_index]

            _, _, h, w = anno_0[0].size()

            with torch.no_grad():
                _output = self.forward_colorization(rgb_0, anno_0, rgb_1, ref_index, i+1)
                _output = F.interpolate(_output, (h,w), mode='bilinear')

                output = torch.argmax(_output, 1, keepdim=True).float()
                outputs.append(output)

            ###
            pad =  ((0,0), (0,0))
            if i == 0:
                # output first mask
                out_img = anno_0[0][0, 0].cpu().numpy().astype(np.uint8)
                out_img = np.pad(out_img, pad, 'edge').astype(np.uint8)
                seg_preds.append(out_img[None,:])
                
            out_img = output[0, 0].cpu().numpy().astype(np.uint8)
            out_img = np.pad(out_img, pad, 'edge').astype(np.uint8)
            seg_preds.append(out_img[None,:])
        
        
        seg_preds = np.stack(seg_preds, axis=1)
        all_seg_preds.append(seg_preds)
        
        if self.test_cfg.get('save_np', False):
            if len(all_seg_preds) > 1:
                return [all_seg_preds]
            else:
                return [all_seg_preds[0]]
        else:
            if len(all_seg_preds) > 1:
                all_seg_preds = np.stack(all_seg_preds, axis=1)
            else:
                all_seg_preds = all_seg_preds[0]
            # unravel batch dim
            return list(all_seg_preds)

    def forward_train(self, images_lab, imgs):
        
        bsz, n, c, t, h, w = imgs.shape
        
        images_lab_gt = [images_lab[:,0,:,i].clone() for i in range(t)]
        images_lab = [images_lab[:,0,:,i] for i in range(t)]
        imgs = [imgs[:,0,:,i] for i in range(t)]
        
        _, ch = self.dropout2d_lab(images_lab)
        
        losses = {}
        sum_loss, err_maps = self.compute_lphoto(images_lab, images_lab_gt, ch)
        losses['l1_loss'] = sum_loss
        
        return losses

        

    def dropout2d_lab(self, arr): # drop same layers for all images
        if not self.training:
            return arr

        drop_ch_num = int(np.random.choice(np.arange(1, 2), 1))
        drop_ch_ind = np.random.choice(np.arange(1,3), drop_ch_num, replace=False)

        for a in arr:
            for dropout_ch in drop_ch_ind:
                a[:, dropout_ch] = 0
            a *= (3 / (3 - drop_ch_num))

        return arr, drop_ch_ind # return channels not masked
    
    def compute_lphoto(self, image_lab, images_lab_gt, ch):
        b, c, h, w = image_lab[0].size()

        ref_x = [lab for lab in image_lab[:-1]]   # [im1, im2, im3]
        ref_y = [rgb[:,ch] for rgb in images_lab_gt[:-1]]  # [y1, y2, y3]
        tar_x = image_lab[-1]  # im4
        tar_y = images_lab_gt[-1][:,ch]  # y4


        outputs = self.forward_colorization(ref_x, ref_y, tar_x, [0,2], 4)   # only train with pairwise data

        outputs = F.interpolate(outputs, (h, w), mode='bilinear')
        loss = F.smooth_l1_loss(outputs*20, tar_y*20, reduction='mean')

        err_maps = torch.abs(outputs - tar_y).sum(1).detach()

        return loss, err_maps