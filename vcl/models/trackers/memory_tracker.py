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
                 backbone,
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
        self.feature_extraction = ResNet18(3)
        self.post_convolution = nn.Conv2d(256, 64, 3, 1, 1)
        self.D = 4

        self.test_cfg = test_cfg
        self.train_cfg = train_cfg
        
        self.test_cfg.ref = 2

        self.R = 12 # radius

        self.colorizer = Colorizer(self.D, self.R, self.C)
        
    def forward_train(self, rgb_r, quantized_r, rgb_t, ref_index=None,current_ind=None):
        feats_r = [self.post_convolution(self.feature_extraction(rgb)) for rgb in rgb_r]
        feats_t = self.post_convolution(self.feature_extraction(rgb_t))
        
        # feats_r = [self.backbone(rgb) for rgb in rgb_r]
        # feats_t = self.backbone(rgb_t)

        quantized_t = self.colorizer(feats_r, feats_t, quantized_r, ref_index, current_ind)
        return quantized_t
    

    def forward_test(self, imgs, ref_seg_map, img_meta,
                save_image=False,
                save_path=None,
                iteration=None):
        num_frame = imgs.shape[3]
        
        images_rgb = [imgs[:,0,:,i] for i in range(num_frame)]
        annotations = [ref_seg_map[:,None,:]]

        all_seg_preds = []
        seg_preds = []
        
        N = len(images_rgb)
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

            rgb_0 = [images_rgb[ind] for ind in ref_index]
            rgb_1 = images_rgb[i+1]

            anno_0 = [outputs[ind] for ind in ref_index]

            _, _, h, w = anno_0[0].size()

            with torch.no_grad():
                _output = self.forward_train(rgb_0, anno_0, rgb_1, ref_index, i+1)
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