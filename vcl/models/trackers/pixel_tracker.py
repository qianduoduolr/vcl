# Copyright (c) OpenMMLab. All rights reserved.
import numbers
import os.path as osp
from collections import *

import mmcv
from mmcv.runner import auto_fp16, load_checkpoint

from ..base import BaseModel
from ..builder import build_backbone, build_loss, build_components, build_model
from ..registry import MODELS
from vcl.utils.helpers import *
from vcl.utils import *


import torch.nn as nn
import torch
import torch.nn.functional as F
import math


@MODELS.register_module()
class Pixel_Tracker(BaseModel):

    def __init__(self,
                 backbone,
                 temperature,
                 cts_loss=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None
                 ):
        """ Space-Time Memory Network for Video Object Segmentation

        Args:
            depth ([type]): ResNet depth for encoder
            pixel_loss ([type]): loss option
            train_cfg ([type], optional): [description]. Defaults to None.
            test_cfg ([type], optional): [description]. Defaults to None.
            pretrained ([type], optional): [description]. Defaults to None.
        """
        super().__init__()


    def forward_train(self, imgs):

        bsz, num_clips, t, c, h, w = imgs.shape
        tar = self.backbone(imgs[:,0,-1])
        refs = list([self.backbone(imgs[:,0,i]) for i in range(t-1)])

        for i in range(t-1):
            out = self.mining_process(tar, refs[i], bsz)
        

    
    def mining_process(self, tar, ref):

        aff = torch.einsum('bci,bcj->bij',[tar, ref])
        Q = (aff * aff) / (aff.max(dim=-2) * aff.max(dim=-1))
        
        Q_opt = self.shinck_opt(Q)
        C = Q_opt * self.mask

        return C

    
    def shinck_opt(self, Q):
        return Q
