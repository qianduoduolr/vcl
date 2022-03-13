# Copyright (c) OpenMMLab. All rights reserved.
import numbers
from os import stat_result
import os.path as osp
from collections import *
from tkinter.messagebox import NO

import mmcv
from mmcv.runner import auto_fp16, load_state_dict, load_checkpoint

from ..base import BaseModel
from ..builder import build_backbone, build_loss
from ..registry import MODELS
from vcl.utils.helpers import *
from vcl.utils import *
from vcl.models.common import *


import torch.nn as nn
import torch
import torch.nn.functional as F
import math


@MODELS.register_module()
class Warp_Tracker(BaseModel):

    def __init__(self,
                 backbone,
                 l1_loss=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 ):
        """ Distilation tracker

        Args:
            depth ([type]): ResNet depth for encoder
            pixel_loss ([type]): loss option
            train_cfg ([type], optional): [description]. Defaults to None.
            test_cfg ([type], optional): [description]. Defaults to None.
            pretrained ([type], optional): [description]. Defaults to None.
        """

        super(Warp_Tracker, self).__init__()

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.l1_loss  = l1_loss

        self.pretrained = pretrained

        logger = get_root_logger()

        self.backbone = build_backbone(backbone)

        
        self.init_weights()
    
    def init_weights(self):
        
        self.backbone.init_weights()        
        if self.pretrained is not None:
            _ = load_checkpoint(self, self.pretrained, map_location='cpu')


    def forward_train(self, imgs, imgs_reg, affine_imgs, affine_imgs_reg):

       pass