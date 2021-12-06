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
from vcl.models.common.sinkhorn_layers import SinkhornDistance


import torch.nn as nn
import torch
import torch.nn.functional as F


@MODELS.register_module()
class Pixel_Tracker(BaseModel):

    def __init__(self,
                 backbone,
                 temperature,
                 thres=0.9,
                 radius=[2,2,3,5,5],
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
        self.backbone = build_backbone(backbone)
        self.sinkhorn = SinkhornDistance(eps=0.01, max_iter=30, reduction=None)
        self.masks = self.make_mask(32, radius).cuda()
        self.THRES = thres


    def forward_train(self, imgs):

        bsz, num_clips, t, c, h, w = imgs.shape
        tar = F.normalize(self.backbone(imgs[:,0,0]), dim=1)
        refs = F.normalize(self.backbone(imgs[:,0,1:].flatten(0,1)), dim=1)

        tar = tar.flatten(2)
        refs = refs.reshape(bsz, t-1, *tar.shape[1:]).flatten(3)
        # check cycle consistency and optimal transport
        Ts, affs = self.mining_process(tar, refs)
        # temporal window
        C = Ts * self.masks.unsqueeze(0)
        # C = Ts * 1

        losses= {}
        sim = 1 - affs
        losses['rec_loss'] = (sim * C).sum() / (C.sum() + 1e-3)
        # print(losses['rec_loss'].item(), C.max())

        return losses
    
    def mining_process(self, tar, refs):

        affs = torch.einsum('bci,btcj->btij',[tar, refs]).flatten(0,1)

        # a = affs.detach().cpu()[0].numpy()

        aff_i = torch.max(affs, dim=-1, keepdim=True)[0]
        aff_j = torch.max(affs, dim=-2, keepdim=True)[0]
        Q = (affs * affs) / (torch.matmul(aff_i, aff_j))
        
        # apply optimal transport
        T = self.sink_opt(tar, refs, Q=Q)
        T = T.reshape(*refs.shape[:2], T.shape[-1], -1)
        affs = affs.reshape(*refs.shape[:2], T.shape[-1], -1)

        return T, affs

    
    def sink_opt(self, tar, refs, Q):

        Q = (1 - Q).detach()
        T = self.sinkhorn(refs, C=Q)
        T = T * tar.shape[-1]
        T = (T>=self.THRES).float()

        return T

    def make_mask(self, size, radius):
        masks = []
        for r in radius:
            mask = []
            for i in range(size):
                for j in range(size):
                    m = torch.zeros((size, size)).cuda()
                    m[max(0, i-r):min(size, i+r+1), max(0, j-r):min(size, j+r+1)] = 1
                    mask.append(m.reshape(-1))
            masks.append(torch.stack(mask))
        return torch.stack(masks)



    def train_step(self, data_batch, optimizer, progress_ratio):

        # parser loss
        losses = self(**data_batch, test_mode=False)
        loss, log_vars = self.parse_losses(losses)

        # optimizer

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        log_vars.pop('loss')
        outputs = dict(
            log_vars=log_vars,
            num_samples=len(data_batch['imgs'])
        )

        return outputs
