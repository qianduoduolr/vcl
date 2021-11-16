# Copyright (c) OpenMMLab. All rights reserved.
import numbers
import os.path as osp
from collections import *

import mmcv
from mmcv.runner import auto_fp16, load_checkpoint

from ..base import BaseModel
from ..builder import build_backbone, build_loss
from ..registry import MODELS
from vcl.utils.helpers import *
from vcl.utils import *


import torch.nn as nn
import torch
import torch.nn.functional as F


@MODELS.register_module()
class Dist_Tracker(BaseModel):

    def __init__(self,
                 backbone,
                 patch_size,
                 moment,
                 temperature,
                 ce_loss=None,
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

        super(Dist_Tracker, self).__init__()

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.patch_size = patch_size
        self.moment = moment

        logger = get_root_logger()

        self.backbone_s = build_backbone(backbone)
        self.backbone_t = build_backbone(backbone)

        # loss
        self.ce_loss = build_loss(ce_loss) if ce_loss else None

        # corr
        if patch_size != -1:
            from spatial_correlation_sampler import SpatialCorrelationSampler
            self.correlation_sampler = SpatialCorrelationSampler(
                kernel_size=1,
                patch_size=patch_size,
                stride=1,
                padding=0,
                dilation=1)

    def forward_train(self, imgs, mask_query_idx, jitter_imgs=None):

        bsz, num_clips, t, c, h, w = imgs.shape
        mask_query_idx = mask_query_idx.bool()

        tar = self.backbone_s(imgs[:,0,-1])
        refs = list([self.backbone_s(imgs[:,0,i]) for i in range(t-1)])
        _, predict_att = self.non_local_attention(tar, refs, bsz)

        with torch.no_grad():
            tar_t = self.backbone_t(imgs[:,0,-1])
            refs_t = list([self.backbone_t(imgs[:,0,i]) for i in range(t-1)])
            if self.patch_size != -1:
                _, target_att = self.local_attention(tar_t, refs_t, bsz, t)
            else:
                _, target_att = self.non_local_attention(tar_t, refs_t, bsz)
        
        target = target_att.reshape(-1, target_att.shape[-1]).argmax(dim=-1, keepdim=True).long().detach()
        predict_att = predict_att.reshape(-1, predict_att.shape[-1])

        losses = {}
        losses['att_loss'] = (self.ce_loss(predict_att, target) * mask_query_idx.reshape(-1)).sum() / mask_query_idx.sum()

        return losses


    def forward(self, test_mode=False, **kwargs):   
        if test_mode:
            return self.forward_test(**kwargs)

        return self.forward_train(**kwargs)

    def train_step(self, data_batch, optimizer):

        # parser loss
        losses = self(**data_batch, test_mode=False)
        loss, log_vars = self.parse_losses(losses)

        # optimizer
        for k,opz in optimizer.items():
            opz.zero_grad()

        loss.backward()
        for k,opz in optimizer.items():
            opz.step()

        moment_update(self.backbone_s, self.backbone_t, self.moment)

        log_vars.pop('loss')
        outputs = dict(
            log_vars=log_vars,
            num_samples=len(data_batch['imgs'])
        )

        return outputs



    def local_attention(self, tar, refs, bsz, t):

        _, feat_dim, w_, h_ = tar.shape
        corrs = []
        for i in range(t-1):
            corr = self.correlation_sampler(tar.contiguous(), refs[i].contiguous()).reshape(bsz, -1, w_, h_)
            corrs.append(corr)

        corrs = torch.cat(corrs, 1)
        att = F.softmax(corrs, 1).unsqueeze(1)

        unfold_fs = list([ F.unfold(ref, kernel_size=self.patch_size, \
            padding=int((self.patch_size-1)/2)).reshape(bsz, feat_dim, -1, w_, h_) for ref in refs])
        unfold_fs = torch.cat(unfold_fs, 2)

        out = (unfold_fs * att).sum(2).reshape(bsz, feat_dim, -1).permute(0,2,1).reshape(-1, feat_dim)

        return out, att
    
    def non_local_attention(self, tar, refs, bsz):

        refs = torch.stack(refs, 2)
        _, feat_dim, w_, h_ = tar.shape

        refs = refs.reshape(bsz, feat_dim, -1).permute(0, 2, 1)
        tar = tar.reshape(bsz, feat_dim, -1).permute(0, 2, 1)

        att = torch.einsum("bic,bjc -> bij", (tar, refs))
        att = F.softmax(att, dim=-1)

        out = torch.matmul(att, refs).reshape(-1, feat_dim)

        return out, att