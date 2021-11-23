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
import math


@MODELS.register_module()
class Dist_Tracker(BaseModel):

    def __init__(self,
                 backbone,
                 patch_size,
                 moment,
                 temperature,
                 dilated_search=False,
                 loss=None,
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
        self.loss_config = loss
        self.patch_size = patch_size
        self.moment = moment
        self.dilated_search = dilated_search

        logger = get_root_logger()

        self.backbone = build_backbone(backbone)
        self.backbone_t = build_backbone(backbone)

        # loss
        self.loss = build_loss(loss)

    def forward_train(self, imgs, mask_query_idx, jitter_imgs=None, progress_ratio=0.0):

        bsz, num_clips, t, c, h, w = imgs.shape
        mask_query_idx = mask_query_idx.bool()

        tar = self.backbone(jitter_imgs[:,0,-1])
        refs = list([self.backbone(jitter_imgs[:,0,i]) for i in range(t-1)])
        _, predict_att = self.non_local_attention(tar, refs, bsz, return_att=True)

        h_,w_ = tar.shape[-2:]
        if self.patch_size != -1:
            if self.dilated_search:
                patch_size = self.patch_size + int((h_ // 2 - self.patch_size - 1) * progress_ratio)
            else:
                patch_size = self.patch_size
            t_masks = self.make_mask(h_, patch_size)

        with torch.no_grad():
            tar_t = self.backbone_t(imgs[:,0,-1])
            refs_t = list([self.backbone_t(imgs[:,0,i]) for i in range(t-1)])

        _, target_att = self.non_local_attention(tar_t, refs_t, bsz, return_att=True)
        if self.patch_size != -1:
            target_att = target_att * t_masks.unsqueeze(0)
        else:
            pass

        target = target_att.reshape(-1, target_att.shape[-1]).detach()
        predict_att = predict_att.reshape(-1, predict_att.shape[-1])

        if self.loss_config.type == 'MSELoss':
            target = target.softmax(-1)
            predict_att = predict_att.softmax(-1)

        losses = {}
        losses['att_loss'] = self.loss(predict_att, target, weight=mask_query_idx.reshape(-1, 1))

        return losses


    def forward(self, test_mode=False, **kwargs):   
        if test_mode:
            return self.forward_test(**kwargs)

        return self.forward_train(**kwargs)

    def train_step(self, data_batch, optimizer, progress_ratio):

        data_batch = {**data_batch, 'progress_ratio':progress_ratio}

        # parser loss
        losses = self(**data_batch, test_mode=False)
        loss, log_vars = self.parse_losses(losses)

        # optimizer
        for k,opz in optimizer.items():
            opz.zero_grad()

        loss.backward()
        for k,opz in optimizer.items():
            opz.step()

        moment_update(self.backbone, self.backbone_t, self.moment)

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
    
    def non_local_attention(self, tar, refs, bsz, return_att=False):

        refs = torch.stack(refs, 2)
        _, feat_dim, w_, h_ = tar.shape

        refs = refs.reshape(bsz, feat_dim, -1).permute(0, 2, 1)
        tar = tar.reshape(bsz, feat_dim, -1).permute(0, 2, 1)

        att = torch.einsum("bic,bjc -> bij", (tar, refs))
        
        if return_att:
            return None, att

        att = F.softmax(att, dim=-1)

        out = torch.matmul(att, refs).reshape(-1, feat_dim)

        return out, att
    
    def make_mask(self, size, t_size):
        masks = []
        for i in range(size):
            for j in range(size):
                mask = torch.zeros((size, size)).cuda()
                mask[max(0, i-t_size):min(size, i+t_size+1), max(0, j-t_size):min(size, j+t_size+1)] = 1
                masks.append(mask.reshape(-1))
        return torch.stack(masks)