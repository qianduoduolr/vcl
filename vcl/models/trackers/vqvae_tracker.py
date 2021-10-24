# Copyright (c) OpenMMLab. All rights reserved.
import numbers
import os.path as osp
from collections import *

import mmcv
from mmcv.runner import auto_fp16
from spatial_correlation_sampler import SpatialCorrelationSampler

from ..base import BaseModel
from ..builder import build_backbone, build_loss, build_components
from ..registry import MODELS
from vcl.utils.helpers import *


import torch.nn as nn
import torch
import torch.nn.functional as F
import math


@MODELS.register_module()
class Vqvae_Tracker(BaseModel):

    def __init__(self,
                 backbone,
                 vqvae,
                 ce_loss,
                 patch_size,
                 fc=True,
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
        super(Vqvae_Tracker, self).__init__()

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.patch_size = patch_size

        self.backbone = build_backbone(backbone)
        self.vqvae = build_components(vqvae).cuda()

        # loss
        self.ce_loss = build_loss(ce_loss)

        # corr
        self.correlation_sampler = SpatialCorrelationSampler(
            kernel_size=1,
            patch_size=patch_size,
            stride=1,
            padding=0,
            dilation=1)

        # fc
        self.fc = fc
        if self.fc:
            self.predictor = nn.Linear(self.backbone.feat_dim, vqvae.n_embed)
        else:
            self.vq_emb = self.vqvae.quantize.embed
            self.embedding_layer = nn.Linear(self.backbone.feat_dim, self.vq_emb.shape[0])

        if pretrained is not None:
            self.init_weights(pretrained)

    def init_weights(self, pretrained):
        ckpt = torch.load(pretrained)
        self.vqvae.load_state_dict(ckpt)
        print('load pretrained VQVAE Successfully!')

    def forward_train(self, imgs, mask_query_idx):
        bsz, num_clips, t, c, w, h = imgs.shape
        mask_query_idx = mask_query_idx.bool()

        fs = self.backbone(imgs.reshape(bsz * t, c, h, w))
        fs = fs.reshape(bsz, t, self.backbone.feat_dim, fs.shape[-1], fs.shape[-1])

        # vqvae tokenize for query frame
        with torch.no_grad():
            _, quant, diff, ind, embed = self.vqvae.encode(imgs[:, 0, -1])
            ind = ind.reshape(-1, 1).long()

        refs = list([ fs[:,idx] for idx in range(t-1)])
        tar = fs[:, -1]
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

        if self.fc:
            predict = self.predictor(out)
        else:
            predict = self.embedding_layer(out)
            predict = torch.mm(predict, self.vq_emb)

        losses = {}
        losses['ce_loss'] = (self.ce_loss(predict, ind) * mask_query_idx.reshape(-1)).sum() / mask_query_idx.sum()
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
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        log_vars.pop('loss')
        outputs = dict(
            log_vars=log_vars,
            num_samples=len(data_batch['imgs'])
        )

        return outputs