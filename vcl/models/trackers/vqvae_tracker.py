# Copyright (c) OpenMMLab. All rights reserved.
import numbers
import os.path as osp
from collections import *

import mmcv
from mmcv.runner import auto_fp16, load_checkpoint
from spatial_correlation_sampler import SpatialCorrelationSampler
from dall_e  import map_pixels, unmap_pixels, load_model

from ..base import BaseModel
from ..builder import build_backbone, build_loss, build_components
from ..registry import MODELS
from vcl.utils.helpers import *
from vcl.utils import *


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
                 pretrained_vq,
                 temperature,
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

        logger = get_root_logger()

        self.backbone = build_backbone(backbone)

        self.vq_type = vqvae.type
        if vqvae.type is not 'DALLE_Encoder':
            self.vqvae = build_components(vqvae).cuda()
            _ = load_checkpoint(self.vqvae, pretrained_vq, map_location='cpu')
            logger.info('load pretrained VQVAE successfully')
            self.vq_emb = self.vqvae.quantize.embed
            self.n_embed = vqvae.n_embed
            self.vq_t = temperature
            self.vq_enc = self.vqvae.encode
        else:
            self.vq_enc = load_model(pretrained_vq).cuda()
            self.n_embed = self.vq_enc.vocab_size
            logger.info('load pretrained VQVAE successfully')

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
            self.predictor = nn.Linear(self.backbone.feat_dim, self.n_embed)
        else:
            self.embedding_layer = nn.Linear(self.backbone.feat_dim, self.vq_emb.shape[0])

        if pretrained:
            _ = load_checkpoint(self, pretrained, map_location='cpu')
            logger.info('load pretrained model successfully')

    def forward_train(self, imgs, mask_query_idx):
        bsz, num_clips, t, c, w, h = imgs.shape
        mask_query_idx = mask_query_idx.bool()

        fs = self.backbone(imgs.reshape(bsz * t, c, h, w))
        fs = fs.reshape(bsz, t, -1, fs.shape[-1], fs.shape[-1])

        # vqvae tokenize for query frame
        with torch.no_grad():
            if self.vq_type == 'VQVAE':
                self.vqvae.eval()
                _, quant, diff, ind, embed = self.vq_enc(imgs[:, 0, -1])
                ind = ind.reshape(-1, 1).long().detach()
            else:
                self.vq_enc.eval()
                ind = self.vq_enc(imgs[:, 0, -1])
                ind = torch.argmax(ind, axis=1).reshape(-1, 1).long().detach()

        if self.patch_size != -1:
            out = self.local_attention(fs, bsz, t)
        else:
            out = self.non_local_attention(fs, bsz)

        if self.fc:
            predict = self.predictor(out)
        else:
            predict = self.embedding_layer(out)
            predict = nn.functional.normalize(predict, dim=-1)
            predict = torch.mm(predict, nn.functional.normalize(self.vq_emb, dim=0))
            predict = torch.div(predict, self.vq_t)


        losses = {}
        losses['ce_loss'] = (self.ce_loss(predict, ind) * mask_query_idx.reshape(-1)).sum() / mask_query_idx.sum()
        return losses

    def forward_test(self, imgs, mask_query_idx,
                    save_image=False,
                    save_path=None,
                    iteration=None):
        bsz, num_clips, t, c, w, h = imgs.shape
        mask_query_idx = mask_query_idx.bool()

        fs = self.backbone(imgs.reshape(bsz * t, c, h, w))
        fs = fs.reshape(bsz, t, -1, fs.shape[-1], fs.shape[-1])

        # vqvae tokenize for query frame
        with torch.no_grad():
            if self.vq_type == 'VQVAE':
                _, quant, diff, ind, embed = self.vqvae.encode(imgs[:, 0, -1])
                ind = ind.reshape(-1, 1).long().detach()
            else:
                ind = self.vqvae(imgs[:, 0, -1])
                ind = torch.argmax(ind, axis=1).reshape(-1, 1).long().detach()

        refs = list([ fs[:,idx] for idx in range(t-1)])
        tar = fs[:, -1]
        _, feat_dim, w_, h_ = tar.shape
        corrs = []
        for i in range(t-1):
            corr = self.correlation_sampler(tar.contiguous(), refs[i].contiguous()).reshape(bsz, -1, w_, h_)
            corrs.append(corr)

        corrs = torch.cat(corrs, 1)
        att = F.softmax(corrs, 1).unsqueeze(1)

        visualize_att(imgs, att, random.randint(0,3400), w_, self.patch_size)

        unfold_fs = list([ F.unfold(ref, kernel_size=self.patch_size, \
            padding=int((self.patch_size-1)/2)).reshape(bsz, feat_dim, -1, w_, h_) for ref in refs])
        unfold_fs = torch.cat(unfold_fs, 2)

        out = (unfold_fs * att).sum(2).reshape(bsz, feat_dim, -1).permute(0,2,1).reshape(-1, feat_dim)

        if self.fc:
            predict = self.predictor(out)
        else:
            predict = self.embedding_layer(out)
            predict = torch.mm(predict, self.vq_emb)

        out2 = torch.argmax(predict.cpu().detach(), axis=1).numpy()
        out1 = ind.cpu().numpy()

        return out1, out2

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

        log_vars.pop('loss')
        outputs = dict(
            log_vars=log_vars,
            num_samples=len(data_batch['imgs'])
        )

        return outputs

    def local_attention(self, fs, bsz, t):

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

        return out
    
    def non_local_attention(self, fs, bsz):

        refs = fs[:, :-1].permute(0, 2, 1, 3, 4)
        tar = fs[:, -1]
        _, feat_dim, w_, h_ = tar.shape

        refs = refs.reshape(bsz, feat_dim, -1).permute(0, 2, 1)
        tar = tar.reshape(bsz, feat_dim, -1).permute(0, 2, 1)

        att = torch.einsum("bic,bjc -> bij", (tar, refs))
        att = F.softmax(att, dim=-1)

        out = torch.matmul(att, refs).reshape(-1, feat_dim)

        return out


