# Copyright (c) OpenMMLab. All rights reserved.
import numbers
import os.path as osp
from collections import *

import mmcv
from mmcv.runner import auto_fp16, load_checkpoint
from dall_e  import map_pixels, unmap_pixels, load_model

from vcl.models.losses.losses import Ce_Loss

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
class Vqvae_Tracker(BaseModel):

    def __init__(self,
                 backbone,
                 vqvae,
                 patch_size,
                 pretrained_vq,
                 temperature,
                 sim_siam_head=None,
                 multi_head_weight=[1.0],
                 ce_loss=None,
                 mse_loss=None,
                 cts_loss=None,
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
        if not isinstance(vqvae, list):
            vqvae = list([vqvae])
            pretrained_vq = list([pretrained_vq])

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.patch_size = patch_size
        self.num_head = len(vqvae)
        self.multi_head_weight = multi_head_weight

        logger = get_root_logger()

        self.backbone = build_backbone(backbone)
        if sim_siam_head is not None:
            self.head = build_components(sim_siam_head)
            self.head.init_weights()
        else:
            self.head = None

        self.vq_type = vqvae[0].type
        if self.vq_type != 'DALLE_Encoder':
            for idx, i in enumerate(range(len(vqvae))):
                i = str(i).replace('0', '')
                setattr(self, f'vqvae{i}', build_model(vqvae[idx]).cuda())
                _ = load_checkpoint(getattr(self, f'vqvae{i}'), pretrained_vq[idx], map_location='cpu')
                logger.info(f'load {i}th pretrained VQVAE successfully')
                setattr(self, f'vq_emb{i}', getattr(self, f'vqvae{i}').quantize.embed)
                setattr(self, f'n_embed{i}', vqvae[idx].n_embed)
                setattr(self, f'vq_t{i}', temperature)
                setattr(self, f'vq_enc{i}', getattr(self, f'vqvae{i}').encode)
        else:
            assert self.num_head == 1
            self.vq_enc = load_model(pretrained_vq).cuda()
            self.n_embed = self.vq_enc.vocab_size
            logger.info('load pretrained VQVAE successfully')

        # loss
        self.ce_loss = build_loss(ce_loss) if ce_loss else None
        self.mse_loss = build_loss(mse_loss) if mse_loss else None
        self.cts_loss = build_loss(cts_loss) if cts_loss else None

        # corr
        if patch_size != -1:
            from spatial_correlation_sampler import SpatialCorrelationSampler
            self.correlation_sampler = SpatialCorrelationSampler(
                kernel_size=1,
                patch_size=patch_size,
                stride=1,
                padding=0,
                dilation=1)

        # fc
        self.fc = fc
        if self.fc:
            for i in range(len(vqvae)):
                i = str(i).replace('0', '')
                setattr(self, f'predictor{i}', nn.Linear(self.backbone.feat_dim, getattr(self, f'n_embed{i}')))
        else:
            self.embedding_layer = nn.Linear(self.backbone.feat_dim, self.vq_emb.shape[0])

        if pretrained:
            _ = load_checkpoint(self, pretrained, map_location='cpu')
            logger.info('load pretrained model successfully')

    def forward_train(self, imgs, mask_query_idx, jitter_imgs=None):

        bsz, num_clips, t, c, h, w = imgs.shape

        # vqvae tokenize for query frame
        with torch.no_grad():
            out_ind= []
            for i in range(self.num_head):
                i = str(i).replace('0', '')
                vqvae = getattr(self, f'vqvae{i}')
                vq_enc = getattr(self, f'vq_enc{i}')
                vqvae.eval()
                _, _, _, ind, _ = vq_enc(imgs[:, 0, -1])
                ind = ind.unsqueeze(1).repeat(1, t-1, 1, 1).reshape(-1, 1).long().detach()
                out_ind.append(ind)

        if jitter_imgs is not None:
            imgs = jitter_imgs

        mask_query_idx = mask_query_idx.bool().unsqueeze(1).repeat(1,t-1,1)

        tar = self.backbone(imgs[:,0,-1])
        refs = list([self.backbone(imgs[:,0,i]) for i in range(t-1)])

        if self.patch_size != -1:
            out, att = self.local_attention(tar, refs, bsz, t)
        else:
            out, att = self.non_local_attention_split(tar, refs, bsz)

        losses = {}

        if self.ce_loss:
            for idx, i in enumerate(range(self.num_head)):
                i = str(i).replace('0', '')
                predict = getattr(self, f'predictor{i}')(out.flatten(0,2))
                loss = self.ce_loss(predict, out_ind[idx])
                losses[f'ce{i}_loss'] = (loss * mask_query_idx.reshape(-1)).sum() / mask_query_idx.sum() * self.multi_head_weight[idx]

        if self.mse_loss and t > 2:
            out_roll = torch.roll(out, 1, dims=1)
            losses['mse_loss'] = self.mse_loss(out.flatten(0,2), out_roll.flatten(0,2), mask_query_idx.reshape(-1,1))

        if self.cts_loss:
            losses['cts_loss'] = self.forward_img_head(tar, refs[-1])

        return losses

    def forward_test(self, imgs, mask_query_idx,
                    save_image=False,
                    save_path=None,
                    iteration=None):
        bsz, num_clips, t, c, w, h = imgs.shape

        tar = self.backbone(imgs[:,0,-1])
        refs = list([self.backbone(imgs[:,0,i]) for i in range(t-1)])

        # vqvae tokenize for query frame
        with torch.no_grad():
            if self.vq_type == 'VQVAE':
                _, quant, diff, ind, embed = self.vq_enc(imgs[:, 0, -1])
                ind = ind.reshape(-1, 1).long().detach()
            else:
                ind = self.vqvae(imgs[:, 0, -1])
                ind = torch.argmax(ind, axis=1).reshape(-1, 1).long().detach()


        out, att = self.non_local_attention(tar, refs, bsz)


        visualize_att(imgs, att, iteration, mask_query_idx, tar.shape[-1], self.patch_size, dst_path=save_path, norm_mode='mean-std')

        if self.backbone.out_indices[0] == 3:
            if self.fc:
                predict = self.predictor(out)
            else:
                if out.shape[-1] != self.vq_emb.shape[0]:
                    predict = self.embedding_layer(out)
                else:
                    predict = out
                predict = nn.functional.normalize(predict, dim=-1)
                predict = torch.mm(predict, nn.functional.normalize(self.vq_emb, dim=0))
                predict = torch.div(predict, self.vq_t)

            out2 = torch.argmax(predict.cpu().detach(), axis=1).numpy()
            out1 = ind.cpu().numpy()

            return out1, out2
        else:
            return None

    def forward(self, test_mode=False, **kwargs):

        if test_mode:
            return self.forward_test(**kwargs)

        return self.forward_train(**kwargs)

    def train_step(self, data_batch, optimizer, progress_ratio):

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
    
    def non_local_attention_split(self, tar, refs, bsz):

        refs = torch.stack(refs, 1)
        _, t, feat_dim, w_, h_ = refs.shape

        # refs = refs.reshape(bsz, feat_dim, -1).permute(0, 2, 1)
        # tar = tar.reshape(bsz, feat_dim, -1).permute(0, 2, 1)
        tar = tar.flatten(2).permute(0, 2, 1)
        refs = refs.flatten(3).permute(0, 1, 3, 2)
        att = torch.einsum("bic,btjc -> btij", (tar, refs))
        att = F.softmax(att, dim=-1)

        out = torch.matmul(att, refs)

        return out, att
    
    def non_local_attention(self, tar, refs, bsz):

        refs = torch.stack(refs, 2)
        _, feat_dim, t, _, _ = refs.shape

        refs = refs.reshape(bsz, feat_dim, -1).permute(0, 2, 1)
        tar = tar.reshape(bsz, feat_dim, -1).permute(0, 2, 1)

        att = torch.einsum("bic,bjc -> bij", (tar, refs))
        att = F.softmax(att, dim=-1)

        out = torch.matmul(att, refs).reshape(-1, feat_dim)

        return out, att


    def forward_img_head(self, x1, x2):

        if isinstance(x1, tuple):
            x1 = x1[-1]
        if isinstance(x2, tuple):
            x2 = x2[-1]

        z1, p1 = self.head(x1)
        z2, p2 = self.head(x2)
        
        loss = self.cts_loss(p1, z2.detach()) * 0.5 + self.cts_loss(p2, z1.detach()) * 0.5

        return loss


@MODELS.register_module()
class Vqvae_Tracker_v2(BaseModel):

    def __init__(self,
                 backbone,
                 vqvae,
                 patch_size,
                 pretrained_vq,
                 temperature,
                 ce_loss=None,
                 l2_loss=None,
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
        super(Vqvae_Tracker_v2, self).__init__()

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.patch_size = patch_size

        logger = get_root_logger()

        self.backbone = build_backbone(backbone)
        if backbone.type == 'Vq_Res':
            self.embed_dim = self.backbone.res_blocks.feat_dim
        else:
            self.embed_dim = self.backbone.transformer_blocks.embed_dim

        self.vq_type = vqvae.type
        if vqvae.type != 'DALLE_Encoder':
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
        self.ce_loss = build_loss(ce_loss) if ce_loss else None
        self.l2_loss = build_loss(l2_loss) if l2_loss else None

        # fc
        self.fc = fc

        if self.fc:
            self.predictor = nn.Linear(self.embed_dim, self.n_embed)
        else:
            self.embedding_layer = nn.Linear(self.embed_dim, self.vq_emb.shape[0])

        if pretrained:
            _ = load_checkpoint(self, pretrained, map_location='cpu')
            logger.info('load pretrained model successfully')

    def forward_train(self, imgs, mask_query_idx, jitter_imgs=None):

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

        if jitter_imgs is not None:
            imgs = jitter_imgs

        bsz, num_clips, t, c, h, w = imgs.shape
        mask_query_idx = mask_query_idx.bool()


        tar = self.backbone(imgs[:, 0, -1])
        refs = list([self.backbone(imgs[:, 0, i]) for i in range(t-1)])

        out, _ = self.non_local_attention(tar, refs, bsz)

        losses = {}

        if self.ce_loss:
            if self.fc:
                predict = self.predictor(out)
            else:
                if out.shape[-1] != self.vq_emb.shape[0]:
                    predict = self.embedding_layer(out)
                else:
                    predict = out
                predict = nn.functional.normalize(predict, dim=-1)
                predict = torch.mm(predict, nn.functional.normalize(self.vq_emb, dim=0))
                predict = torch.div(predict, self.vq_t)

            loss = self.ce_loss(predict, ind)
            losses['ce_loss'] = ( loss * mask_query_idx.reshape(-1)).sum() / mask_query_idx.sum()

        if self.l2_loss:
            predict = self.embedding_layer(out)
            predict = nn.functional.normalize(predict, dim=-1)
            embeds = torch.index_select(self.vq_emb, dim=1, index=ind.view(-1)).permute(1,0)
            embeds = nn.functional.normalize(embeds, dim=-1)
            losses['l2_loss'] = (self.l2_loss(predict, embeds) * mask_query_idx.reshape(-1,1)).sum() / mask_query_idx.sum()

        return losses

    def forward_test(self, imgs, mask_query_idx,
                    save_image=False,
                    save_path=None,
                    iteration=None):
        bsz, num_clips, t, c, w, h = imgs.shape

        tar = self.backbone(imgs[:,0,-1])
        refs = list([self.backbone(imgs[:,0,i]) for i in range(t-1)])

        # vqvae tokenize for query frame
        with torch.no_grad():
            if self.vq_type == 'VQVAE':
                _, quant, diff, ind, embed = self.vqvae.encode(imgs[:, 0, -1])
                ind = ind.reshape(-1, 1).long().detach()
            else:
                ind = self.vqvae(imgs[:, 0, -1])
                ind = torch.argmax(ind, axis=1).reshape(-1, 1).long().detach()


        out, att = self.non_local_attention(tar, refs, bsz)

        visualize_att(imgs, att, iteration, mask_query_idx, tar.shape[-1], self.patch_size, dst_path=save_path, norm_mode='mean-std')

        if self.backbone.out_indices[0] == 1:
            if self.fc:
                predict = self.predictor(out)
            else:
                if out.shape[-1] != self.vq_emb.shape[0]:
                    predict = self.embedding_layer(out)
                else:
                    predict = out
                predict = nn.functional.normalize(predict, dim=-1)
                predict = torch.mm(predict, nn.functional.normalize(self.vq_emb, dim=0))
                predict = torch.div(predict, self.vq_t)

            out2 = torch.argmax(predict.cpu().detach(), axis=1).numpy()
            out1 = ind.cpu().numpy()

            return out1, out2
        else:
            return None

    def forward(self, test_mode=False, **kwargs):

        if test_mode:
            return self.forward_test(**kwargs)

        return self.forward_train(**kwargs)

    def train_step(self, data_batch, optimizer, progress_ratio):

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

    
    def non_local_attention(self, tar, refs, bsz):

        refs = torch.stack(refs, 2)
        _, feat_dim, t, _, _ = refs.shape

        refs = refs.reshape(bsz, feat_dim, -1).permute(0, 2, 1)
        tar = tar.reshape(bsz, feat_dim, -1).permute(0, 2, 1)

        att = torch.einsum("bic,bjc -> bij", (tar, refs))
        att = F.softmax(att, dim=-1)

        out = torch.matmul(att, refs).reshape(-1, feat_dim)

        return out, att
