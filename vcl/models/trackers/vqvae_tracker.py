# Copyright (c) OpenMMLab. All rights reserved.
import numbers
import os.path as osp
from collections import *

import mmcv
from mmcv.runner import auto_fp16, load_checkpoint
from dall_e  import map_pixels, unmap_pixels, load_model
from vcl.models.common.correlation import frame_transform

from vcl.models.losses.losses import Ce_Loss
from vcl.models.common.correlation import *

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
                 fc=True,
                 train_cfg=None,
                 test_cfg=None,
                 per_ref=True,
                 pretrained=None
                 ):
        """ original vqvae tracker
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
        self.per_ref = per_ref

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
            out_quant = []
            for i in range(self.num_head):
                i = str(i).replace('0', '')
                vqvae = getattr(self, f'vqvae{i}')
                vq_enc = getattr(self, f'vq_enc{i}')
                vqvae.eval()
                emb, quant, _, ind, _ = vq_enc(imgs[:, 0, -1])
                
                if self.per_ref:
                    ind = ind.unsqueeze(1).repeat(1, t-1, 1, 1).reshape(-1, 1).long().detach()
                    quant = quant.unsqueeze(1).repeat(1, t-1, 1, 1, 1).permute(0,1,3,4,2).flatten(0,3).detach()
                    mask_query_idx = mask_query_idx.bool().unsqueeze(1).repeat(1,t-1,1)
                else:
                    ind = ind.reshape(-1, 1).long().detach()
                    quant = quant.permute(0,2,3,1).flatten(0,2).detach()
                    mask_query_idx = mask_query_idx.bool()
                    
                out_ind.append(ind)
                out_quant.append(quant)

        if jitter_imgs is not None:
            imgs = jitter_imgs

        tar = self.backbone(imgs[:,0,-1])
        refs = list([self.backbone(imgs[:,0,i]) for i in range(t-1)])

        if self.patch_size != -1:
            out, att = local_attention(self.correlation_sampler, tar, refs, self.patch_size)
        else:
            out, att = non_local_attention(tar, refs, per_ref=self.per_ref)

        losses = {}

        if self.ce_loss:
            for idx, i in enumerate(range(self.num_head)):
                i = str(i).replace('0', '')
                if self.fc:
                    predict = getattr(self, f'predictor{i}')(out)
                else:
                    predict = self.embedding_layer(out)
                    predict = nn.functional.normalize(predict, dim=-1)
                    vq_emb = getattr(self, f'vq_emb{i}')
                    predict = torch.mm(predict, nn.functional.normalize(vq_emb, dim=0))
                    predict = torch.div(predict, 0.1) # temperature is set to 0.1
                    
                loss = self.ce_loss(predict, out_ind[idx])
                losses[f'ce{i}_loss'] = (loss * mask_query_idx.reshape(-1)).sum() / mask_query_idx.sum() * self.multi_head_weight[idx]
                
        if self.mse_loss:
            for idx, i in enumerate(range(self.num_head)):
                i = str(i).replace('0', '')
                predict = self.embedding_layer(out)
                loss = self.mse_loss(predict, out_quant[idx]).mean(-1)
                losses[f'mse{i}_loss'] = (loss * mask_query_idx.reshape(-1)).sum() / mask_query_idx.sum() * self.multi_head_weight[idx]
        
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

        out, att = non_local_attention(tar, refs)

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

@MODELS.register_module()
class Vqvae_Tracker_V2(BaseModel):

    def __init__(self,
                 backbone,
                 vqvae,
                 patch_size,
                 pretrained_vq,
                 temperature,
                 ce_loss=None,
                 fc=True,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None
                 ):
        """ use mask annotations as vq ce_loss label
        """
        super(Vqvae_Tracker_V2, self).__init__()
        if not isinstance(vqvae, list):
            vqvae = list([vqvae])
            pretrained_vq = list([pretrained_vq])

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.patch_size = patch_size
        self.num_head = len(vqvae)

        logger = get_root_logger()

        self.backbone = build_backbone(backbone)

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

        # fc
        self.fc = fc
        if self.fc:
            for i in range(len(vqvae)):
                i = str(i).replace('0', '')
                setattr(self, f'predictor{i}', nn.Linear(self.backbone.feat_dim, 16))
        else:
            raise NotImplementedError

        if pretrained:
            _ = load_checkpoint(self, pretrained, map_location='cpu')
            logger.info('load pretrained model successfully')

    def forward_train(self, imgs, mask_query_idx, masks):

        bsz, num_clips, t, c, h, w = imgs.shape

        mask_query_idx = mask_query_idx.bool().unsqueeze(1).repeat(1,t-1,1)
        out_ind = [ masks[:,-1:].repeat(1, t-1, 1, 1).reshape(-1, 1).long().detach() ]

        tar = self.backbone(imgs[:,0,-1])
        refs = list([self.backbone(imgs[:,0,i]) for i in range(t-1)])

        if self.patch_size != -1:
            out, att = local_attention(self.correlation_sampler, tar, refs, self.patch_size)
        else:
            out, att = non_local_attention(tar, refs, per_ref=True)

        losses = {}

        if self.ce_loss:
            for idx, i in enumerate(range(self.num_head)):
                i = str(i).replace('0', '')
                predict = getattr(self, f'predictor{i}')(out)
                loss = self.ce_loss(predict, out_ind[idx])
                losses[f'ce{i}_loss'] = (loss * mask_query_idx.reshape(-1)).sum() / mask_query_idx.sum()

        return losses


@MODELS.register_module()
class Vqvae_Tracker_V3(BaseModel):

    def __init__(self,
                 backbone,
                 vqvae,
                 patch_size,
                 pretrained_vq,
                 multi_head_weight=[1.0],
                 mse_loss=None,
                 l1_loss=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None
                 ):
        """ add mse_loss 
        """
        super(Vqvae_Tracker_V3, self).__init__()

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.patch_size = patch_size
        self.num_head = len(vqvae)
        self.multi_head_weight = multi_head_weight

        logger = get_root_logger()

        self.backbone = build_backbone(backbone)

        self.vq_type = vqvae.type
        if self.vq_type != 'DALLE_Encoder':
            self.vqvae =  build_model(vqvae).cuda()
            _ = load_checkpoint(self.vqvae, pretrained_vq, map_location='cpu')
            logger.info(f'load pretrained VQVAE successfully')
            self.vq_enc = self.vqvae.encode
            self.vq_dec = self.vqvae.decode
        else:
            assert self.num_head == 1
            self.vq_enc = load_model(pretrained_vq).cuda()
            self.n_embed = self.vq_enc.vocab_size
            logger.info('load pretrained VQVAE successfully')

        # loss
        self.mse_loss = build_loss(mse_loss) if mse_loss else None
        self.l1_loss = build_loss(l1_loss) if mse_loss else None

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
        if pretrained:
            _ = load_checkpoint(self, pretrained, map_location='cpu')
            logger.info('load pretrained model successfully')

    def forward_train(self, imgs, jitter_imgs=None):

        bsz, num_clips, t, c, h, w = imgs.shape

        embeds = []
        with torch.no_grad():
            self.vqvae.eval()
            for i in range(t):
                embed, quant, _, ind, _ = self.vq_enc(imgs[:, 0, i])
                embeds.append(embed.flatten(1,2))

        embeds_refs = torch.stack(embeds[:-1], 1)
        embeds_refs_cross = embeds[0].flatten(0,1).unsqueeze(0).repeat(bsz, 1, 1)


        tar = self.backbone(jitter_imgs[:,0,-1])
        refs = list([self.backbone(jitter_imgs[:,0,i]) for i in range(t-1)])
        refs_cross_video = [refs[0] for i in range(bsz)] 

        out, att = non_local_attention(tar, refs, per_ref=True)
        out, att_cross_video_, att_cross_video = non_local_attention(tar, refs_cross_video, per_ref=False)

        losses = {}

        trans_embeds = torch.einsum('btij,btjc->btic', [att, embeds_refs])
        trans_embeds = trans_embeds.flatten(0,1).reshape(-1, *embed.shape[1:])
        trans_quants = self.vq_enc(trans_embeds, encoding=False)[1]
        decs = self.vq_dec(trans_quants)

        trans_embeds_cross = torch.einsum('bik,bkc->bic', [att_cross_video_, embeds_refs_cross])
        trans_embeds_cross = trans_embeds_cross.reshape(-1, *embed.shape[1:])
        trans_quants_cross = self.vq_enc(trans_embeds_cross, encoding=False)[1]
        decs_cross = self.vq_dec(trans_quants_cross)

        target = imgs[:,0,-1:].repeat(1,t-1,1,1,1).flatten(0,1)
        losses['mse_loss'] = self.mse_loss(decs, target)

        if self.l1_loss:
            label = torch.zeros(*att_cross_video.shape).cuda()
            m = torch.eye(bsz).cuda()
            m = (1 - m)[:,:,None,None].repeat(1,1,*att_cross_video.shape[-2:])
            losses['l1_loss'] = self.l1_loss(att_cross_video, label, weight=m)
        
            losses['cross_loss'] = self.mse_loss(decs_cross, decs)

        return losses


@MODELS.register_module()
class Vqvae_Tracker_V4(BaseModel):

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
                 fc=True,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None
                 ):
        """ combine mse_loss and vq ce_loss
        """
        super(Vqvae_Tracker_V4, self).__init__()
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
                setattr(self, f'vq_dec{i}', getattr(self, f'vqvae{i}').decode) if idx >= 1 else None

        else:
            assert self.num_head == 1
            self.vq_enc = load_model(pretrained_vq).cuda()
            self.n_embed = self.vq_enc.vocab_size
            logger.info('load pretrained VQVAE successfully')

        # loss
        self.ce_loss = build_loss(ce_loss) if ce_loss else None
        self.mse_loss = build_loss(mse_loss) if mse_loss else None

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
        for i in range(len(vqvae)):
            if i > 1: break
            i = str(i).replace('0', '')
            setattr(self, f'predictor{i}', nn.Linear(self.backbone.feat_dim, getattr(self, f'n_embed{i}')))
    

        if pretrained:
            _ = load_checkpoint(self, pretrained, map_location='cpu')
            logger.info('load pretrained model successfully')

    def forward_train(self, imgs, mask_query_idx, jitter_imgs=None):

        bsz, num_clips, t, c, h, w = imgs.shape
        
        embeds = []
        out_ind = []
        # vqvae tokenize for query frame
        with torch.no_grad():
            for idx, i in enumerate(range(self.num_head)):
                i = str(i).replace('0', '')
                vqvae = getattr(self, f'vqvae{i}')
                vq_enc = getattr(self, f'vq_enc{i}')
                vqvae.eval()
                if idx >= 1:
                    for i in range(t):
                        embed, quant, _, ind, _ = vq_enc(imgs[:, 0, i])
                        embeds.append(embed.flatten(1,2))
                else:
                    _, _, _, ind, _ = vq_enc(imgs[:, 0, -1])
                    ind = ind.unsqueeze(1).repeat(1, t-1, 1, 1).reshape(-1, 1).long().detach()
                out_ind.append(ind)
        embeds_refs = torch.stack(embeds[:-1], 1)


        mask_query_idx = mask_query_idx.bool().unsqueeze(1).repeat(1,t-1,1)

        tar = self.backbone(jitter_imgs[:,0,-1])
        refs = list([self.backbone(jitter_imgs[:,0,i]) for i in range(t-1)])

        if self.patch_size != -1:
            out, att = local_attention(self.correlation_sampler, tar, refs, self.patch_size)
        else:
            out, att = non_local_attention(tar, refs, per_ref=True)


        losses = {}

        if self.ce_loss:
            predict = self.predictor(out)
            loss = self.ce_loss(predict, out_ind[0])
            losses['ce_loss'] = (loss * mask_query_idx.reshape(-1)).sum() / mask_query_idx.sum()

        if self.mse_loss:
            trans_embeds = torch.einsum('btij,btjc->btic', [att, embeds_refs])
            trans_embeds = trans_embeds.flatten(0,1).reshape(-1, *embed.shape[1:])
            trans_quants = self.vq_enc1(trans_embeds, encoding=False)[1]
            decs = self.vq_dec1(trans_quants)
            target = imgs[:,0,-1:].repeat(1,t-1,1,1,1).flatten(0,1)
            losses['mse_loss'] = self.mse_loss(decs, target)

        return losses
    
    

@MODELS.register_module()
class Vqvae_Tracker_V5(Vqvae_Tracker):

    def __init__(self,
                 l1_loss=None,
                 **kwargs
                 ):
        """ ce_loss with cross video/sparstity constrant
        """
        super(Vqvae_Tracker_V5, self).__init__(**kwargs)
        
        self.l1_loss = build_loss(l1_loss)

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
        refs_cross_video = [refs[0] for i in range(bsz)]

        out, att = non_local_attention(tar, refs, per_ref=True)
        out_cross_video, _, att_cross_video = non_local_attention(tar, refs_cross_video, per_ref=False)

        losses = {}

        if self.ce_loss:
            for idx, i in enumerate(range(self.num_head)):
                i = str(i).replace('0', '')
                predict = getattr(self, f'predictor{i}')(out)
                predict_cross_video = getattr(self, f'predictor{i}')(out_cross_video)
                loss = self.ce_loss(predict, out_ind[idx])
                loss_cross_video = self.ce_loss(predict_cross_video, out_ind[idx])
                losses[f'ce{i}_loss'] = (loss * mask_query_idx.reshape(-1)).sum() / mask_query_idx.sum() * self.multi_head_weight[idx]
                losses[f'ce{i}_cross_loss'] = (loss_cross_video * mask_query_idx.reshape(-1)).sum() / mask_query_idx.sum() * self.multi_head_weight[idx]

        if self.l1_loss:
            label = torch.zeros(*att_cross_video.shape).cuda()
            m = torch.eye(bsz).cuda()
            m = (1 - m)[:,:,None,None].repeat(1,1,*att_cross_video.shape[-2:])
            losses['l1_loss'] = self.l1_loss(att_cross_video, label, weight=m)

        return losses
    
    

@MODELS.register_module()
class Vqvae_Tracker_V6(BaseModel):

    def __init__(self,
                 backbone,
                 vqvae,
                 patch_size,
                 pretrained_vq,
                 temperature,
                 video_num=3457,
                 ce_loss=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None
                 ):
        """ original vqvae tracker
        """
        super(Vqvae_Tracker_V6, self).__init__()
        if not isinstance(vqvae, list):
            vqvae = list([vqvae])
            pretrained_vq = list([pretrained_vq])

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.patch_size = patch_size
        self.num_head = len(vqvae)
        self.video_num = video_num


        logger = get_root_logger()

        self.backbone = build_backbone(backbone)

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
        self.ce_loss = build_loss(ce_loss)

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
        for i in range(video_num):
            setattr(self, f'predictor{i}', nn.Linear(self.backbone.feat_dim, self.n_embed))
        
        if pretrained:
            _ = load_checkpoint(self, pretrained, map_location='cpu')
            logger.info('load pretrained model successfully')

    def forward_train(self, imgs, mask_query_idx, video_idx, jitter_imgs=None):

        bsz, num_clips, t, c, h, w = imgs.shape

        # vqvae tokenize for query frame
        with torch.no_grad():
            for i in range(self.num_head):
                i = str(i).replace('0', '')
                vqvae = getattr(self, f'vqvae{i}')
                vq_enc = getattr(self, f'vq_enc{i}')
                vqvae.eval()
                emb, _, _, ind, _ = vq_enc(imgs[:, 0, -1])
                ind = ind.unsqueeze(1).repeat(1, t-1, 1, 1).long().detach()

        if jitter_imgs is not None:
            imgs = jitter_imgs

        mask_query_idx = mask_query_idx.bool().unsqueeze(1).repeat(1,t-1,1)

        tar = self.backbone(imgs[:,0,-1])
        refs = list([self.backbone(imgs[:,0,i]) for i in range(t-1)])

        if self.patch_size != -1:
            out, att = local_attention(self.correlation_sampler, tar, refs, self.patch_size)
        else:
            out, att = non_local_attention(tar, refs, per_ref=True, flatten=False)

        losses = {}

        loss = 0
        for i in range(bsz):
            idx = video_idx[i].item() 
            predictor = getattr(self, f'predictor{idx}')
            predict = predictor(out[i, 0])
            loss_per_video = self.ce_loss(predict, ind[i].reshape(-1,1))
            mask = mask_query_idx[i:i+1]
            
            loss_per_video = (loss_per_video * mask.reshape(-1)).sum() / (mask.sum() + 1e-9)
            loss += loss_per_video
            
        losses['ce_loss'] = loss / bsz


        return losses
    
    
@MODELS.register_module()
class Vqvae_Tracker_V7(BaseModel):

    def __init__(self,
                 backbone,
                 vqvae,
                 patch_size,
                 pretrained_vq,
                 temperature,
                 ce_loss=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None
                 ):
        """ original vqvae tracker
        """
        super(Vqvae_Tracker_V7, self).__init__()
        if not isinstance(vqvae, list):
            vqvae = list([vqvae])
            if pretrained_vq is not None:
                pretrained_vq = list([pretrained_vq])

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.patch_size = patch_size
        self.num_head = len(vqvae)

        logger = get_root_logger()

        self.backbone = build_backbone(backbone)

        self.vq_type = vqvae[0].type
        if self.vq_type != 'DALLE_Encoder':
            for idx, i in enumerate(range(len(vqvae))):
                i = str(i).replace('0', '')
                setattr(self, f'vqvae{i}', build_model(vqvae[idx]).cuda())
                if pretrained_vq is not None:
                    _ = load_checkpoint(getattr(self, f'vqvae{i}'), pretrained_vq[idx], map_location='cpu')
                    logger.info(f'load {i}th pretrained VQVAE successfully')
                setattr(self, f'vq_emb{i}', getattr(self, f'vqvae{i}').quantize.embed)
                setattr(self, f'n_embed{i}', vqvae[idx].n_embed)
                setattr(self, f'vq_t{i}', temperature)
                setattr(self, f'vq_enc_per_video{i}', getattr(self, f'vqvae{i}').encode_per_video)
        else:
            assert self.num_head == 1
            self.vq_enc = load_model(pretrained_vq).cuda()
            self.n_embed = self.vq_enc.vocab_size
            logger.info('load pretrained VQVAE successfully')

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

        # fc
        for i in range(len(vqvae)):
            i = str(i).replace('0', '')
            setattr(self, f'predictor{i}', nn.Linear(self.backbone.feat_dim, getattr(self, f'n_embed{i}')))

        if pretrained:
            _ = load_checkpoint(self, pretrained, map_location='cpu')
            logger.info('load pretrained model successfully')

    def forward_train(self, imgs, mask_query_idx, img_meta, jitter_imgs=None):

        bsz, num_clips, t, c, h, w = imgs.shape

        # vqvae tokenize for query frame
        with torch.no_grad():
            out_ind = []
            for i in range(self.num_head):
                inds = []
                i = str(i).replace('0', '')
                vqvae = getattr(self, f'vqvae{i}')
                vq_enc_per_video = getattr(self, f'vq_enc_per_video{i}')
                vqvae.eval()
                
                for i in range(bsz):
                    img = imgs[i:i+1]
                    vname = img_meta[i]['video_name']
                    emb, _, _, ind, _ = vq_enc_per_video(img[:, 0, -1], vname)
                    ind = ind.unsqueeze(1).repeat(1, t-1, 1, 1)
                    inds.append(ind)
                    
                ind = torch.cat(inds, 0).reshape(-1, 1).long().detach()
                out_ind.append(ind)

        if jitter_imgs is not None:
            imgs = jitter_imgs

        mask_query_idx = mask_query_idx.bool().unsqueeze(1).repeat(1,t-1,1)

        tar = self.backbone(imgs[:,0,-1])
        refs = list([self.backbone(imgs[:,0,i]) for i in range(t-1)])

        if self.patch_size != -1:
            out, att = local_attention(self.correlation_sampler, tar, refs, self.patch_size)
        else:
            out, att = non_local_attention(tar, refs, per_ref=True)

        losses = {}

        if self.ce_loss:
            for idx, i in enumerate(range(self.num_head)):
                i = str(i).replace('0', '')
                predict = getattr(self, f'predictor{i}')(out)
                loss = self.ce_loss(predict, out_ind[idx])
                losses[f'ce{i}_loss'] = (loss * mask_query_idx.reshape(-1)).sum() / mask_query_idx.sum()
        
        return losses
    
    
@MODELS.register_module()
class Vqvae_Tracker_V8(BaseModel):

    def __init__(self,
                 backbone,
                 vqvae,
                 patch_size,
                 pretrained_vq,
                 temperature,
                 video_num,
                 ce_loss=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None
                 ):
        """ original vqvae tracker
        """
        super(Vqvae_Tracker_V8, self).__init__()
        if not isinstance(vqvae, list):
            vqvae = list([vqvae])
            if pretrained_vq is not None:
                pretrained_vq = list([pretrained_vq])

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.patch_size = patch_size
        self.num_head = len(vqvae)

        logger = get_root_logger()

        self.backbone = build_backbone(backbone)

        self.vq_type = vqvae[0].type
        if self.vq_type != 'DALLE_Encoder':
            for idx, i in enumerate(range(len(vqvae))):
                i = str(i).replace('0', '')
                setattr(self, f'vqvae{i}', build_model(vqvae[idx]).cuda())
                if pretrained_vq is not None:
                    _ = load_checkpoint(getattr(self, f'vqvae{i}'), pretrained_vq[idx], map_location='cpu')
                    logger.info(f'load {i}th pretrained VQVAE successfully')
                setattr(self, f'vq_emb{i}', getattr(self, f'vqvae{i}').quantize.embed)
                setattr(self, f'n_embed{i}', vqvae[idx].n_embed)
                setattr(self, f'vq_t{i}', temperature)
                setattr(self, f'vq_enc_per_video{i}', getattr(self, f'vqvae{i}').encode_per_video)
        else:
            assert self.num_head == 1
            self.vq_enc = load_model(pretrained_vq).cuda()
            self.n_embed = self.vq_enc.vocab_size
            logger.info('load pretrained VQVAE successfully')

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

        # fc
        for i in range(video_num):
            setattr(self, f'predictor{i}', nn.Linear(self.backbone.feat_dim, self.n_embed))

        if pretrained:
            _ = load_checkpoint(self, pretrained, map_location='cpu')
            logger.info('load pretrained model successfully')

    def forward_train(self, imgs, mask_query_idx, img_meta, video_idx, jitter_imgs=None):

        bsz, num_clips, t, c, h, w = imgs.shape

        # vqvae tokenize for query frame
        with torch.no_grad():
            out_ind = []
            for i in range(self.num_head):
                inds = []
                i = str(i).replace('0', '')
                vqvae = getattr(self, f'vqvae{i}')
                vq_enc_per_video = getattr(self, f'vq_enc_per_video{i}')
                vqvae.eval()
                
                for i in range(bsz):
                    img = imgs[i:i+1]
                    vname = img_meta[i]['video_name']
                    emb, _, _, ind, _ = vq_enc_per_video(img[:, 0, -1], vname)
                    ind = ind.unsqueeze(1).repeat(1, t-1, 1, 1)
                    inds.append(ind)
                    
                ind = torch.cat(inds, 0).long().detach()
                out_ind.append(ind)

        if jitter_imgs is not None:
            imgs = jitter_imgs

        mask_query_idx = mask_query_idx.bool().unsqueeze(1).repeat(1,t-1,1)

        tar = self.backbone(imgs[:,0,-1])
        refs = list([self.backbone(imgs[:,0,i]) for i in range(t-1)])

        if self.patch_size != -1:
            out, att = local_attention(self.correlation_sampler, tar, refs, self.patch_size)
        else:
            out, att = non_local_attention(tar, refs, per_ref=True, flatten=False)

        losses = {}

        loss = 0
        for i in range(bsz):
            idx = video_idx[i].item() 
            predictor = getattr(self, f'predictor{idx}')
            predict = predictor(out[i, 0])
            loss_per_video = self.ce_loss(predict, ind[i].reshape(-1,1))
            mask = mask_query_idx[i:i+1]
            
            loss_per_video = (loss_per_video * mask.reshape(-1)).sum() / (mask.sum() + 1e-9)
            loss += loss_per_video
            
        losses['ce_loss'] = loss / bsz
        
        return losses