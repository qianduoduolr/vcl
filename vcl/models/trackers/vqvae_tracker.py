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
                 temperature=1.0,
                 sim_siam_head=None,
                 multi_head_weight=[1.0],
                 ce_loss=None,
                 mse_loss=None,
                 fc=True,
                 train_cfg=None,
                 test_cfg=None,
                 per_ref=True,
                 mask_radius=-1,
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
        self.temperature = temperature

        self.logger = get_root_logger()

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
                self.logger.info(f'load {i}th pretrained VQVAE successfully')
                setattr(self, f'vq_emb{i}', getattr(self, f'vqvae{i}').quantize.embed)
                setattr(self, f'n_embed{i}', vqvae[idx].n_embed)
                setattr(self, f'vq_t{i}', temperature)
                setattr(self, f'vq_enc{i}', getattr(self, f'vqvae{i}').encode)
        else:
            assert self.num_head == 1
            self.vq_enc = load_model(pretrained_vq).cuda()
            self.n_embed = self.vq_enc.vocab_size
            self.logger.info('load pretrained VQVAE successfully')

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

        # init weights
        self.init_weights(pretrained)
        
        if mask_radius != -1:
            self.mask = make_mask(32, mask_radius)
        else:
            self.mask = None
        
            
    def init_weights(self, pretrained):
        
        if pretrained:
            _ = load_checkpoint(self, pretrained, map_location='cpu')
            self.logger.info('load pretrained model successfully')
        
        return 

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
                else:
                    ind = ind.reshape(-1, 1).long().detach()
                    quant = quant.permute(0,2,3,1).flatten(0,2).detach()
                    
                out_ind.append(ind)
                out_quant.append(quant)
                
        mask_query_idx = mask_query_idx.bool().unsqueeze(1).repeat(1,t-1,1) if self.per_ref else mask_query_idx.bool()
            
        if jitter_imgs is not None:
            imgs = jitter_imgs

        tar = self.backbone(imgs[:,0,-1])
        refs = list([self.backbone(imgs[:,0,i]) for i in range(t-1)])

        if self.patch_size != -1:
            out, att = local_attention(self.correlation_sampler, tar, refs, self.patch_size)
        else:
            out, att = non_local_attention(tar, refs, per_ref=self.per_ref, temprature=self.temperature)

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
        
        if self.mask:
            mask = torch.ones(*att.shape).cuda() - self.mask
            target = torch.zeros(*att.shape).cuda()
            loss = F.l1_loss(mask*att, target, reduction='none')
            losses['att_sparse_loss'] = 10 * (loss * mask_query_idx.unsqueeze(-1)).sum() / mask_query_idx.sum()
                    
        return losses

    def forward_test(self, imgs, mask_query_idx,
                    save_image=False,
                    save_path=None,
                    iteration=None):
        bsz, num_clips, t, c, h, w = imgs.shape
        
        tar = self.backbone(imgs[:,0, 0])
        refs = list([self.backbone(imgs[:,0,i]) for i in range(1,t)])
        atts = []
        
        # for short term
        out, att_s = non_local_attention(tar, [refs[0]], per_ref=self.per_ref)
        
        # for long term
        if True:
            _, att = non_local_attention(tar, refs, per_ref=self.per_ref)
            att = att.reshape(bsz, att_s.shape[-1], t-1, -1)
            for i in range(t-1):
                atts.append(att[:,:,i])
        else:
            att_l = torch.eye(att_s.shape[-1]).repeat(bsz, 1, 1).cuda()
            for i in range(t-1):
                if i == 0:
                    att_l = torch.einsum('bij,bjk->bik', [att_l, att_s]) 
                else:
                    _, att = non_local_attention(refs[i-1], [refs[i]], per_ref=self.per_ref)
                    att_l = torch.einsum('bij,bjk->bik', [att_l, att]) 
                    
                atts.append(att_l)
        
        if len(atts) == 0:
            atts.append(att_s)
            
        visualize_att(imgs, atts, iteration, True, mask_query_idx, tar.shape[-1], self.patch_size, dst_path=save_path, norm_mode='mean-std')

        # vqvae tokenize for query frame
        with torch.no_grad():
            _, quant, diff, ind, embed = self.vq_enc(imgs[:, 0, -1])
            ind = ind.reshape(-1, 1).long().detach()


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
        """ per video per fc
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
        """
        per video per vq and gloable fc
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
    """ per video per vq and fc

    Args:
        BaseModel ([type]): [description]
    """
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
    

@MODELS.register_module()
class Vqvae_Tracker_V9(Vqvae_Tracker):
    """
    Args:
        Vqvae_Tracker ([type]): add a ce_loss on target frame
    """        
    
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
            out, att = non_local_attention(tar, refs, per_ref=self.per_ref, temprature=self.temperature)

        losses = {}

        # if self.ce_loss:
        #     for idx, i in enumerate(range(self.num_head)):
        #         i = str(i).replace('0', '')
        #         if self.fc:
        #             predict = getattr(self, f'predictor{i}')(out)
        #         else:
        #             predict = self.embedding_layer(out)
        #             predict = nn.functional.normalize(predict, dim=-1)
        #             vq_emb = getattr(self, f'vq_emb{i}')
        #             predict = torch.mm(predict, nn.functional.normalize(vq_emb, dim=0))
        #             predict = torch.div(predict, 0.1) # temperature is set to 0.1
                    
        #         loss = self.ce_loss(predict, out_ind[idx])
        #         losses[f'ce{i}_loss'] = (loss * mask_query_idx.reshape(-1)).sum() / mask_query_idx.sum() * self.multi_head_weight[idx]

        predict_tar = self.predictor(tar.permute(0,2,3,1).flatten(0,2))
        loss_tar = self.ce_loss(predict_tar, out_ind[0])
        losses[f'target_ce_loss'] = (loss_tar * mask_query_idx.reshape(-1)).sum() / mask_query_idx.sum()
        
        return losses
        

@MODELS.register_module()
class Vqvae_Tracker_V10(Vqvae_Tracker):
    """
    Args:
        Vqvae_Tracker ([type]): [induse long-term relationship]
    """        
    def __init__(self, soft_ce_loss, **kwargs):
        super().__init__(**kwargs)
        self.soft_ce_loss = build_loss(soft_ce_loss)
        self.fc = nn.Sequential(
            nn.Conv2d(512,2048,1),
            nn.ReLU(),
            nn.Conv2d(2048,512,1)
        )
    
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
                emb, quant, _, ind, _ = vq_enc(imgs[:, 0, -1])
                
                if self.per_ref:
                    ind = ind.unsqueeze(1).repeat(1, t-1, 1, 1).reshape(-1, 1).long().detach()
                    mask_query_idx = mask_query_idx.bool().unsqueeze(1).repeat(1,t-1,1)
                else:
                    ind = ind.reshape(-1, 1).long().detach()
                    mask_query_idx = mask_query_idx.bool()
                
                out_ind.append(ind)

        if jitter_imgs is not None:
            imgs = jitter_imgs

        tar = self.backbone(imgs[:,0, 0])
        refs = list([self.backbone(imgs[:,0,i]) for i in range(1,t)])
        
        # for short term
        out_s, att_s = non_local_attention(tar, [refs[0]], per_ref=self.per_ref)
        
        # for long term
        att_l = torch.eye(att_s.shape[-1]).repeat(bsz, 1, 1).cuda()
        refs.insert(0, tar)

        fs = torch.stack(refs, 1).flatten(0,1)
        fs = nn.functional.normalize(self.fc(fs), dim=1).permute(0, 2, 3, 1).reshape(bsz, t, att_s.shape[-1], -1)

        atts = torch.einsum('btic,btjc->btij',[fs[:,:-1], fs[:,1:]]) / 0.05
        atts_reverse = torch.flip(atts, [1]).permute(0,1,3,2)
        atts_cycle = torch.cat([atts, atts_reverse], 1)
        for i in range(atts_cycle.shape[1]):
            att = atts_cycle[:, i].softmax(dim=-1)
            att_l = torch.einsum('bij,bjk->bik', [att_l, att])
            
        del fs, refs, atts, atts_cycle, atts_reverse
        
        label = torch.arange(att_s.shape[-1]).repeat(bsz, 1).cuda().reshape(-1, 1)
            
        losses = {}
        if self.ce_loss:
            for idx, i in enumerate(range(self.num_head)):
                i = str(i).replace('0', '')
                predict_s = getattr(self, f'predictor{i}')(out_s)
                # loss_s = self.ce_loss(predict_s, out_ind[idx])
                # loss_l = self.soft_ce_loss(att_l.flatten(0,1), att_s.flatten(0,1).detach())
                loss_l = self.ce_loss(att_l.flatten(0,1), label)
                # losses[f'ce{i}_short_loss'] = (loss_s * mask_query_idx.reshape(-1)).sum() / mask_query_idx.sum() * self.multi_head_weight[idx]
                losses[f'ce{i}_long_loss'] = (loss_l * mask_query_idx.reshape(-1)).sum() / mask_query_idx.sum()

        
        return losses
    



@MODELS.register_module()
class Vqvae_Tracker_V11(Vqvae_Tracker):
    """
    Args:
        Vqvae_Tracker ([type]): [induse long-term relationship  cycle-consistency]
    """        
    def __init__(self, soft_ce_loss, **kwargs):
        super().__init__(**kwargs)
        self.soft_ce_loss = build_loss(soft_ce_loss)
        self.fc = nn.Sequential(
            nn.Conv2d(512,2048,1),
            nn.ReLU(),
            nn.Conv2d(2048,512,1)
        )
        self._xent_targets = dict()
        self.edgedrop_rate = 0.1
        self.dropout = nn.Dropout(p=self.edgedrop_rate, inplace=False)
        
        # self.featdrop_rate = getattr(args, 'featdrop', 0)
        self.temperature = 0.07
        
    
    def infer_dims(self):
        in_sz = 256
        dummy = torch.zeros(1, 3, 1, in_sz, in_sz).to(next(self.encoder.parameters()).device)
        dummy_out = self.encoder(dummy)
        self.enc_hid_dim = dummy_out.shape[1]
        self.map_scale = in_sz // dummy_out.shape[-1]

    def make_head(self, depth=1):
        head = []
        if depth >= 0:
            dims = [self.enc_hid_dim] + [self.enc_hid_dim] * depth + [128]
            for d1, d2 in zip(dims, dims[1:]):
                h = nn.Linear(d1, d2)
                head += [h, nn.ReLU()]
            head = head[:-1]

        return nn.Sequential(*head)

    def zeroout_diag(self, A, zero=0):
        mask = (torch.eye(A.shape[-1]).unsqueeze(0).repeat(A.shape[0], 1, 1).bool() < 1).float().cuda()
        return A * mask

    def affinity(self, x1, x2):
        in_t_dim = x1.ndim
        if in_t_dim < 4:  # add in time dimension if not there
            x1, x2 = x1.unsqueeze(-2), x2.unsqueeze(-2)

        A = torch.einsum('bctn,bctm->btnm', x1, x2)
        # if self.restrict is not None:
        #     A = self.restrict(A)

        return A.squeeze(1) if in_t_dim < 4 else A
    
    def stoch_mat(self, A, zero_diagonal=False, do_dropout=True, do_sinkhorn=False):
        ''' Affinity -> Stochastic Matrix '''

        if zero_diagonal:
            A = self.zeroout_diag(A)

        if do_dropout and self.edgedrop_rate > 0:
            A[torch.rand_like(A) < self.edgedrop_rate] = -1e20

        return F.softmax(A/self.temperature, dim=-1)
    
    
    def xent_targets(self, A):
        B, N = A.shape[:2]
        key = '%s:%sx%s' % (str(A.device), B,N)

        if key not in self._xent_targets:
            I = torch.arange(A.shape[-1])[None].repeat(B, 1)
            self._xent_targets[key] = I.view(-1).to(A.device)

        return self._xent_targets[key]

    def forward_train_cycle(self, q, mask_query_idx=None):
        '''
        Input is B x T x N*C x H x W, where either
           N>1 -> list of patches of images
           N=1 -> list of images
        '''
        bsz, T, C, H, W = q.shape
        
        q = q.flatten(0,1)
        q = nn.functional.normalize(self.fc(q), dim=1).reshape(bsz, T, q.shape[1], -1).permute(0,2,1,3)
        # q = q.reshape(bsz, T, q.shape[1], -1).permute(0,2,1,3)
        
        #################################################################
        # Compute walks 
        #################################################################
        walks = dict()
        As = self.affinity(q[:, :, :-1], q[:, :, 1:])
        A12s = [self.stoch_mat(As[:, i], do_dropout=True) for i in range(T-1)]

        #################################################### Palindromes
        A21s = [self.stoch_mat(As[:, i].transpose(-1, -2), do_dropout=True) for i in range(T-1)]
        AAs = []
        for i in list(range(1, len(A12s))):
            g = A12s[:i+1] + A21s[:i+1][::-1]
            aar = aal = g[0]
            for _a in g[1:]:
                aar, aal = aar @ _a, _a @ aal

            AAs.append((f"r{i}", aar))

        for i, aa in AAs:
            walks[f"cyc {i}"] = [aa, self.xent_targets(aa)]


        #################################################################
        # Compute loss 
        #################################################################
        xents = [torch.tensor([0.]).cuda()]
        losses = dict()

        for name, (A, target) in walks.items():
            logits = torch.log(A+1e-20).flatten(0, -2)
            if mask_query_idx == None:
                loss = self.ce_loss(logits, target.unsqueeze(1)).mean()
            else:
                loss = self.ce_loss(logits, target.unsqueeze(1))
                loss = (loss * mask_query_idx.reshape(-1)).sum() / mask_query_idx.sum()
            acc = (torch.argmax(logits, dim=-1) == target).float().mean()
            # diags.update({f"{H} xent {name}": loss.detach(),
            #               f"{H} acc {name}": acc})
            xents += [loss]


        loss = sum(xents)/max(1, len(xents)-1)
        
        return loss


    def forward_train(self, imgs, mask_query_idx, jitter_imgs=None):
    
        bsz, num_clips, t, c, h, w = imgs.shape

        # self.eval()
        
        # vqvae tokenize for query frame
        with torch.no_grad():
            out_ind= []
            for i in range(self.num_head):
                i = str(i).replace('0', '')
                vqvae = getattr(self, f'vqvae{i}')
                vq_enc = getattr(self, f'vq_enc{i}')
                vqvae.eval()
                emb, quant, _, ind, _ = vq_enc(imgs[:, 0, -1])
                
                if self.per_ref:
                    ind = ind.unsqueeze(1).repeat(1, t-1, 1, 1).reshape(-1, 1).long().detach()
                    mask_query_idx = mask_query_idx.bool().unsqueeze(1).repeat(1,t-1,1)
                else:
                    ind = ind.reshape(-1, 1).long().detach()
                    mask_query_idx = mask_query_idx.bool()
                
                out_ind.append(ind)

        if jitter_imgs is not None:
            imgs = jitter_imgs

        tar = self.backbone(imgs[:,0, 0])
        refs = list([self.backbone(imgs[:,0,i]) for i in range(1,t)])

        
        # for short term
        out_s, att_s = non_local_attention(tar, [refs[0]], per_ref=self.per_ref)
     
        losses = {}
        if self.ce_loss:
            for idx, i in enumerate(range(self.num_head)):
                i = str(i).replace('0', '')
                predict_s = getattr(self, f'predictor{i}')(out_s)
                loss = self.ce_loss(predict_s, out_ind[idx])
                losses['ce_loss'] = (loss * mask_query_idx.reshape(-1)).sum() / mask_query_idx.sum()

        # for long term
        refs.insert(0, tar)
        all_feats = torch.stack(refs,1)
        losses['cycle_loss'] = self.forward_train_cycle(all_feats, mask_query_idx)
        
        # self.train()
        
        return losses
    
    

@MODELS.register_module()
class Vqvae_Tracker_V12(Vqvae_Tracker):
    
    def forward_train(self, imgs, mask_query_idx, frames_mask, jitter_imgs=None):
        
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
            out, att = non_local_attention(tar, refs, per_ref=self.per_ref, temprature=self.temperature)

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
        
        mask = torch.ones(*att.shape).cuda() - frames_mask[:,:-1,None].flatten(3)
        target = torch.zeros(*att.shape).cuda()
        loss_reg = F.l1_loss(mask*att, target, reduction='none')
        losses['att_sparse_flow_loss'] = 10 * (loss_reg * mask_query_idx.unsqueeze(-1)).sum() / mask_query_idx.sum()
                    
        return losses
    
    
@MODELS.register_module()
class Vqvae_Tracker_V13(Vqvae_Tracker):
    
    def forward_train(self, imgs, mask_query_idx, frames_mask, jitter_imgs=None):
        
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
        mask = frames_mask[:,:-1,None].flatten(3)

        if self.patch_size != -1:
            out, att = local_attention(self.correlation_sampler, tar, refs, self.patch_size)
        else:
            out, att = non_local_attention(tar, refs, per_ref=self.per_ref, temprature=self.temperature, mask=mask)
        
        losses = {}
        
        # a = att[0,0,0].detach().cpu().reshape(32,32).numpy()
        # b = mask[0,0,0].detach().cpu().reshape(32,32).numpy()

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
                    
        return losses
    
    
@MODELS.register_module()
class Vqvae_Tracker_V14(Vqvae_Tracker):
    
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
                emb, quant, _, ind_tar, _ = vq_enc(imgs[:, 0, -1])
                
                if self.per_ref:
                    ind = ind_tar.unsqueeze(1).repeat(1, t-1, 1, 1).reshape(-1, 1).long().detach()
                    quant = quant.unsqueeze(1).repeat(1, t-1, 1, 1, 1).permute(0,1,3,4,2).flatten(0,3).detach()
                else:
                    ind = ind_tar.reshape(-1, 1).long().detach()
                    quant = quant.permute(0,2,3,1).flatten(0,2).detach()
                    
                out_ind.append(ind)
                out_quant.append(quant)
                
            emb, quant, _, ind_ref, _ = vq_enc(imgs[:, 0, 0])
            
        if jitter_imgs is not None:
            imgs = jitter_imgs

        # determined query
        ind_tar = ind_tar.flatten(1).unsqueeze(-1).repeat(1,1,ind_tar.shape[-1] * ind_tar.shape[-2])
        ind_ref = ind_ref.flatten(1).unsqueeze(1).repeat(1,ind_ref.shape[-1] * ind_ref.shape[-2],1)
        mask_query_idx = ((ind_tar == ind_ref) * self.mask.unsqueeze(0)).sum(-1)
        mask_query_idx = (mask_query_idx > 0)
        if self.per_ref:
            mask_query_idx = mask_query_idx.bool().unsqueeze(1).repeat(1,t-1,1)
        
        tar = self.backbone(imgs[:,0,-1])
        refs = list([self.backbone(imgs[:,0,i]) for i in range(t-1)])

        if self.patch_size != -1:
            out, att = local_attention(self.correlation_sampler, tar, refs, self.patch_size)
        else:
            out, att = non_local_attention(tar, refs, per_ref=self.per_ref, temprature=self.temperature)
        
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
                    
        return losses
    
@MODELS.register_module()
class Vqvae_Tracker_V15(Vqvae_Tracker):
    
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
                emb, quant_tar, _, ind_tar, _ = vq_enc(imgs[:, 0, -1])
                
                if self.per_ref:
                    ind = ind_tar.unsqueeze(1).repeat(1, t-1, 1, 1).reshape(-1, 1).long().detach()
                else:
                    ind = ind_tar.reshape(-1, 1).long().detach()
                    
                out_ind.append(ind)
                
                emb, quant_ref, _, ind_ref, _ = vq_enc(imgs[:, 0, 0])
            
        if jitter_imgs is not None:
            imgs = jitter_imgs

        # determined query
        ind_tar = ind_tar.flatten(1).unsqueeze(-1).repeat(1,1,ind_tar.shape[-1] * ind_tar.shape[-2])
        ind_ref = ind_ref.flatten(1).unsqueeze(1).repeat(1,ind_ref.shape[-1] * ind_ref.shape[-2],1)
        mask_query_idx = ((ind_tar == ind_ref) * self.mask.unsqueeze(0)).sum(-1)
        mask_query_idx = (mask_query_idx > 0)
        if self.per_ref:
            mask_query_idx = mask_query_idx.bool().unsqueeze(1).repeat(1,t-1,1)
        
        tar = self.backbone(imgs[:,0,-1])
        refs = list([self.backbone(imgs[:,0,i]) for i in range(t-1)])

        if self.patch_size != -1:
            _, att = local_attention(self.correlation_sampler, tar, refs, self.patch_size)
        else:
            _, att = non_local_attention(tar, refs, per_ref=self.per_ref, temprature=self.temperature)
        
        losses = {}
        
        out = frame_transform(att, [quant_ref])
        
        if self.ce_loss:
            for idx, i in enumerate(range(self.num_head)):
                i = str(i).replace('0', '')
                if self.fc:
                    predict = getattr(self, f'predictor{i}')(out)
                else:
                    vq_emb = getattr(self, f'vq_emb{i}')
                    predict = torch.mm(out, vq_emb)
                    # predict = torch.div(predict, 0.1) # temperature is set to 0.1
                    
                loss = self.ce_loss(predict, out_ind[idx])
                losses[f'ce{i}_loss'] = (loss * mask_query_idx.reshape(-1)).sum() / mask_query_idx.sum() * self.multi_head_weight[idx]
                    
        return losses


@MODELS.register_module()
class Vqvae_Tracker_V16(Vqvae_Tracker):
    
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
                emb, quant_tar, _, ind_tar, _ = vq_enc(imgs[:, 0, -1])
                
                if self.per_ref:
                    ind = ind_tar.unsqueeze(1).repeat(1, t-1, 1, 1).reshape(-1, 1).long().detach()
                else:
                    ind = ind_tar.reshape(-1, 1).long().detach()
                    
                out_ind.append(ind)
                
                emb, quant_ref, _, ind_ref, _ = vq_enc(imgs[:, 0, 0])
            
        if jitter_imgs is not None:
            imgs = jitter_imgs

        # determined query
        ind_tar = ind_tar.flatten(1).unsqueeze(-1).repeat(1,1,ind_tar.shape[-1] * ind_tar.shape[-2])
        ind_ref = ind_ref.flatten(1).unsqueeze(1).repeat(1,ind_ref.shape[-1] * ind_ref.shape[-2],1)
        mask_query_idx = ((ind_tar == ind_ref) * self.mask.unsqueeze(0)).sum(-1)
        mask_query_idx = (mask_query_idx > 0)
        if self.per_ref:
            mask_query_idx = mask_query_idx.bool().unsqueeze(1).repeat(1,t-1,1)
        
        tar = self.backbone(imgs[:,0,-1])
        refs = list([self.backbone(imgs[:,0,i]) for i in range(t-1)])

        if self.patch_size != -1:
            _, att = local_attention(self.correlation_sampler, tar, refs, self.patch_size)
        else:
            _, att = non_local_attention(tar, refs, per_ref=self.per_ref, temprature=self.temperature)
        
        losses = {}
        
        out = frame_transform(att, [quant_ref])
        
        if self.ce_loss:
            for idx, i in enumerate(range(self.num_head)):
                i = str(i).replace('0', '')
                vq_emb = getattr(self, f'vq_emb{i}')
                predict = torch.mm(out, vq_emb)
                # predict = torch.div(predict, 0.1) # temperature is set to 0.1
                loss = self.ce_loss(predict, out_ind[idx])
                losses[f'ce{i}_loss'] = (loss * mask_query_idx.reshape(-1)).sum() / mask_query_idx.sum() * self.multi_head_weight[idx]
        
        predict_tar = self.predictor(tar.permute(0,2,3,1).flatten(0,2))
        loss_tar = self.ce_loss(predict_tar, out_ind[0])
        losses[f'target_ce_loss'] = (loss_tar * mask_query_idx.reshape(-1)).sum() / mask_query_idx.sum()
        
        return losses