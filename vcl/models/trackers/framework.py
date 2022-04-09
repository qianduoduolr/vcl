import numbers
from os import stat_result
import os.path as osp
from collections import *
from tkinter.messagebox import NO

import mmcv
from mmcv.runner import auto_fp16, load_state_dict, load_checkpoint

from ..base import BaseModel
from ..builder import build_backbone, build_loss, build_model
from ..registry import MODELS
from vcl.utils.helpers import *
from vcl.utils import *
from vcl.models.common import *


import torch.nn as nn
import torch
import torch.nn.functional as F



@MODELS.register_module()
class Framework(BaseModel):

    def __init__(self,
                 backbone,
                 backbone_t,
                 momentum,
                 temperature,
                 vq,
                 pretrained_vq,
                 temperature_t=1.0,
                 downsample_rate=8,
                 feat_size=32,
                 mask_radius=-1,
                 loss=None,
                 ce_loss=None,
                 loss_weight=None,
                 scaling=True,
                 norm=False,
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

        super().__init__()

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.downsample_rate = downsample_rate

        self.momentum = momentum
        self.pretrained = pretrained
        self.temperature = temperature
        self.temperature_t = temperature_t
        self.scaling = scaling
        self.norm = norm
        self.loss_weight = loss_weight

        self.logger = get_root_logger()

        # build backbone
        self.backbone = build_backbone(backbone)
        self.backbone_t = build_backbone(backbone_t)
        
        # build vq
        if not isinstance(vq, list):
            vq = list([vq])
            pretrained_vq = list([pretrained_vq])
        self.num_head = len(vq)
            
        for idx, i in enumerate(range(len(vq))):
                i = str(i).replace('0', '')
                setattr(self, f'vqvae{i}', build_model(vq[idx]).cuda())
                _ = load_checkpoint(getattr(self, f'vqvae{i}'), pretrained_vq[idx], map_location='cpu')
                self.logger.info(f'load {i}th pretrained VQVAE successfully')
                setattr(self, f'vq_emb{i}', getattr(self, f'vqvae{i}').quantize.embed)
                setattr(self, f'n_embed{i}', vq[idx].n_embed)
                setattr(self, f'vq_t{i}', temperature)
                setattr(self, f'vq_enc{i}', getattr(self, f'vqvae{i}').encode)
 
        # loss
        self.loss = build_loss(loss) if loss != None else None
        self.ce_loss = build_loss(ce_loss) if ce_loss != None else None
        
        if mask_radius != -1:
            self.mask = make_mask(feat_size, mask_radius)
        else:
            self.mask = None
        
        self.init_weights()
    
    def init_weights(self):
        
        self.backbone.init_weights()
        self.backbone_t.init_weights()
        
        if self.pretrained is not None:
            _ = load_checkpoint(self, self.pretrained, map_location='cpu')


    def forward_train(self, imgs, images_lab=None):

        bsz, num_clips, t, c, h, w = imgs.shape

        images_lab_gt = [images_lab[:,0,i,:].clone() for i in range(t)]
        images_lab = [images_lab[:,0,i,:] for i in range(t)]
        _, ch = self.dropout2d_lab(images_lab)
        
        # forward to get feature
        fs1, fs2 = self.backbone(torch.stack(images_lab,1).flatten(0,1))
        fs1 = fs1.reshape(bsz, t, *fs1.shape[-3:])
        tar1, refs1 = fs1[:, -1], fs1[:, :-1]
        fs2 = fs2.reshape(bsz, t, *fs2.shape[-3:])
        tar2, refs2 = fs2[:, -1], fs2[:, :-1]
        
        _, att = non_local_attention(tar1, refs1, scaling=True, mask=self.mask)
        _, att_l = non_local_attention(tar1, refs1, scaling=self.scaling)
        _, att_g = non_local_attention(tar2, refs2, scaling=self.scaling)


        with torch.no_grad():
            self.backbone_t.eval()
            fs_t1, fs_t2 = self.backbone_t(imgs.flatten(0,2))
            fs_t1 = fs_t1.reshape(bsz, t, *fs_t1.shape[-3:])
            tar_t1, refs_t1 = fs_t1[:, -1], fs_t1[:, :-1]
            
            fs_t2 = fs_t2.reshape(bsz, t, *fs_t2.shape[-3:])
            tar_t2, refs_t2 = fs_t2[:, -1], fs_t2[:, :-1]

            _, target_att1 = non_local_attention(tar_t1, refs_t1, temprature=self.temperature_t)
            _, target_att2 = non_local_attention(tar_t2, refs_t2, temprature=self.temperature_t)
            

        losses = {}
        losses['att1_loss'] = self.loss_weight['att1_loss'] * self.loss(att_l, target_att1)
        losses['att2_loss'] = self.loss_weight['att2_loss'] * self.loss(att_g, target_att2)
        
        
        # for mast l1_loss
        ref_gt = [self.prep(gt[:,ch]) for gt in images_lab_gt[:-1]]
        outputs = frame_transform(att, ref_gt, flatten=False)
        outputs = outputs[:,0].permute(0,2,1).reshape(bsz, -1, *fs1.shape[-2:])     
        loss, _ = self.compute_lphoto(images_lab_gt, ch, outputs)
        losses['l1_loss'] = self.loss_weight['l1_loss'] * loss
        
        
        # for ce_loss
        if self.ce_loss:
            # vqvae tokenize for query frame
            with torch.no_grad():
                out_ind= []
                outs = []
                vq_inds = []
                for i in range(self.num_head):
                    i = str(i).replace('0', '')
                    vqvae = getattr(self, f'vqvae{i}')
                    vq_enc = getattr(self, f'vq_enc{i}')
                    vqvae.eval()
                    encs, quants, _, inds, _ = vq_enc(imgs.flatten(0,2))
            
                    quants = quants.reshape(bsz, t, *quants.shape[-3:])
                    out_quant_refs = quants[:, :-1].flatten(3).permute(0, 1, 3, 2)
                    outs.append(out_quant_refs)
        
                    inds = inds.reshape(bsz, t, *inds.shape[-2:])
                    vq_inds.append([inds[:, -1], inds[:, -2]])
                
                    ind = inds[:, -1].unsqueeze(1).repeat(1, t-1, 1, 1).reshape(-1, 1).long().detach()
                    out_ind.append(ind)

            
            for idx, i in enumerate(range(self.num_head)):                
                i = str(i).replace('0', '')
                
                # change query if use vq sample
                mask_query_idx = self.query_vq_sample(vq_inds[idx][0], vq_inds[idx][1], t, self.mask)
                
                out = frame_transform(att, outs[idx], per_ref=True)
                vq_emb = getattr(self, f'vq_emb{i}')
                out = F.normalize(out, dim=-1)
                vq_emb = F.normalize(vq_emb, dim=0)
                predict = torch.mm(out, vq_emb)
                    
                loss = self.ce_loss(predict, out_ind[idx])                
                losses[f'ce{i}_loss'] = self.loss_weight[f'ce{i}_loss'] * (loss * mask_query_idx.reshape(-1)).sum() / mask_query_idx.sum()

        return losses

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

        if self.momentum is not -1:
            moment_update(self.backbone, self.backbone_t, self.momentum)

        log_vars.pop('loss')
        outputs = dict(
            log_vars=log_vars,
            num_samples=len(data_batch['imgs'])
        )

        return outputs
    
    def dropout2d_lab(self, arr): # drop same layers for all images
        if not self.training:
            return arr

        drop_ch_num = int(np.random.choice(np.arange(1, 2), 1))
        drop_ch_ind = np.random.choice(np.arange(1,3), drop_ch_num, replace=False)

        for a in arr:
            for dropout_ch in drop_ch_ind:
                a[:, dropout_ch] = 0
            a *= (3 / (3 - drop_ch_num))

        return arr, drop_ch_ind # return channels not masked
    
    def compute_lphoto(self, images_lab_gt, ch, outputs):
        
        b, c, h, w = images_lab_gt[0].size()
        tar_y = images_lab_gt[-1][:,ch]  # y4
        outputs = F.interpolate(outputs, (h, w), mode='bilinear')
        loss = F.smooth_l1_loss(outputs*20, tar_y*20, reduction='mean')
        err_maps = torch.abs(outputs - tar_y).sum(1).detach()

        return loss, err_maps
    
    def prep(self, image, mode='default'):
        
        bsz,c,_,_ = image.size()
        x = image.float()[:,:,::self.downsample_rate,::self.downsample_rate]

        return x
    
    def query_vq_sample(self, ind_tar, ind_ref, t, mask=None):
        
        # determined query
        ind_tar = ind_tar.flatten(1).unsqueeze(-1).repeat(1,1,ind_tar.shape[-1] * ind_tar.shape[-2])
        ind_ref = ind_ref.flatten(1).unsqueeze(1).repeat(1,ind_ref.shape[-1] * ind_ref.shape[-2],1)
        mask_query_idx = ((ind_tar == ind_ref) * mask.unsqueeze(0)).sum(-1) if mask != None else ((ind_tar == ind_ref)).sum(-1)
        mask_query_idx = (mask_query_idx > 0)
        mask_query_idx = mask_query_idx.bool().unsqueeze(1).repeat(1,t-1,1)
        
        return mask_query_idx
    
    

@MODELS.register_module()
class Framework_V2(BaseModel):

    def __init__(self,
                 backbone,
                 backbone_t,
                 momentum,
                 temperature,
                 pool_type='mean',
                 weight=20,
                 num_stage=2,
                 feat_size=[64, 32],
                 radius=[12, 6],
                 downsample_rate=[4, 8],
                 temperature_t=1.0,
                 loss=None,
                 loss_weight=None,
                 scaling=True,
                 norm=False,
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

        super().__init__()
        
        self.num_stage = num_stage
        self.feat_size = feat_size
        self.pool_type = pool_type

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.downsample_rate = downsample_rate

        self.momentum = momentum
        self.pretrained = pretrained
        self.temperature = temperature
        self.temperature_t = temperature_t
        self.scaling = scaling
        self.norm = norm
        self.loss_weight = loss_weight
        self.logger = get_root_logger()

        # build backbone
        self.backbone = build_backbone(backbone)
        self.backbone_t = build_backbone(backbone_t)
 
        # loss
        self.loss = build_loss(loss) if loss != None else None
        self.mask = [ make_mask(feat_size[i], radius[i]) for i in range(len(radius))]
        
        self.weight = weight
        self.init_weights()
    
    def init_weights(self):
        
        self.backbone.init_weights()
        self.backbone_t.init_weights()
        
        if self.pretrained is not None:
            _ = load_checkpoint(self, self.pretrained, map_location='cpu')
    
    def forward_train(self, imgs, images_lab=None):
            
        bsz, num_clips, t, c, h, w = imgs.shape
        
        images_lab_gt = [images_lab[:,0,i,:].clone() for i in range(t)]
        images_lab = [images_lab[:,0,i,:] for i in range(t)]
        _, ch = self.dropout2d_lab(images_lab)
        
        # forward to get feature
        fs = self.backbone(torch.stack(images_lab,1).flatten(0,1))
        fs = [f.reshape(bsz, t, *f.shape[-3:]) for f in fs]
        
        tar_pyramid, refs_pyramid = [f[:, -1] for f in fs], [ f[:, :-1] for f in fs]
        
        losses = {}
        
        atts = []
        atts_dis = []
        for idx, (tar, refs) in enumerate(zip(tar_pyramid, refs_pyramid)):
            # get correlation for distillation
            if idx == len(fs) - 1:
                _, att_g = non_local_attention(tar, refs, scaling=self.scaling)
                break
            # get correlation attention map            
            _, att = non_local_attention(tar, refs, mask=self.mask[idx], scaling=True)            
            # for mast l1_loss
            ref_gt = [self.prep(gt[:,ch], downsample_rate=self.downsample_rate[idx]) for gt in images_lab_gt[:-1]]
            outputs = frame_transform(att, ref_gt, flatten=False)
            outputs = outputs[:,0].permute(0,2,1).reshape(bsz, -1, self.feat_size[idx], self.feat_size[idx])     
            losses[f'stage{idx}_l1_loss'] = self.compute_lphoto(images_lab_gt, ch, outputs)[0] * self.loss_weight[f'stage{idx}_l1_loss']
            atts.append(att)
        
        # for layer distillation loss
        if self.pool_type == 'mean':
            att_ = atts[0].reshape(bsz, -1, *fs[0].shape[-2:])
            att_ = F.avg_pool2d(att_, 2, stride=2).flatten(-2).permute(0,2,1).reshape(bsz, -1, *fs[0].shape[-2:])
            target = F.avg_pool2d(att_, 2, stride=2).flatten(-2).permute(0,2,1)
        elif self.pool_type == 'max':
            att_ = atts[0].reshape(bsz, -1, *fs[0].shape[-2:])
            att_ = F.max_pool2d(att_, 2, stride=2).flatten(-2).permute(0,2,1).reshape(bsz, -1, *fs[0].shape[-2:])
            target = F.max_pool2d(att_, 2, stride=2).flatten(-2).permute(0,2,1)
            
        losses['layer_dist_loss'] = self.loss_weight['layer_dist_loss'] * self.loss(atts[-1][:,0], target)
        
        # for correlation distillation loss
        with torch.no_grad():
            self.backbone_t.eval()
            fs_t = self.backbone_t(imgs.flatten(0,2))
            fs_t = fs_t.reshape(bsz, t, *fs_t.shape[-3:])
            tar_t, refs_t = fs_t[:, -1], fs_t[:, :-1]
            _, target_att = non_local_attention(tar_t, refs_t, temprature=self.temperature_t, norm=self.norm)
            
        losses['correlation_dist_loss'] = self.loss_weight['correlation_dist_loss'] * self.loss(att_g, target_att)
            
        return losses
    
    def dropout2d_lab(self, arr): # drop same layers for all images
        if not self.training:
            return arr

        drop_ch_num = int(np.random.choice(np.arange(1, 2), 1))
        drop_ch_ind = np.random.choice(np.arange(1,3), drop_ch_num, replace=False)

        for a in arr:
            for dropout_ch in drop_ch_ind:
                a[:, dropout_ch] = 0
            a *= (3 / (3 - drop_ch_num))

        return arr, drop_ch_ind # return channels not masked
    
    def compute_lphoto(self, images_lab_gt, ch, outputs, upsample=True):
        b, c, h, w = images_lab_gt[0].size()

        tar_y = images_lab_gt[-1][:,ch]  # y4

        if upsample:
            outputs = F.interpolate(outputs, (h, w), mode='bilinear')
            loss = F.smooth_l1_loss(outputs * self.weight, tar_y * self.weight, reduction='mean')
        else:
            tar_y = self.prep(images_lab_gt[-1])[:,ch]
            loss = F.smooth_l1_loss(outputs * self.weight, tar_y * self.weight, reduction='mean')

        err_maps = torch.abs(outputs - tar_y).sum(1).detach()

        return loss, err_maps
    
    def prep(self, image, downsample_rate):
        _,c,_,_ = image.size()
        x = image.float()[:,:,::downsample_rate,::downsample_rate]

        return x