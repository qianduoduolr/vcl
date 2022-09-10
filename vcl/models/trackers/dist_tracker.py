# Copyright (c) OpenMMLab. All rights reserved.
import numbers
from os import stat_result
import os.path as osp
from collections import *
from tkinter.messagebox import NO
from torch import distributed

import mmcv
from mmcv.runner import auto_fp16, load_state_dict, load_checkpoint

from ..base import BaseModel
from ..builder import build_backbone, build_components, build_loss
from ..registry import MODELS
from vcl.utils.helpers import *
from vcl.utils import *
from vcl.models.common import *


import torch.nn as nn
import torch
import torch.nn.functional as F
import math


@MODELS.register_module()
class Dist_Tracker(BaseModel):

    def __init__(self,
                 backbone,
                 backbone_t,
                 momentum,
                 temperature,
                 temperature_t=1.0,
                 downsample_rate=8,
                 feat_size=32,
                 mask_radius=-1,
                 loss=None,
                 loss_weight=None,
                 loss_feat=None,
                 l1_loss=None,
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

        super(Dist_Tracker, self).__init__()

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.l1_loss  = l1_loss
        self.downsample_rate = downsample_rate

        self.momentum = momentum
        self.feat_size = feat_size
        self.radius = mask_radius

        self.pretrained = pretrained
        self.temperature = temperature
        self.temperature_t = temperature_t
        self.scaling = scaling
        self.norm = norm
        self.loss_weight = loss_weight

        logger = get_root_logger()

        self.backbone = build_backbone(backbone)
        self.backbone_t = build_backbone(backbone_t)

        # loss
        self.loss = build_loss(loss)
        self.loss_feat = build_loss(loss)
        
        
        if mask_radius != -1:
            self.mask = make_mask(feat_size, mask_radius)
        else:
            self.mask = None


    def forward_train(self, imgs, images_lab=None):

        bsz, num_clips, t, c, h, w = imgs.shape

        images_lab_gt = [images_lab[:,0,i,:].clone() for i in range(t)]
        images_lab = [images_lab[:,0,i,:] for i in range(t)]
        _, ch = self.dropout2d_lab(images_lab)
        
        # forward to get feature
        fs = self.backbone(torch.stack(images_lab,1).flatten(0,1))
        fs = fs.reshape(bsz, t, *fs.shape[-3:])
        tar, refs = fs[:, -1], fs[:, :-1]
        _, att_g = non_local_attention(tar, refs)
        _, att = non_local_attention(tar, refs, mask=self.mask)


        with torch.no_grad():
            self.backbone_t.eval()
            fs_t = self.backbone_t(imgs.flatten(0,2))
            fs_t = fs_t.reshape(bsz, t, *fs_t.shape[-3:])
            tar_t, refs_t = fs_t[:, -1], fs_t[:, :-1]

            _, target_att = non_local_attention(tar_t, refs_t)

        losses = {}
        losses['att_loss'] = self.loss(att_g, target_att)
        
        
        # for mast l1_loss
        if self.l1_loss:
            ref_gt = [self.prep(gt[:,ch]) for gt in images_lab_gt[:-1]]
            outputs = frame_transform(att, ref_gt, flatten=False)
            outputs = outputs[:,0].permute(0,2,1).reshape(bsz, -1, *fs.shape[-2:])     
            losses['l1_loss'], _ = self.compute_lphoto(images_lab_gt, ch, outputs)

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
    


@MODELS.register_module()
class Dist_Tracker_V2(Dist_Tracker):
    
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
        
        _, att_g = non_local_attention(tar2, refs2, temprature=self.temperature, scaling=self.scaling, norm=self.norm)
        # _, att_g = non_local_attention(tar2, refs2)
        _, att = non_local_attention(tar1, refs1, scaling=True, mask=self.mask)


        with torch.no_grad():
            self.backbone_t.eval()
            fs_t = self.backbone_t(imgs.flatten(0,2))
            fs_t = fs_t.reshape(bsz, t, *fs_t.shape[-3:])
            tar_t, refs_t = fs_t[:, -1], fs_t[:, :-1]

            _, target_att = non_local_attention(tar_t, refs_t, temprature=self.temperature_t, norm=self.norm)

        losses = {}
        losses['att_loss'] = self.loss(att_g, target_att)
        
        
        # for mast l1_loss
        if self.l1_loss:
            ref_gt = [self.prep(gt[:,ch]) for gt in images_lab_gt[:-1]]
            outputs = frame_transform(att, ref_gt, flatten=False)
            outputs = outputs[:,0].permute(0,2,1).reshape(bsz, -1, *fs1.shape[-2:])     
            losses['l1_loss'], _ = self.compute_lphoto(images_lab_gt, ch, outputs)

        return losses
    



@MODELS.register_module()
class Dist_Tracker_Channel_Att(Dist_Tracker_V2):
    
    def forward_train(self, imgs, images_lab=None):
        bsz, num_clips, t, c, h, w = imgs.shape

        images_lab_gt = [images_lab[:,0,i,:].clone() for i in range(t)]
        images_lab = [images_lab[:,0,i,:] for i in range(t)]
        _, ch = self.dropout2d_lab(images_lab)
        
        # forward to get feature

        fs1_, fs2_ = self.backbone(torch.stack(images_lab,1).flatten(0,1))
        fs1 = fs1_.reshape(bsz, t, *fs1_.shape[-3:])
        tar1, refs1 = fs1[:, -1], fs1[:, :-1]
        fs2 = fs2_.reshape(bsz, t, *fs2_.shape[-3:])
        tar2, refs2 = fs2[:, -1], fs2[:, :-1]
        
        _, att_g = non_local_attention(tar2, refs2, scaling=True)
        _, att = non_local_attention(tar1, refs1, scaling=True, mask=self.mask)
        sf = fs2_.mean(1)


        with torch.no_grad():
            self.backbone_t.eval()
            fs_t_ = self.backbone_t(imgs.flatten(0,2))
            fs_t = fs_t_.reshape(bsz, t, *fs_t_.shape[-3:])
            tar_t, refs_t = fs_t[:, -1], fs_t[:, :-1]
            _, target_att = non_local_attention(tar_t, refs_t)
            tf = fs_t_.mean(1)

        losses = {}
        losses['att_loss'] = self.loss(att_g, target_att)
        losses['feat_att_loss'] = self.loss_feat(sf, tf)
        
        
        # for mast l1_loss
        if self.l1_loss:
            ref_gt = [self.prep(gt[:,ch]) for gt in images_lab_gt[:-1]]
            outputs = frame_transform(att, ref_gt, flatten=False)
            outputs = outputs[:,0].permute(0,2,1).reshape(bsz, -1, *fs1.shape[-2:])     
            losses['l1_loss'], _ = self.compute_lphoto(images_lab_gt, ch, outputs)

        return losses
    
    
@MODELS.register_module()
class Dist_Tracker_Weighted(Dist_Tracker_V2):
    def __init__(self, thres=-1, wr=False, weight=20, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.T = thres
        self.weight = weight
        self.wr = wr
        
        
    def forward_train(self, imgs, images_lab=None):
        bsz, num_clips, t, c, h, w = imgs.shape

        images_lab_gt = [images_lab[:,0,i,:].clone() for i in range(t)]
        images_lab = [images_lab[:,0,i,:] for i in range(t)]
        _, ch = self.dropout2d_lab(images_lab)
        
        # forward to get feature

        fs1_, fs2_ = self.backbone(torch.stack(images_lab,1).flatten(0,1))
        fs1 = fs1_.reshape(bsz, t, *fs1_.shape[-3:])
        tar1, refs1 = fs1[:, -1], fs1[:, :-1]
        fs2 = fs2_.reshape(bsz, t, *fs2_.shape[-3:])
        tar2, refs2 = fs2[:, -1], fs2[:, :-1]
        
        _, att_g = non_local_attention(tar2, refs2, temprature=self.temperature, scaling=self.scaling, norm=self.norm)
        _, att = non_local_attention(tar1, refs1, scaling=True, mask=self.mask)


        with torch.no_grad():
            self.backbone_t.eval()
            fs_t_ = self.backbone_t(imgs.flatten(0,2))
            fs_t = fs_t_.reshape(bsz, t, *fs_t_.shape[-3:])
            tar_t, refs_t = fs_t[:, -1], fs_t[:, :-1]
            _, target_att = non_local_attention(tar_t, refs_t, temprature=self.temperature_t, norm=self.norm)

            
            if self.T == -1:
                tf = (tar_t.mean(1).flatten(-2)).softmax(-1)
                weight = tf.unsqueeze(-1).repeat(1, 1, target_att.shape[-1])
            else:
                tf = (tar_t.mean(1).flatten(-2))
                tf_sorted, _ = torch.sort(tf, dim=-1, descending=True)
                idx = int(tf.shape[-1] * self.T) - 1
                T = tf_sorted[:, idx:idx+1]
                M = ( tf > T).bool()
                weight = torch.masked_fill(tf, ~M, float('-inf')).softmax(-1)
                weight_rec = weight.reshape(bsz, fs2.shape[-1], -1) if self.wr else None
                weight = weight.unsqueeze(-1).repeat(1, 1, target_att.shape[-1])
            # x = tf[0].reshape(16,16).detach().cpu().numpy()
            # x = weight[0].reshape(16,16).detach().cpu().numpy()
            # img = tensor2img(imgs[0,-1,-1], norm_mode='mean-std')
                         
        losses = {}
        losses['att_loss'] = self.loss(att_g[:,0], target_att[:,0], weight=weight)
        
        # for mast l1_loss
        if self.l1_loss:
            ref_gt = [self.prep(gt[:,ch]) for gt in images_lab_gt[:-1]]
            outputs = frame_transform(att, ref_gt, flatten=False)
            outputs = outputs[:,0].permute(0,2,1).reshape(bsz, -1, *fs1.shape[-2:])     
            losses['l1_loss'], _ = self.compute_lphoto(images_lab_gt, ch, outputs, weight_rec=weight_rec)

        return losses
    
    def compute_lphoto(self, images_lab_gt, ch, outputs, weight_rec=None):
        b, c, h, w = images_lab_gt[0].size()

        tar_y = images_lab_gt[-1][:,ch]  # y4
        
        if weight_rec == None:
            outputs = F.interpolate(outputs, (h, w), mode='bilinear')
            loss = F.smooth_l1_loss(outputs * self.weight, tar_y * self.weight, reduction='mean')
        else:
            tar_y = self.prep(images_lab_gt[-1])[:,ch]
            loss = F.smooth_l1_loss(outputs * self.weight, tar_y * self.weight, reduction='none')
            loss = (loss * weight_rec.unsqueeze(1)).sum() / (weight_rec.sum() + 1e-12)

        err_maps = torch.abs(outputs - tar_y).sum(1).detach()

        return loss, err_maps
    

@MODELS.register_module()
class Dist_Tracker_Pyramiad(Dist_Tracker):
    
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
        losses['att1_loss'] = self.loss(att_l, target_att1)
        losses['att2_loss'] = self.loss(att_g, target_att2)
        
        
        # for mast l1_loss
        if self.l1_loss:
            ref_gt = [self.prep(gt[:,ch]) for gt in images_lab_gt[:-1]]
            outputs = frame_transform(att, ref_gt, flatten=False)
            outputs = outputs[:,0].permute(0,2,1).reshape(bsz, -1, *fs1.shape[-2:])     
            losses['l1_loss'], _ = self.compute_lphoto(images_lab_gt, ch, outputs)

        return losses


@MODELS.register_module()
class Dist_Tracker_Inter_Video(Dist_Tracker):
    
    
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
        
        _, att_g = non_local_attention(tar2, refs2, temprature=self.temperature, scaling=self.scaling)
        _, att = non_local_attention(tar1, refs1, scaling=True, mask=self.mask)
        
        # refs2_flatten = F.normalize(refs2[:,0].permute(0, 2, 3, 1).flatten(0,2), dim=-1)
        # tar2_flatten = F.normalize(tar2.permute(0, 2, 3, 1).flatten(0,2), dim=-1)
        tar2_flatten = tar2.permute(0, 2, 3, 1).flatten(0,2)
        refs2_flatten = refs2[:,0].permute(0, 2, 3, 1).flatten(0,2)
        
        att_inter = torch.einsum("xc, cy -> xy", [tar2_flatten, refs2_flatten.permute(1,0)]).softmax(-1)


        with torch.no_grad():
            self.backbone_t.eval()
            fs_t = self.backbone_t(imgs.flatten(0,2))
            fs_t = fs_t.reshape(bsz, t, *fs_t.shape[-3:])
            tar_t, refs_t = fs_t[:, -1], fs_t[:, :-1]
            _, target_att = non_local_attention(tar_t, refs_t, temprature=self.temperature)
            
            # refst_flatten = F.normalize(refs_t[:,0].permute(0, 2, 3, 1).flatten(0,2), dim=-1)
            # tart_flatten = F.normalize(tar_t.permute(0, 2, 3, 1).flatten(0,2), dim=-1)
            tart_flatten = tar_t.permute(0, 2, 3, 1).flatten(0,2)
            refst_flatten = refs_t[:,0].permute(0, 2, 3, 1).flatten(0,2)
            
            att_inter_target = torch.einsum("xc, cy -> xy", [tart_flatten, refst_flatten.permute(1,0)]).softmax(-1)

        losses = {}
        losses['att_loss'] = self.loss_weight['att_loss'] * self.loss(att_g, target_att)
        losses['att_inter_loss'] = self.loss_weight['att_inter_loss'] * self.loss(att_inter, att_inter_target)
        
        # for mast l1_loss
        if self.l1_loss:
            ref_gt = [self.prep(gt[:,ch]) for gt in images_lab_gt[:-1]]
            outputs = frame_transform(att, ref_gt, flatten=False)
            outputs = outputs[:,0].permute(0,2,1).reshape(bsz, -1, *fs1.shape[-2:])     
            losses['l1_loss'], _ = self.compute_lphoto(images_lab_gt, ch, outputs)

        return losses


@MODELS.register_module()
class Dist_Tracker_Pyramid_Raft(Dist_Tracker):
    
    def __init__(self, num_levels, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        self.num_levels = num_levels    
        # for i in range(self.num_levels):
        #     setattr(self, f'mask{i}', make_mask((self.feat_size, self.feat_size // 2 ** i), self.radius))
        
        
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
        
        atts = self.non_local_attention_pyramid(tar2, refs2, temprature=self.temperature, scaling=self.scaling, norm=self.norm)
        _, att = non_local_attention(tar1, refs1, scaling=True, mask=self.mask)


        with torch.no_grad():
            self.backbone_t.eval()
            fs_t = self.backbone_t(imgs.flatten(0,2))
            fs_t = fs_t.reshape(bsz, t, *fs_t.shape[-3:])
            tar_t, refs_t = fs_t[:, -1], fs_t[:, :-1]

            target_atts = self.non_local_attention_pyramid(tar_t, refs_t, temprature=self.temperature_t, norm=self.norm)

        losses = {}
        for idx, (att_p, target_att_p) in enumerate(zip(atts, target_atts)):
            losses[f'att_{idx}_loss'] = self.loss_weight[f'att_{idx}_loss'] * self.loss(att_p, target_att_p)
        
        
        # for mast l1_loss
        if self.l1_loss:
            ref_gt = [self.prep(gt[:,ch]) for gt in images_lab_gt[:-1]]
            outputs = frame_transform(att, ref_gt, flatten=False)
            outputs = outputs[:,0].permute(0,2,1).reshape(bsz, -1, *fs1.shape[-2:])     
            losses['l1_loss'], _ = self.compute_lphoto(images_lab_gt, ch, outputs)

        return losses
    
    def non_local_attention_pyramid(self, tar, refs, temprature=1.0, scaling=False, norm=False):
        
        if isinstance(refs, list):
            refs = torch.stack(refs, 1)

        tar = tar.flatten(2).permute(0, 2, 1)
        bsz, t, feat_dim, h_, w_ = refs.shape
        refs = refs.flatten(3).permute(0, 1, 3, 2)
        
        if norm:
            tar = F.normalize(tar, dim=-1)
            refs = F.normalize(refs, dim=-1)
            
        # calc correlation
        att = torch.einsum("bic,btjc -> btij", (tar, refs)) / temprature 
        att_pyramid = [att]
        
        att = att.reshape(-1, 1, h_, w_)
        for i in range(self.num_levels-1):
            att_ = F.avg_pool2d(att, 2, stride=2)
            att_ = att_.reshape(bsz, 1, h_*w_, -1)
            att_pyramid.append(att_)
        
        for idx, att in enumerate(att_pyramid):
            
            if scaling:
                # scaling
                att = att / torch.sqrt(torch.tensor(feat_dim).float()) 

            # att *= mask
            # mask = getattr(self, f'mask{idx}')
            # att.masked_fill_(~mask.bool(), float('-inf'))
            att = F.softmax(att, dim=-1)
            att_pyramid[idx] = att
        
        return att_pyramid


@MODELS.register_module()
class Dist_Tracker_EWC(Dist_Tracker_V2):
    def __init__(self, ewc, ewc_weight=1, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        self.regularizer = build_components(ewc)
        self.ewc_weight = ewc_weight
    
    
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
        
        _, att_g = non_local_attention(tar2, refs2, temprature=self.temperature, scaling=self.scaling, norm=self.norm)
        # _, att_g = non_local_attention(tar2, refs2)
        _, att = non_local_attention(tar1, refs1, scaling=True, mask=self.mask)


        with torch.no_grad():
            self.backbone_t.eval()
            fs_t = self.backbone_t(imgs.flatten(0,2))
            fs_t = fs_t.reshape(bsz, t, *fs_t.shape[-3:])
            tar_t, refs_t = fs_t[:, -1], fs_t[:, :-1]

            _, target_att = non_local_attention(tar_t, refs_t, temprature=self.temperature_t, norm=self.norm)

        losses = {}
        losses['att_loss'] = self.loss(att_g, target_att)
        
        
        # for mast l1_loss
        if self.l1_loss:
            ref_gt = [self.prep(gt[:,ch]) for gt in images_lab_gt[:-1]]
            outputs = frame_transform(att, ref_gt, flatten=False)
            outputs = outputs[:,0].permute(0,2,1).reshape(bsz, -1, *fs1.shape[-2:])     
            losses['l1_loss'], _ = self.compute_lphoto(images_lab_gt, ch, outputs)
        
        return losses
    
    def train_step(self, data_batch, optimizer, progress_ratio):
        
        # parser other loss
        losses = self(**data_batch, test_mode=False)
        loss, log_vars = self.parse_losses(losses)
        
        # optimizer
        for k,opz in optimizer.items():
            opz.zero_grad()
        loss.backward()
        for k,opz in optimizer.items():
            opz.step()
            
         # parser ewc loss
        losses = {}
        if distributed.is_initialized():
            if distributed.get_rank() == 0:
                self.regularizer.update(self.backbone)
        else:
            self.regularizer.update(self.backbone)
        losses['ewc_reg_loss'] = self.ewc_weight * self.regularizer.penalty(self.backbone)
        loss, log_vars_reg = self.parse_losses(losses)
        # optimizer
        for k,opz in optimizer.items():
            opz.zero_grad()
        loss.backward()
        for k,opz in optimizer.items():
            opz.step()
        for k,v in log_vars_reg.items():
            log_vars[k] = v
        
        if self.momentum is not -1:
            moment_update(self.backbone, self.backbone_t, self.momentum)

        log_vars.pop('loss')
        outputs = dict(
            log_vars=log_vars,
            num_samples=len(data_batch['imgs'])
        )

        return outputs


@MODELS.register_module()
class Dist_Tracker_Memory(BaseModel):
    
    def __init__(self, 
                 backbone,
                 backbone_t,
                 momentum,
                 temperature,
                 K, 
                 backbone_k,  
                 ce_loss, 
                 T, 
                 m=0.999,
                 temperature_t=1.0,
                 downsample_rate=8,
                 feat_size=32,
                 mask_radius=-1,
                 loss=None,
                 loss_weight=None,
                 loss_feat=None,
                 l1_loss=None,
                 scaling=True,
                 norm=False,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 dim_mlp=2048, 
                 dim_out=128, 
                 *args, 
                 **kwargs):
        super().__init__(*args, **kwargs)
        
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.l1_loss  = l1_loss
        self.downsample_rate = downsample_rate

        self.momentum = momentum
        self.feat_size = feat_size
        self.radius = mask_radius

        self.pretrained = pretrained
        self.temperature = temperature
        self.temperature_t = temperature_t
        self.scaling = scaling
        self.norm = norm
        self.loss_weight = loss_weight

        self.backbone = build_backbone(backbone)
        self.backbone_t = build_backbone(backbone_t)

        # loss
        self.loss = build_loss(loss)
        self.loss_feat = build_loss(loss)
        
        
        if mask_radius != -1:
            self.mask = make_mask(feat_size, mask_radius)
        else:
            self.mask = None
        
        self.K = K
        self.backbone_k = build_backbone(backbone_k) 
        self.dim_out = dim_out
        
        # memory for student
        self.register_buffer("queue", torch.randn(self.dim_out, K).cuda())
        self.queue = nn.functional.normalize(self.queue, dim=0)
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long).cuda())
        
        # memory for teacher
        self.register_buffer("queue_t", torch.randn(self.dim_out, K).cuda())
        self.queue_t = nn.functional.normalize(self.queue_t, dim=0)
        self.register_buffer("queue_ptrt", torch.zeros(1, dtype=torch.long).cuda())
        
        self.backbone.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), nn.Linear(dim_mlp, self.dim_out))
        self.backbone_k.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), nn.Linear(dim_mlp, self.dim_out))
       
        self.backbone_t.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), nn.Linear(dim_mlp, self.dim_out))

        self.ce_loss = build_loss(ce_loss)
        self.T = T
        self.m = m
        
        self.init_weights()
    
    def init_weights(self):
        
        # overall state_dict
        state_dict = torch.load(self.pretrained, map_location='cpu')['state_dict']
        enc_q = { k.replace('module.encoder_q.', ''):v for k,v in state_dict.items() if k.find('module.encoder_q.') != -1 }
        enc_q_fc = { k.replace('module.encoder_q.fc.', ''):v for k,v in state_dict.items() if k.find('module.encoder_q.fc.') != -1  }

        # backbone loading
        self.backbone.pretrained = enc_q
        self.backbone.init_weights()
        self.backbone.fc.load_state_dict(enc_q_fc, strict=True)
        # copy params to backbone_k
        for param_q, param_k in zip(self.backbone.parameters(), self.backbone_k.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient
        
        # backbone distilaition loading
        self.backbone_t.pretrained = enc_q
        self.backbone_t.init_weights()
        self.backbone_t.fc.load_state_dict(enc_q_fc, strict=True)

        
    def forward_train(self, imgs, images_lab=None):
        bsz, num_clips, t, c, h, w = imgs.shape

        images_lab_gt = [images_lab[:,0,i,:].clone() for i in range(t)]
        images_lab = [images_lab[:,0,i,:] for i in range(t)]
        _, ch = self.dropout2d_lab(images_lab)
        
        # forward to get feature
        fs1, _, fs2 = self.backbone(torch.stack(images_lab,1).flatten(0,1))
        fs1 = fs1.reshape(bsz, t, *fs1.shape[-3:])
        tar1, refs1 = fs1[:, -1], fs1[:, :-1]
        
        fs2 = fs2.reshape(bsz, t, -1)
        q = fs2[:, 0]
        q = nn.functional.normalize(q, dim=1)
        
        # forward to get feature k
        with torch.no_grad():
            self._momentum_update_key_encoder()
            
            # shuffle for making use of BN
            if distributed.is_initialized():
                im_k, idx_unshuffle = self._batch_shuffle_ddp(images_lab[-1].contiguous())
            else:
                im_k = images_lab[-1]

            _, k = self.backbone_k(im_k)  # keys: NxC
            k = nn.functional.normalize(k, dim=1)

            # undo shuffle
            if distributed.is_initialized():
                k = self._batch_unshuffle_ddp(k.contiguous(), idx_unshuffle)

            
            # forward for teacher network
            self.backbone_t.eval()
            _, q_t = self.backbone_t(imgs[:,0,0])
            _, k_t = self.backbone_t(imgs[:,0,1])
            q_t = nn.functional.normalize(q_t, dim=1)
            k_t = nn.functional.normalize(k_t, dim=1)
            
        
         # dequeue and enqueue
        self._dequeue_and_enqueue(k)
        self._dequeue_and_enqueue_t(k_t)
        
        # logit for backbone
        l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)
        l_neg = torch.einsum('nc,ck->nk', [q, self.queue.clone().detach()])
        logits = torch.cat([l_pos, l_neg], dim=1) / self.T
        
        # logit for backbone_t
        l_pos_t = torch.einsum('nc,nc->n', [q_t, k_t]).unsqueeze(-1)
        l_neg_t = torch.einsum('nc,ck->nk', [q_t, self.queue_t.clone().detach()])
        logits_t = torch.cat([l_pos_t, l_neg_t], dim=1) / self.T
        
        losses = {}
        # for mast l1_loss
        if self.l1_loss:
            _, att = non_local_attention(tar1, refs1, scaling=True, mask=self.mask)
            ref_gt = [self.prep(gt[:,ch]) for gt in images_lab_gt[:-1]]
            outputs = frame_transform(att, ref_gt, flatten=False)
            outputs = outputs[:,0].permute(0,2,1).reshape(bsz, -1, *fs1.shape[-2:])     
            losses['l1_loss'], _ = self.compute_lphoto(images_lab_gt, ch, outputs)

        # for logit dist loss
        losses['logit_dist_loss'] = self.ce_loss(logits, logits_t)
        
        return losses

    
    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        # gather keys before updating queue
        if distributed.is_initialized():
            keys = concat_all_gather(keys)

        batch_size = keys.shape[0]

        ptr = int(self.queue_ptr)
        assert self.K % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.queue[:, ptr:ptr + batch_size] = keys.T
        ptr = (ptr + batch_size) % self.K  # move pointer

        self.queue_ptr[0] = ptr
    
    @torch.no_grad()
    def _dequeue_and_enqueue_t(self, keys):
        # gather keys before updating queue
        if distributed.is_initialized():
            keys = concat_all_gather(keys)

        batch_size = keys.shape[0]

        ptr = int(self.queue_ptrt)
        assert self.K % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.queue_t[:, ptr:ptr + batch_size] = keys.T
        ptr = (ptr + batch_size) % self.K  # move pointer

        self.queue_ptrt[0] = ptr
        

    @torch.no_grad()
    def _batch_shuffle_ddp(self, x):
        """
        Batch shuffle, for making use of BatchNorm.
        *** Only support DistributedDataParallel (DDP) model. ***
        """
        # gather from all gpus
        batch_size_this = x.shape[0]
        x_gather = concat_all_gather(x)
        batch_size_all = x_gather.shape[0]

        num_gpus = batch_size_all // batch_size_this

        # random shuffle index
        idx_shuffle = torch.randperm(batch_size_all).cuda()

        # broadcast to all gpus
        torch.distributed.broadcast(idx_shuffle, src=0)

        # index for restoring
        idx_unshuffle = torch.argsort(idx_shuffle)

        # shuffled index for this gpu
        gpu_idx = torch.distributed.get_rank()
        idx_this = idx_shuffle.view(num_gpus, -1)[gpu_idx]

        return x_gather[idx_this], idx_unshuffle

    @torch.no_grad()
    def _batch_unshuffle_ddp(self, x, idx_unshuffle):
        """
        Undo batch shuffle.
        *** Only support DistributedDataParallel (DDP) model. ***
        """
        # gather from all gpus
        batch_size_this = x.shape[0]
        x_gather = concat_all_gather(x)
        batch_size_all = x_gather.shape[0]

        num_gpus = batch_size_all // batch_size_this

        # restored index for this gpu
        gpu_idx = torch.distributed.get_rank()
        idx_this = idx_unshuffle.view(num_gpus, -1)[gpu_idx]

        return x_gather[idx_this]


    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(self.backbone.parameters(), self.backbone_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)


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