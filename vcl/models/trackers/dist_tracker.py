# Copyright (c) OpenMMLab. All rights reserved.
import numbers
from os import stat_result
import os.path as osp
from collections import *
from tkinter.messagebox import NO

import mmcv
from mmcv.runner import auto_fp16, load_state_dict, load_checkpoint

from ..base import BaseModel
from ..builder import build_backbone, build_loss
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
    def __init__(self, thres=-1, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.T = thres
    
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


        with torch.no_grad():
            self.backbone_t.eval()
            fs_t_ = self.backbone_t(imgs.flatten(0,2))
            fs_t = fs_t_.reshape(bsz, t, *fs_t_.shape[-3:])
            tar_t, refs_t = fs_t[:, -1], fs_t[:, :-1]
            _, target_att = non_local_attention(tar_t, refs_t)
            
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
            losses['l1_loss'], _ = self.compute_lphoto(images_lab_gt, ch, outputs)

        return losses
    

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

            _, target_att1 = non_local_attention(tar_t1, refs_t1)
            _, target_att2 = non_local_attention(tar_t2, refs_t2)
            

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
class Dist_Tracker_Memory(Dist_Tracker):
    
    def __init__(self, K, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        self.K = K
        
        # memory for student
        self.register_buffer("queue", torch.randn(self.backbone.feat_dim, K))
        self.queue = nn.functional.normalize(self.queue, dim=0)
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))
        
        # memory for teacher
        self.register_buffer("queue_t", torch.randn(self.backbone_t.feat_dim, K))
        self.queue = nn.functional.normalize(self.queue_t, dim=0)
        self.register_buffer("queue_ptrt", torch.zeros(1, dtype=torch.long))
    
    
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


        with torch.no_grad():
            self.backbone_t.eval()
            fs_t = self.backbone_t(imgs.flatten(0,2))
            fs_t = fs_t.reshape(bsz, t, *fs_t.shape[-3:])
            tar_t, refs_t = fs_t[:, -1], fs_t[:, :-1]

            _, target_att = non_local_attention(tar_t, refs_t, temprature=self.temperature)

        losses = {}
        losses['att_loss'] = self.loss(att_g, target_att)
        
        
        # for mast l1_loss
        if self.l1_loss:
            ref_gt = [self.prep(gt[:,ch]) for gt in images_lab_gt[:-1]]
            outputs = frame_transform(att, ref_gt, flatten=False)
            outputs = outputs[:,0].permute(0,2,1).reshape(bsz, -1, *fs1.shape[-2:])     
            losses['l1_loss'], _ = self.compute_lphoto(images_lab_gt, ch, outputs)

        return losses

    
    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        # gather keys before updating queue
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
        keys = concat_all_gather(keys)

        batch_size = keys.shape[0]

        ptr = int(self.queue_ptrt)
        assert self.K % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.queue_t[:, ptr:ptr + batch_size] = keys.T
        ptr = (ptr + batch_size) % self.K  # move pointer

        self.queue_ptrt[0] = ptr
        


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

    