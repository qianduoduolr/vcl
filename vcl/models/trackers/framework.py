import numbers
from os import stat_result
import os.path as osp
from collections import *
from tkinter.messagebox import NO

import mmcv
from mmcv.runner import auto_fp16, load_state_dict, load_checkpoint

from ..base import BaseModel
from ..builder import build_backbone, build_components, build_loss, build_model
from ..registry import MODELS
from vcl.utils.helpers import *
from vcl.utils import *
from vcl.models.common import *


import torch.nn as nn
import torch
import torch.nn.functional as F

@MODELS.register_module()
class Framework_V2(BaseModel):

    def __init__(self,
                 backbone,
                 backbone_t,
                 momentum,
                 temperature,
                 head=None,
                 pool_type='mean',
                 weight=20,
                 num_stage=2,
                 feat_size=[64, 32],
                 radius=[12, 6],
                 downsample_rate=[4, 8],
                 temperature_t=1.0,
                 T=0.7,
                 loss=None,
                 loss_weight=None,
                 scaling=True,
                 norm=False,
                 detach=False,
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
        from mmcv.ops import Correlation
        
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
        self.T = T
    
        self.detach = detach
        self.loss_weight = loss_weight
        self.logger = get_root_logger()

        # build backbone
        self.backbone = build_backbone(backbone)
        self.backbone_t = build_backbone(backbone_t) if backbone_t != None else None
        self.head = build_components(head) if head != None else None

        # loss
        self.loss = build_loss(loss) if loss != None else None
        self.mask = [ make_mask(feat_size[i], radius[i]) for i in range(len(radius))]   
        self.corr = [Correlation(max_displacement=R) for R in radius ]
        
        self.weight = weight
        self.init_weights()
    
    def init_weights(self):
        
        if self.pretrained is not None:
            _ = load_checkpoint(self, self.pretrained, map_location='cpu')
        else:
            pass
    
    def forward_train(self, imgs, images_lab=None):
            
        bsz, num_clips, t, c, h, w = imgs.shape
        
        images_lab_gt = [images_lab[:,0,i,:].clone() for i in range(t)]
        images_lab = [images_lab[:,0,i,:] for i in range(t)]
        _, ch = self.dropout2d_lab(images_lab)
        
        # forward to get feature
        fs = self.backbone(torch.stack(images_lab,1).flatten(0,1))
        if isinstance(fs, tuple): fs = tuple(fs)
        else: fs = [fs]
        # if isinstance(fs, tuple):
        #     fs = [f.reshape(bsz, t, *f.shape[-3:]) for f in fs]
        # else:
        #     fs = [fs.reshape(bsz, t, *fs.shape[-3:])]
        
        # tar_pyramid, refs_pyramid = [f[:, -1] for f in fs], [ f[:, :-1] for f in fs]
        
        losses = {}
        
        atts = []
        corrs = []
        # for idx, (tar, refs) in enumerate(zip(tar_pyramid, refs_pyramid)):
        for idx, f in enumerate(fs):
            if self.head is not None and idx == 0:
                f = self.head(f)
            f = f.reshape(bsz, t, *f.shape[-3:])
            
            # get correlation for distillation
            if idx == len(fs) - 1 and self.backbone_t != None:
                _, att_g = non_local_attention(f[:,-1], f[:,:-1], scaling=self.scaling)
                break
                        
            # get correlation attention map            
            _, att = non_local_attention(f[:,-1], f[:,:-1], mask=self.mask[idx], scaling=True)      
                  
            # for mast l1_loss
            ref_gt = [self.prep(gt[:,ch], downsample_rate=self.downsample_rate[idx]) for gt in images_lab_gt[:-1]]
            outputs = frame_transform(att, ref_gt, flatten=False)
            outputs = outputs[:,0].permute(0,2,1).reshape(bsz, -1, self.feat_size[idx], self.feat_size[idx])     
            losses[f'stage{idx}_l1_loss'] = self.compute_lphoto(images_lab_gt, ch, outputs)[0] * self.loss_weight[f'stage{idx}_l1_loss']
            
            # save corr and att
            corr = self.corr[idx](f[:,-1], f[:,0])
            corrs.append(corr)
            atts.append(att)
            
        # get weight base on correlation entropy
        if self.T != -1:
            corr_feat = corrs[-1].reshape(bsz, -1, self.feat_size[-1], self.feat_size[-1])
            corr = corr_feat.softmax(1)
            corr_en = (-torch.log(corr+1e-12)).sum(1).flatten(-2)
            corr_sorted, _ = torch.sort(corr_en, dim=-1, descending=True)
            idx = int(corr_en.shape[-1] * self.T) - 1
            T = corr_sorted[:, idx:idx+1]
            sparse_mask = (corr_en > T).reshape(bsz, 1, *corr_feat.shape[-2:]).float().detach()
            weight = sparse_mask.flatten(-2).permute(0,2,1).repeat(1, 1, atts[-1].shape[-1])
        else:
            weight = None
        
        if len(corrs) > 1:
            # for layer distillation loss
            if self.pool_type == 'mean':
                att_ = atts[0].reshape(bsz, -1, *fs[0].shape[-2:])
                att_ = F.avg_pool2d(att_, 2, stride=2).flatten(-2).permute(0,2,1).reshape(bsz, -1, *fs[0].shape[-2:])
                target = F.avg_pool2d(att_, 2, stride=2).flatten(-2).permute(0,2,1)
            elif self.pool_type == 'max':
                att_ = atts[0].reshape(bsz, -1, *fs[0].shape[-2:])
                att_ = F.max_pool2d(att_, 2, stride=2).flatten(-2).permute(0,2,1).reshape(bsz, -1, *fs[0].shape[-2:])
                target = F.max_pool2d(att_, 2, stride=2).flatten(-2).permute(0,2,1)
            
            if not self.detach:
                losses['layer_dist_loss'] = self.loss_weight['layer_dist_loss'] * self.loss(atts[-1][:,0], target, weight=weight)
            else:
                losses['layer_dist_loss'] = self.loss_weight['layer_dist_loss'] * self.loss(atts[-1][:,0], target.detach(), weight=weight)
            
        # for correlation distillation loss
        if self.backbone_t is not None:
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