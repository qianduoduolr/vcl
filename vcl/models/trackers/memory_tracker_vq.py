from builtins import isinstance, list
import enum
import os.path as osp
from collections import *
from pickle import NONE
from re import A
from turtle import forward

import mmcv
import tempfile
from mmcv.runner import auto_fp16, load_checkpoint
from dall_e  import map_pixels, unmap_pixels, load_model
from torch import bilinear, unsqueeze
from mmcv.runner import CheckpointLoader
from mmcv.ops import Correlation


from vcl.models.common.correlation import *
from vcl.models.common.hoglayer import *
from vcl.models.losses.losses import l1_loss

from ..base import BaseModel
from ..builder import build_backbone, build_components, build_loss
from ..registry import MODELS
from vcl.utils import *

import torch.nn as nn
import torch.nn.functional as F
from .modules import *
from .memory_tracker import *


@MODELS.register_module()
class Memory_Tracker_Custom_Vq(Memory_Tracker_Custom):
    def __init__(self, vq, head=None, backbone_vq=None, backbone_t=None, mode='vq', temperature=1.0, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.vq = build_components(vq)
        self.vq_enc = self.vq.encode
        self.vq_emb = self.vq.quantize.embed.cuda()
        self.temperature = temperature
        self.mode = mode

        self.backbone_t = build_backbone(backbone_t) if backbone_t is not None else None
        self.head = build_components(head) if head is not None else None

    def forward_train_corrspondence(self, images_lab, imgs=None):
        
        bsz, _, n, c, h, w = images_lab.shape
        
        images_lab_gt = [images_lab[:,0,i].clone() for i in range(n)]
        images_lab = [images_lab[:,0,i] for i in range(n)]
        _, ch = self.dropout2d_lab(images_lab)
        
        # forward to get feature
        fs = self.backbone(torch.stack(images_lab,1).flatten(0,1))
        fs = fs.reshape(bsz, n, *fs.shape[-3:])
        tar, refs = fs[:, -1], fs[:, :-1]
        
        # get correlation attention map      
        _, att = non_local_attention(tar, refs, mask=self.mask[0], scaling=self.scaling, per_ref=self.per_ref, temprature=self.temperature)    

        if self.forward_backward_t != -1:
            att_ = non_local_attention(tar, refs, att_only=True, norm=True)    
            m = self.forward_backward_check(att_)
        else:
            m = None
        
        losses = {}
        if self.conc_loss !=  None:
            att_cos = non_local_attention(tar, refs, att_only=True)
            losses['conc_loss'] = self.conc_loss(att_cos)
        
        # for mast l1_loss
        outputs = self.frame_reconstruction(images_lab_gt, att, ch, feat_size=self.feat_size[0], downsample_rate=self.downsample_rate[0]) 
        losses['l1_loss'] = self.loss_weight['l1_loss'] * self.compute_lphoto(images_lab_gt, ch, outputs, upsample=self.upsample, mask=m)[0]
        
        # for vq reconstruction and vq update
        quants, inds, diff = self.vq_enc(fs.flatten(0,1))
        quants = quants.reshape(bsz, n, *quants.shape[-3:])
        inds = inds.reshape(bsz, n, *inds.shape[-2:])
        
        if self.per_ref:
            ind = inds[:, -1].unsqueeze(1).repeat(1, n-1, 1, 1).reshape(-1, 1).long().detach()
        else:
            ind = inds[:, -1].reshape(-1, 1).long().detach()
            
        quant_refs = quants[:, :-1].flatten(3).permute(0, 1, 3, 2).detach()
      
        out = frame_transform(att, quant_refs, per_ref=self.per_ref)
        out = F.normalize(out, dim=-1)
        vq_emb = F.normalize(self.vq_emb, dim=0)
        predict = torch.mm(out, vq_emb) / self.temperature
                    
        losses['vqr_ce_loss'] = self.loss_weight['vqr_ce_loss'] * build_loss(dict(type='Ce_Loss'))(predict, ind)

        losses['vq_diff_loss'] = self.loss_weight['vq_diff_loss'] * diff

        vis_results = dict(err=None, imgs=imgs[0,0])

        return losses, vis_results

    def forward_train_vq(self, images_lab, imgs=None):

        bsz, _, n, c, h, w = images_lab.shape
        
        images_lab_gt = [images_lab[:,0,i].clone() for i in range(n)]
        images = [imgs[:,0,i].clone() for i in range(n)]
      
        
        ############ VQ Correlation Calculation ###############
        with torch.no_grad():
            # forward to get feature
            fs = self.backbone(torch.stack(images,1).flatten(0,1))
            fs = fs.reshape(bsz, n, *fs.shape[-3:])
            tar, refs = fs[:, -1], fs[:, :-1]
            
            # get correlation attention map      
            _, att = non_local_attention(tar, refs, mask=None, scaling=self.scaling, per_ref=self.per_ref, temprature=self.temperature)    

        ############ VQ Correlation Calculation After Head ################
        fs_vq = self.head(fs.flatten(0,1))
        # for vq reconstruction and vq update
        fs_quants, inds, diff = self.vq_enc(fs_vq)
        fs_quants = fs_quants.reshape(bsz, n, *fs_quants.shape[-3:])
        tar_quants, refs_quants = fs_quants[:, -1], fs_quants[:, :-1]
        # get correlation attention map      
        _, att_vq_g = non_local_attention(tar_quants, refs_quants, mask=None, scaling=self.scaling, per_ref=self.per_ref, temprature=self.temperature) 

        _, att_vq_l = non_local_attention(tar_quants, refs_quants, mask=self.mask[0], scaling=self.scaling, per_ref=self.per_ref, temprature=self.temperature)

        ############ Correlation Calculation Using Temporal Model ###############
        with torch.no_grad():
            # forward to get feature
            fs = self.backbone_t(torch.stack(images_lab_gt,1).flatten(0,1))
            fs = fs.reshape(bsz, n, *fs.shape[-3:])
            tar, refs = fs[:, -1], fs[:, :-1]
            
            # get correlation attention map      
            _, att_t = non_local_attention(tar, refs, mask=self.mask[0], scaling=self.scaling, per_ref=self.per_ref, temprature=self.temperature)    

        ############# Self-Distillation ####################
        losses = {}
        loss_func = build_loss(dict(type='MSELoss'))
        losses['t_loss'] = self.loss_weight['t_loss'] * loss_func(att_vq_l, att_t.detach())
        losses['s_loss'] = self.loss_weight['s_loss'] * loss_func(att_vq_g, att.detach())
        losses['diff_loss'] = self.loss_weight['diff_loss'] * diff


        vis_results = dict(err=None, imgs=imgs[0,0])

        return losses, vis_results



    def forward(self, test_mode=False, **kwargs):
        """Forward function for vq model.

        Args:
            imgs (Tensor): Input image(s).
            labels (Tensor): Ground-truth label(s).
            test_mode (bool): Whether in test mode.
            kwargs (dict): Other arguments.

        Returns:
            Tensor: Forward results.
        """

        if test_mode:
            return self.forward_test(**kwargs)

        if self.mode == 'vq':
            return self.forward_train_vq(**kwargs)
        else:
            return self.forward_train_corrspondence(**kwargs)
