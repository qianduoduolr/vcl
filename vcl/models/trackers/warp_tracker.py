# Copyright (c) OpenMMLab. All rights reserved.
import numbers
from os import stat_result
import os.path as osp
from collections import *
from tkinter.messagebox import NO

import mmcv
from mmcv.runner import auto_fp16, load_state_dict, load_checkpoint

from ..base import BaseModel
from ..builder import build_backbone, build_loss, build_components
from ..registry import MODELS
from vcl.utils.helpers import *
from vcl.utils import *
from vcl.models.common import *


import torch.nn as nn
import torch
import torch.nn.functional as F
import math


@MODELS.register_module()
class Warp_Tracker(BaseModel):

    def __init__(self,
                 backbone,
                 head,
                 temperature=0.05,
                 grid_size=8,
                 l1_loss=None,
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

        super(Warp_Tracker, self).__init__()

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.l1_loss  = l1_loss
        self.grid_size = grid_size
        self.temperature = temperature

        self.pretrained = pretrained

        logger = get_root_logger()

        self.backbone = build_backbone(backbone)
        self.head = build_components(head)
        self.eye = None

        
        self.init_weights()
    
    def init_weights(self):
        
        self.backbone.init_weights()        
        if self.pretrained is not None:
            _ = load_checkpoint(self, self.pretrained, map_location='cpu')


    def _align(self, x, t):
        tf = F.affine_grid(t, size=x.size(), align_corners=False)
        return F.grid_sample(x, tf, align_corners=False, mode="nearest")

    def _key_val(self, ctr, q):
        """
        Args:
            ctr: [N,K]
            q: [BHW,K]
        Returns:
            val: [BHW,N]
        """

        # [BHW,K] x [N,K].t -> [BHWxN]
        vals = torch.mm(q, ctr.t()) # [BHW,N]

        # normalising attention
        return vals / self.temperature

    def _sample_index(self, x, T, N):
        """Sample indices of the anchors

        Args:
            x: [BT,K,H,W]
        Returns:
            index: [B,N*N,K]
        """

        BT,K,H,W = x.shape
        B = x.view(-1,T,K,H*W).shape[0]

        # sample indices from a uniform grid
        xs, ys = W // N, H // N
        x_sample = torch.arange(0, W, xs).view(1, 1, N)
        y_sample = torch.arange(0, H, ys).view(1, N, 1)

        # Random offsets
        # [B x 1 x N]
        x_sample = x_sample + torch.randint(0, xs, (B, 1, 1))
        # [B x N x 1]
        y_sample = y_sample + torch.randint(0, ys, (B, 1, 1))

        # batch index
        # [B x N x N]
        hw_index = torch.LongTensor(x_sample + y_sample * W)

        return hw_index

    def _sample_from(self, x, index, T, N):
        """Gather the features based on the index

        Args:
            x: [BT,K,H,W]
            index: [B,N,N] defines the indices of NxN grid for a single
                           frame in each of B video clips
        Returns:
            anchors: [BNN,K] sampled features given by index from x
        """

        BT,K,H,W = x.shape
        x = x.view(-1,T,K,H*W)
        B = x.shape[0]

        # > [B,T,K,HW] > [B,T,HW,K] > [B,THW,K]
        x = x.permute([0,1,3,2]).reshape(B,-1,K)

        # every video clip will have the same samples
        # on the grid
        # [B x N x N] -> [B x N*N x 1] -> [B x N*N x K]
        index = index.view(B,-1,1).expand(-1,-1,K)

        # selecting from the uniform grid
        y = x.gather(1, index.to(x.device))

        # [BNN,K]
        return y.flatten(0,1)

    def _mark_from(self, x, index, T, N, fill_value=0):
        """This is analogous to _sample_from except that
        here we simply "mark" the sampled positions in the tensor
        Used for visualisation only.
        Since it is a binary mask, K == 1

        Args:
            x: [BT,1,H,W] binary mask
            index: [B,N,N] defines the indices of NxN grid for a single
                           frame in each of B video clips
        Returns:
            y: [BT,1,H,W] marked sample positions
        """

        BT,K,H,W = x.shape
        assert K == 1, "Expected binary mask"
        x = x.view(-1,T,K,H*W)
        B = x.shape[0]

        # > [B,T,K,HW] > [B,T,HW,K] > [B,THW,K]
        x = x.permute([0,1,3,2]).reshape(B,-1,K)

        # every video clip will have the same samples
        # on the grid
        # [B x N x N] -> [B x N*N x 1] -> [B x N*N x K]
        index = index.view(B,-1,1).expand(-1,-1,K)

        # selecting from the uniform grid
        # [B x T*H*W x K]
        y = x.scatter(1, index.to(x.device), fill_value)

        # [B x T*H*W x K] -> [BT x K x H x W]
        return y.view(-1,H*W,K).permute([0,2,1]).view(-1,K,H,W)

    def _cluster_grid(self, k1, k2, aff1, aff2, T, index=None):
        """ Selecting clusters within a sequence
        Args:
            k1: [BT,K,H,W]
            k2: [BT,K,H,W]
        """

        BT,K,H,W = k1.shape
        assert BT % T == 0, "Batch not divisible by sequence length"
        B = BT // T

        # N = [G x G]
        N = self.grid_size ** 2

        # [BT,K,H,W] -> [BTHW,K]
        flatten = lambda x: x.flatten(2,3).permute([0,2,1]).flatten(0,1)

        # [BTHW,BN] -> [BT,BN,H,W]
        def unflatten(x, aff=None):
            x = x.view(BT,H*W,-1).permute([0,2,1]).view(BT,-1,H,W)
            if aff is None:
                return x
            return self._align(x, aff)

        index = self._sample_index(k1, T, N = self.grid_size)
        query1 = self._sample_from(k1, index, T, N = self.grid_size)

        """Computing the distances and pseudo labels"""

        # [BTHW,K]
        k1_ = flatten(k1)
        k2_ = flatten(k2)

        # [BTHW,BN] -> [BTHW,BN] -> [BT,BN,H,W]
        vals_soft = unflatten(self._key_val(query1, k1_), aff1)
        vals_pseudo = unflatten(self._key_val(query1, k2_), aff2)

        # [BT,BN,H,W]
        probs_pseudo = self._pseudo_mask(vals_pseudo, T)
        probs_pseudo2 = self._pseudo_mask(vals_soft, T)

        pseudo = probs_pseudo.argmax(1)
        pseudo2 = probs_pseudo2.argmax(1)

        # mask
        def grid_mask():
            grid_mask = torch.ones(BT,1,H,W).to(pseudo.device)
            return self._mark_from(grid_mask, index, T, N = self.grid_size)

        return vals_soft, pseudo, index, [vals_pseudo, pseudo2, grid_mask]

    # sampling affinity
    def _aff_sample(self, k1, k2, T):
        BT,K,h,w = k1.size()
        B = BT // T
        hw = h*w

        def gen(query):
            grid_mask = torch.ones(B,1,hw).to(k1.device)
            # generating random indices
            indices = torch.randint(0, hw, (B,1,1)).to(k1.device)
            grid_mask.scatter_(2, indices, 0)

            # [B,K,H,W] -> [B,K,1]
            query_ = query[::T].view(B,K,-1).gather(2, indices.expand(-1,K,-1))

            def aff(keys):
                k = keys.view(B,T,K,-1)
                # [B,T,K,HW] x [B,1,K,HW] -> [B,T,HW]
                aff = (k * query_[:,None,:,:]).sum(2)
                return (aff + 1) / 2


            aff1 = aff(k1)
            aff2 = aff(k2)

            return grid_mask.view(B,h,w), aff1.view(BT,h,w), aff2.view(BT,h,w)

        grid_mask1, aff1_1, aff1_2 = gen(k1)
        grid_mask2, aff2_1, aff2_2 = gen(k2)

        return grid_mask1, aff1_1, aff1_2, \
                grid_mask2, aff2_1, aff2_2

    def _pseudo_mask(self, logits, T):
        BT,K,h,w = logits.shape
        assert BT % T == 0, "Batch not divisible by sequence length"
        B = BT // T

        # N = [G x G]
        N = self.grid_size ** 2

        # generating a pseudo label
        # first we need to mask out the affinities across the batch
        if self.eye is None or self.eye.shape[0] != B*T \
                            or self.eye.shape[1] != B*N:
            eye = torch.eye(B)[:,:,None].expand(-1,-1,N).reshape(B,-1)
            eye = eye.unsqueeze(1).expand(-1,T,-1).reshape(B*T, B*N, 1, 1)
            self.eye = eye.to(logits.device)

        probs = F.softmax(logits, 1)
        return probs * self.eye

    def _ref_loss(self, x, y, N = 4):
        B,_,h,w = x.shape

        index = self._sample_index(x, T=1, N=N)
        x1 = self._sample_from(x, index, T=1, N=N)
        y1 = self._sample_from(y, index, T=1, N=N)
        logits = torch.mm(x1, y1.t()) / self.temperature

        labels = torch.arange(logits.size(1)).to(logits.device)
        return F.cross_entropy(logits, labels)

    def _ce_loss(self, x, pseudo_map, T, eps=1e-5):
        error_map = F.cross_entropy(x, pseudo_map, reduction="none", ignore_index=-1)

        BT,h,w = error_map.shape
        errors = error_map.view(-1,T,h,w)
        error_ref, error_t = errors[:,0], errors[:,1:]

        return error_ref.mean(), error_t.mean(), error_map


    def fetch_first(self, x1, x2, T):
        assert x1.shape[1:] == x2.shape[1:]
        c,h,w = x1.shape[1:]

        x1 = x1.view(-1,T+1,c,h,w)
        x2 = x2.view(-1,T-1,c,h,w)

        x2 = torch.cat((x1[:,-1:], x2), 1)
        x1 = x1[:,:-1]

        return x1.flatten(0,1), x2.flatten(0,1)


    def forward_train(self, imgs, imgs_reg, affine_imgs, affine_imgs_reg):
        
        bsz, _, T, C, H, W = imgs.shape
        
        imgs = torch.cat((imgs[:,0], imgs_reg[:,0,::T]), 1)
        imgs = imgs.flatten(0,1)
        imgs_reg = imgs_reg[:,0,1:].flatten(0,1)

        
        # swap affine
        temp = affine_imgs
        affine_imgs = affine_imgs_reg.flatten(0,1)
        affine_imgs_reg = temp.flatten(0,1)
        
        # embedding for self-supervised learning
        res3, res4 = self.backbone(imgs)
        key1 = self.head(res4)
        
        with torch.no_grad():
            res3_reg, res4_reg = self.backbone(imgs_reg)
            key2 = self.head(res4_reg)
        
        # fetching the first frame from the second view
        key1, key2 = self.fetch_first(key1, key2, T)
        vals, pseudo, index, dbg_info = self._cluster_grid(key1, key2, affine_imgs, affine_imgs_reg, T)
        
        als_pseudo, pseudo2, grid_mask = dbg_info

        key1_aligned = self._align(key1, affine_imgs)
        key2_aligned = self._align(key2, affine_imgs_reg)
        
        losses = {}
        outs = {}

        n_ref = T -1
        losses["cross_key_loss"] = 0.1 * self._ref_loss(key1_aligned[::T], key2_aligned[::T], N = n_ref)

        # losses
        _, losses["temp_loss"], outs["error_map"] = self._ce_loss(vals, pseudo, T)

        
        return losses