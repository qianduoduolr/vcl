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
from vcl.models.components import *


@MODELS.register_module()
class Memory_Tracker_Custom_MoCo(Memory_Tracker_Custom):
    def __init__(self, backbone_m, num_k=32, T=0.07, m=0.999, dim=128, K=65536, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.backbone_m = build_backbone(backbone_m)
        self.head = MlpHead(self.backbone.feat_dim, dim)
        self.head_m = MlpHead(self.backbone.feat_dim, dim)
        self.m = m
        self.T = T
        self.num_k = num_k
        self.K = K

        for param_q, param_k in zip(self.backbone.parameters(), self.backbone_m.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient

        for param_q, param_k in zip(self.head.parameters(), self.head_m.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient

        # create the queue
        self.register_buffer("queue", torch.randn(dim, K))
        self.queue = nn.functional.normalize(self.queue, dim=0).cuda()

        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long).cuda())

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder
        """
        moment_update(self.backbone, self.backbone_m, self.m)
        moment_update(self.head, self.head_m, self.m)
    

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        if distributed.is_initialized():
            # gather keys before updating queue
            keys = concat_all_gather(keys)
        else:
            pass

        batch_size = keys.shape[0]

        ptr = int(self.queue_ptr)
        assert self.K % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.queue[:, ptr:ptr + batch_size] = keys.T
        ptr = (ptr + batch_size) % self.K  # move pointer

        self.queue_ptr[0] = ptr

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
        

    def forward_train(self, images_lab, imgs=None):

        bsz, _, n, c, h, w = images_lab.shape
        
        # images_lab_gt = [images_lab[:,0,i].clone() for i in range(n)]
        images_lab = [images_lab[:,0,i] for i in range(n)]
        # _, ch = self.dropout2d_lab(images_lab)
        
        # forward to get feature for backbone
        fs = self.head(self.backbone(torch.stack(images_lab,1).flatten(0,1)))
        fs = nn.functional.normalize(fs, dim=1)
        fs = fs.reshape(bsz, n, *fs.shape[-3:])
        tar, refs = fs[:, -1], fs[:, :-1]
        
        # get correlation attention map      
        _, att = non_local_attention(tar, refs, mask=self.mask, scaling=self.scaling, per_ref=self.per_ref, temprature=self.temperature) 


        with torch.no_grad():
            self._momentum_update_key_encoder()

            x = torch.stack(images_lab,1).flatten(0,1)
            if distributed.is_initialized():
                x, idx_unsh = self._batch_shuffle_ddp(x)
            x = self.head_m(self.backbone_m(x))
            x = nn.functional.normalize(x, dim=1)

            if distributed.is_initialized():
                x = self._batch_unshuffle_ddp(x, idx_unsh)

            fs_m = x.reshape(bsz, n, *x.shape[-3:])
            tar_m, refs_m = fs_m[:, -1], fs_m[:, :-1]
        
        losses = {}
        
        # for feature transformation
        refs_m = refs_m.flatten(-2).permute(0, 1, 3, 2)
        idx_sampled_k = torch.randint(0, self.feat_size[-1] ** 2, (self.num_k,))
        sampled_k = refs_m[:, :, idx_sampled_k].flatten(0,-2).cuda()
        tar_r = frame_transform(att, refs_m, flatten=False, per_ref=self.per_ref)
        
        tar = tar.flatten(-2).permute(0,2,1).unsqueeze(1).repeat(1,n-1,1,1).flatten(0,-2)
        tar_r = tar_r.flatten(0,-2)

        # positive logits: Nx1
        l_pos = torch.einsum('nc,nc->n', [tar, tar_r]).unsqueeze(-1)
        # negative logits: NxK
        l_neg = torch.einsum('nc,ck->nk', [tar, self.queue.clone().detach()])

        # logits: Nx(1+K)
        logits = torch.cat([l_pos, l_neg], dim=1)

        # apply temperature
        logits /= self.T

        # labels: positive key indicators
        labels = torch.zeros((logits.shape[0], 1), dtype=torch.long).cuda()
        
        # dequeue and enqueue
        self._dequeue_and_enqueue(sampled_k)

        # loss function
        losses['cts_rec_loss'] = build_loss(dict(type='Ce_Loss'))(logits, labels)

        vis_results = dict(err=None, imgs=imgs[0,0])

        return losses, vis_results