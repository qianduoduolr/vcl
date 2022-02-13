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


@MODELS.register_module()
class RW_Tracker(BaseModel):
    def __init__(self, backbone, ce_loss, temperature=0.07, edgedrop_rate=0.1, train_cfg=None, test_cfg=None):
        """
        CRW (2021NIPS) with CNN model
        Args:
            temperature (float, optional): [description]. Defaults to 0.07.
            edgedrop_rate (float, optional): [description]. Defaults to 0.1.
        """
        super().__init__()
        self.backbone = build_backbone(backbone)
        self.fc = nn.Linear(512, 128)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self._xent_targets = dict()
        self.edgedrop_rate = edgedrop_rate
        self.dropout = nn.Dropout(p=self.edgedrop_rate, inplace=False)
        self.ce_loss = build_loss(ce_loss)
        self.temperature = temperature
        
    
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

    def forward_train_cycle(self, q):
        '''
        Input is B x T x N*C x H x W, where either
           N>1 -> list of patches of images
           N=1 -> list of images
        '''
        
        bsz, C, T, N = q.shape
        
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
            loss = self.ce_loss(logits, target.unsqueeze(1))
            acc = (torch.argmax(logits, dim=-1) == target).float().mean()
            # diags.update({f"{H} xent {name}": loss.detach(),
            #               f"{H} acc {name}": acc})
            xents += [loss]


        loss = sum(xents)/max(1, len(xents)-1)
        
        return loss, acc


    def forward_train(self, imgs, jitter_imgs=None):
    
        bsz, num_patch, c, t, h, w = imgs.shape
        
        # get node representations
        imgs = imgs.permute(0, 1, 3, 2, 4, 5).flatten(0,2)
        feats = self.backbone(imgs)
        feats = self.pool(feats)[:,:,0,0]
        feats = F.normalize(self.fc(feats), dim=-1)
        feats = feats.reshape(bsz, num_patch, t, -1).permute(0, 3, 2, 1)

        losses = {}

        losses['cycle_loss'], acc = self.forward_train_cycle(feats)
        
        return losses