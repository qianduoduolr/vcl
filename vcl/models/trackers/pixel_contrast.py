import numbers
import os.path as osp
from collections import *

import mmcv
from mmcv.runner import auto_fp16

from vcl.utils.helpers import *
from vcl.core.evaluation.metrics import JFM
import numbers
import os.path as osp
from collections import *


from ..base import BaseModel
from ..builder import build_backbone, build_loss, build_components
from ..registry import MODELS

import math
import torch

@MODELS.register_module()
class PixelContrast(BaseModel):
    def __init__(self, backbone,
                       head,
                       nce_loss,
                       memory_bank,
                       loss_lambda,
                       nce_t=0.1,
                       momentum=0.999,
                       train_cfg=None,
                       test_cfg=None,
                       pretrained=None):
        """[summary]

        Args:
            self ([type]): [description]
            contrast_loss ([type]): [description]
            train_cfg ([type], optional): [description]. Defaults to None.
            test_cfg ([type], optional): [description]. Defaults to None.
            pretrained ([type], optional): [description]. Defaults to None.
        """
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        self.encoder_q = nn.Sequential(
                                        bulid_backbone(backbone),
                                        bulid_component(head),
                                        )

        self.encoder_k = nn.Sequential(
                                        bulid_backbone(backbone),
                                        bulid_component(head),
                                        )

        self.nce_loss = bulid_loss(nce_loss)
        
        # nce config
        self.nce_t = nce_t
        self.nce_k = memory_bank['nce_k']
        self.nce_dim = memory_bank['nce_dim']
        self.nce_k_pixel = memory_bank['nce_k_pixel']
        self.momentum = momentum

        # for memory bank
        self.index = 0
        self.register_buffer('params', torch.tensor([-1]))
        stdv = 1. / math.sqrt(self.nce_dim / 3)
        memory = torch.rand(self.nce_k, self.nce_dim, requires_grad=False).mul_(2 * stdv).add_(-stdv)
        self.register_buffer('memory', memory)

        # for pixel memory bank
        self.index_pixel = 0
        self.register_buffer('params', torch.tensor([-1]))
        stdv = 1. / math.sqrt(self.nce_dim / 3)
        memory = torch.rand(self.nce_k_pixel, self.nce_dim, requires_grad=False).mul_(2 * stdv).add_(-stdv)
        self.register_buffer('pixel_memory', memory)

        self.loss_lambda = loss_lambda 

    
    def forward(self, test_mode=False, **kwargs):
        
        if test_mode:
            return self.forward_test(**kwargs)

        return self.forward_train(**kwargs)

    def train_step(self, data_batch, optimizer):
    
        # parser loss
        losses = self(**data_batch, test_mode=False)
        loss, log_vars = self.parse_losses(losses)

        # optimizer
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        log_vars.pop('loss')
        outputs = dict(
            log_vars=log_vars,
            num_samples=len(data_batch['Fs'])
        )
        return outputs
    
    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(self.encoder_q.parameters(),
                                    self.encoder_k.parameters()):
            param_k.data = param_k.data * self.momentum + \
                           param_q.data * (1. - self.momentum)

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
    def update_memory(self, k_all_sf, memory_name='pixel'):

        k_all = k_all_sf
        all_size = k_all.shape[0]

        if memory_name == 'pixel':
            out_ids = torch.fmod(torch.arange(all_size, dtype=torch.long).cuda() + self.index_pixel, self.nce_k_pixel)
            self.pixel_memory.index_copy_(0, out_ids, k_all)
            self.index_pixel = (self.index_pixel + all_size) % self.nce_k_pixel
        else:
            out_ids = torch.fmod(torch.arange(all_size, dtype=torch.long).cuda() + self.index, self.nce_k)
            self.memory.index_copy_(0, out_ids, k_all)
            self.index = (self.index + all_size) % self.nce_k


    def forward_train(self, images):
        assert images.dim() == 5, \
            "Input must have 5 dims, got: {}".format(img.dim())
        im_q = img[:, :, 0,  ...].contiguous()
        im_k = img[:, :, 1,  ...].contiguous()
        # compute query features
        q_b = self.encoder_q[0](im_q) # backbone features
        q, q_grid, q2 = self.encoder_q[1](q_b)  # queries: NxC; NxCxS^2
        q_b = q_b[0]
        q_b = q_b.view(q_b.size(0), q_b.size(1), -1)

        q = nn.functional.normalize(q, dim=1)
        q2 = nn.functional.normalize(q2, dim=1)
        q_grid = nn.functional.normalize(q_grid, dim=1)
        q_b = nn.functional.normalize(q_b, dim=1)

        # compute key features
        with torch.no_grad():  # no gradient to keys
            self._momentum_update_key_encoder()  # update the key encoder

            # shuffle for making use of BN
            im_k, idx_unshuffle = self._batch_shuffle_ddp(im_k)

            k_b = self.encoder_k[0](im_k)
            k, k_grid, k2 = self.encoder_k[1](k_b)  # keys: NxC; NxCxS^2
            k_b = k_b[0]
            k_b = k_b.view(k_b.size(0), k_b.size(1), -1)

            k = nn.functional.normalize(k, dim=1)
            k2 = nn.functional.normalize(k2, dim=1)
            k_grid = nn.functional.normalize(k_grid, dim=1)
            k_b = nn.functional.normalize(k_b, dim=1)

            # undo shuffle
            k = self._batch_unshuffle_ddp(k, idx_unshuffle)
            k2 = self._batch_unshuffle_ddp(k2, idx_unshuffle)
            k_grid = self._batch_unshuffle_ddp(k_grid, idx_unshuffle)
            k_b = self._batch_unshuffle_ddp(k_b, idx_unshuffle)

        # compute logits
        # Einstein sum is more intuitive
        # positive logits: Nx1
        l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)
        # negative logits: NxK
        l_neg = torch.einsum('nc,ck->nk', [q, self.memory.clone().detach()])

        # feat point set sim
        backbone_sim_matrix = torch.matmul(q_b.permute(0, 2, 1), k_b)
        densecl_sim_ind = backbone_sim_matrix.max(dim=2)[1] # NxS^2

        indexed_k_grid = torch.gather(k_grid, 2, densecl_sim_ind.unsqueeze(1).expand(-1, k_grid.size(1), -1)) # NxCxS^2
        densecl_sim_q = (q_grid * indexed_k_grid).sum(1) # NxS^2

        l_pos_dense = densecl_sim_q.view(-1).unsqueeze(-1) # NS^2X1

        q_grid = q_grid.permute(0, 2, 1)
        q_grid = q_grid.reshape(-1, q_grid.size(2))
        l_neg_dense = torch.einsum('nc,ck->nk', [q_grid,
                                            self.memory_pixel.clone().detach()])

        loss_single = self.nce_loss(torch.cat([l_pos, l_neg], dim=1))
        loss_dense = self.head(torch.cat([l_pos_dense, l_neg_dense], dim=1))

        losses = dict()
        losses['loss_contra_single'] = loss_single * (1 - self.loss_lambda)
        losses['loss_contra_dense'] = loss_dense * self.loss_lambda

        self.update_memory(k)
        self.update_memory(k2, 'instance')

        return losses
