# Copyright (c) OpenMMLab. All rights reserved.
import numbers
import os.path as osp
from collections import *

import mmcv
from mmcv.runner import auto_fp16

from ..base import BaseModel
from ..builder import build_backbone, build_loss
from ..registry import MODELS
from vcl.utils.helpers import *


import torch.nn as nn
import torch
import torch.nn.functional as F
import math


@MODELS.register_module()
class Vqvae_Tracker(BaseModel):

    def __init__(self,
                 backbone,
                 pixel_loss,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None
                 ):
        """ Space-Time Memory Network for Video Object Segmentation

        Args:
            depth ([type]): ResNet depth for encoder
            pixel_loss ([type]): loss option
            train_cfg ([type], optional): [description]. Defaults to None.
            test_cfg ([type], optional): [description]. Defaults to None.
            pretrained ([type], optional): [description]. Defaults to None.
        """
        super(Vqvae_Tracker, self).__init__()

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        self.backbone = build_backbone(backbone)

        # loss
        self.pixel_ce_loss = build_loss(pixel_loss)


        if test_cfg is not None:
            self.MEMORY_EVERY_FRAME = test_cfg.test_memory_every_frame
            self.MEMORY_NUM = test_cfg.memory_num


 
    def Pad_memory(self, mems, num_objects, K):
        pad_mems = []
        for mem in mems:
            pad_mem = ToCuda(torch.zeros(1, K, mem.size()[1], 1, mem.size()[2], mem.size()[3]))
            pad_mem[0,1:num_objects+1,:,0] = mem
            pad_mems.append(pad_mem)
        return pad_mems

    def memorize(self, frame, masks, num_objects): 
        # memorize a frame 
        num_objects = num_objects[0].item()
        _, K, H, W = masks.shape # B = 1

        (frame, masks), pad = pad_divide_by([frame, masks], 16, (frame.size()[2], frame.size()[3]))

        # make batch arg list
        B_list = {'f':[], 'm':[], 'o':[]}
        for o in range(1, num_objects+1): # 1 - no
            B_list['f'].append(frame)
            B_list['m'].append(masks[:,o])
            B_list['o'].append( (torch.sum(masks[:,1:o], dim=1) + \
                torch.sum(masks[:,o+1:num_objects+1], dim=1)).clamp(0,1) )

        # make Batch
        B_ = {}
        for arg in B_list.keys():
            B_[arg] = torch.cat(B_list[arg], dim=0)

        r4, _, _, _, _ = self.Encoder_M(B_['f'], B_['m'], B_['o'])
        k4, v4 = self.KV_M_r4(r4) # num_objects, 128 and 512, H/16, W/16
        k4, v4 = self.Pad_memory([k4, v4], num_objects=num_objects, K=K)
        return k4, v4

    def Soft_aggregation(self, ps, K):
        num_objects, H, W = ps.shape
        em = ToCuda(torch.zeros(1, K, H, W)) 
        em[0,0] =  torch.prod(1-ps, dim=0) # bg prob
        em[0,1:num_objects+1] = ps # obj prob
        em = torch.clamp(em, 1e-7, 1-1e-7)
        logit = torch.log((em /(1-em)))
        return logit

    def segment(self, frame, keys, values, num_objects): 
        num_objects = num_objects[0].item()
        _, K, keydim, T, H, W = keys.shape # B = 1
        # pad
        [frame], pad = pad_divide_by([frame], 16, (frame.size()[2], frame.size()[3]))

        r4, r3, r2, _, _ = self.Encoder_Q(frame)
        k4, v4 = self.KV_Q_r4(r4)   # 1, dim, H/16, W/16
        
        # expand to ---  no, c, h, w
        k4e, v4e = k4.expand(num_objects,-1,-1,-1), v4.expand(num_objects,-1,-1,-1) 
        r3e, r2e = r3.expand(num_objects,-1,-1,-1), r2.expand(num_objects,-1,-1,-1)
        
        # memory select kv:(1, K, C, T, H, W)
        m4, viz = self.Memory(keys[0,1:num_objects+1], values[0,1:num_objects+1], k4e, v4e)

        logits = self.Decoder(m4, r3e, r2e)
        ps = F.softmax(logits, dim=1)[:,1] # no, h, w  
        #ps = indipendant possibility to belong to each object
        
        logit = self.Soft_aggregation(ps, K) # 1, K, H, W

        if pad[2]+pad[3] > 0:
            logit = logit[:,:,pad[2]:-pad[3],:]
        if pad[0]+pad[1] > 0:
            logit = logit[:,:,:,pad[0]:-pad[1]]

        return logit


    def forward_train(self, Fs, Ms, num_objects):
        """
        Args:
            Fs ([type]): frames
            Ms ([type]): masks
            num_objects ([type]): number of objects

        Returns:
            [type]: loss dict
        """
        self.freeze_bn()

        losses = dict()

        Es = torch.zeros_like(Ms).cuda()
        Es[:,:,0] = Ms[:,:,0]

        n1_key, n1_value = self.memorize(Fs[:,:,0], Es[:,:,0], torch.tensor([num_objects]).cuda())

        # segment
        n2_logit = self.segment(Fs[:,:,1], n1_key, n1_value, torch.tensor([num_objects]).cuda())

        n2_label = torch.argmax(Ms[:,:,1],dim = 1).long().cuda()

        Es[:,:,1] = F.softmax(n2_logit, dim=1).detach()

        n2_key, n2_value = self.memorize(Fs[:,:,1], Es[:,:,1], torch.tensor([num_objects]).cuda())
        n12_keys = torch.cat([n1_key, n2_key], dim=3)
        n12_values = torch.cat([n1_value, n2_value], dim=3)

        # segment
        n3_logit = self.segment(Fs[:,:,2], n12_keys, n12_values, torch.tensor([num_objects]).cuda())
        n3_label = torch.argmax(Ms[:,:,2],dim = 1).long().cuda()

        losses['loss_pixel_ce_n2'] = self.pixel_ce_loss(n2_logit,n2_label)
        losses['loss_pixel_ce_n3'] = self.pixel_ce_loss(n3_logit, n3_label)

        return losses

    def forward_test(self, dataset, num_frames, num_objects, video,
                    save_image=False,
                    save_path=None,
                    iteration=None
                    ):
        pass
    


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