from builtins import isinstance, list
import os.path as osp
from collections import *
from pickle import NONE
from re import A

import mmcv
import tempfile
from mmcv.runner import auto_fp16, load_checkpoint
from dall_e  import map_pixels, unmap_pixels, load_model
from torch import bilinear, unsqueeze

from vcl.models.common.warp import *
from vcl.models.common.occlusion_estimation import *

from vcl.models.common.hoglayer import *

from ..base import BaseModel
from ..builder import build_backbone, build_components, build_loss
from ..registry import MODELS
from vcl.utils import *

import torch.nn as nn
import torch.nn.functional as F
from .modules import *



@MODELS.register_module()
class Single_Frame_Tracker(BaseModel):
    def __init__(self,
                 backbone,
                 backbone_t,
                 per_ref=True,
                 head=None,
                 downsample_rate=4,
                 temperature=1,
                 scaling=True,
                 upsample=True,
                 weight=20,
                 test_cfg=None,
                 train_cfg=None,
                 pretrained=None,
                 ):
        """ tracking using single frame

        Args:
            backbone ([type]): [description]
            test_cfg ([type], optional): [description]. Defaults to None.
            train_cfg ([type], optional): [description]. Defaults to None.
        """
        super().__init__()

        self.backbone = build_backbone(backbone)
        self.backbone_t = build_backbone(backbone_t)
        self.downsample_rate = downsample_rate
        if head is not None:
            self.head =  build_components(head)
        else:
            self.head = None
            
        self.per_ref = per_ref
        self.test_cfg = test_cfg
        self.train_cfg = train_cfg
        
        self.logger = get_root_logger()
        
        self.pretrained = pretrained
        self.scaling = scaling
        self.upsample = upsample
        self.weight = weight
        self.temperature = temperature
        self.warp = Warp()
        
        self.flow_conv = nn.Sequential(
            nn.Conv2d(256, 128, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(128, 64, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(64, 32, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(32, 2, 3, 1, 1), 
        )
        
        self.init_weights()

    def init_weights(self):
        self.backbone.init_weights()
        if self.pretrained != None:
            _ = load_checkpoint(self, self.pretrained, strict=False, map_location='cpu')
    
    def forward_train(self, images_lab, imgs=None):
            
        bsz, _, n, c, h, w = images_lab.shape
        
        images_lab_gt = [images_lab[:,0,i].clone() for i in range(n)]
        images_lab = [images_lab[:,0,i] for i in range(n)]
        ch = np.random.choice(np.arange(1,3), 1, replace=False)
        
        # forward to get feature
        flow_forward = self.flow_conv(self.backbone(images_lab[0]))
        # flow_backward = self.flow_conv(self.backbone(images_lab[1]))
        
        # estimate occ
        # occs = occlusion_estimation(flow_forward, flow_backward)
        tar_t = self.backbone_t(imgs[:,0,0])
        tf = (tar_t.mean(1).flatten(-2))
        tf_sorted, _ = torch.sort(tf, dim=-1, descending=True)
        idx = int(tf.shape[-1] * 0.5) - 1
        T = tf_sorted[:, idx:idx+1]
        mask = ( tf > T).bool().reshape(bsz, 1, *tar_t.shape[-2:])
              
        losses = {}
        
        # for forward l1_loss
        ref_gt = self.prep(images_lab_gt[-1][:,ch])
        warp_frame = self.warp(ref_gt, flow_forward)
        losses['fw_loss'], err_map = self.compute_lphoto(images_lab_gt[0][:,ch], warp_frame, mask=mask, upsample=self.upsample)
        
        # for backward l1_loss
        # ref_gt = self.prep(images_lab_gt[0][:,ch])
        # warp_frame = self.warp(ref_gt, flow_backward)
        # losses['bw_loss'], err_map = self.compute_lphoto(images_lab_gt[-1][:,ch], warp_frame, mask=occs['occ_bw'], upsample=self.upsample)
        
        vis_results = dict(err=err_map[0], imgs=imgs[0,0])
        
        return losses, vis_results
    
    def train_step(self, data_batch, optimizer, progress_ratio):
        """Abstract method for one training step.

        All subclass should overwrite it.
        """
        # parser loss
        losses, vis_results = self(**data_batch, test_mode=False)
        loss, log_vars = self.parse_losses(losses)

        # optimizer
        if isinstance(optimizer, dict):
            for k,opz in optimizer.items():
                opz.zero_grad()

            loss.backward()
            for k,opz in optimizer.items():
                opz.step()
        else:
            optimizer.zero_grad()

            loss.backward()
            optimizer.step()

        log_vars.pop('loss')
        outputs = dict(
            log_vars=log_vars,
            num_samples=len(data_batch['imgs']),
            vis_results=vis_results,
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
    
    def compute_lphoto(self, images_lab_gt, outputs, mask=None, upsample=True):
        b, c, h, w = images_lab_gt.size()

        tar_y = images_lab_gt  # y4

        if upsample:
            outputs = F.interpolate(outputs, (h, w), mode='bilinear')
            loss = F.smooth_l1_loss(outputs * self.weight, tar_y * self.weight, reduction='mean')
        else:
            tar_y = self.prep(images_lab_gt)
            loss = F.smooth_l1_loss(outputs * self.weight, tar_y * self.weight, reduction='none')
            if mask != None:
                loss = (loss * mask).sum() / ( mask.sum() + 1e-12 )
            
        err_maps = torch.abs(outputs - tar_y).sum(1).detach()

        return loss, err_maps
    
    def prep(self, image, mode='default'):
        _,c,_,_ = image.size()

        x = image.float()[:,:,::self.downsample_rate,::self.downsample_rate]

        return x
    

@MODELS.register_module()
class CMP(BaseModel):
    def __init__(self,
                 backbone,
                 loss=None,
                 img_enc_dim=256,
                 sparse_enc_dim=16,
                 output_dim=198,
                 decoder_combo=[1,2,4],
                 test_cfg=None,
                 train_cfg=None,
                 pretrained=None,
                 ):
        """ tracking using single frame

        Args:
            backbone ([type]): [description]
            test_cfg ([type], optional): [description]. Defaults to None.
            train_cfg ([type], optional): [description]. Defaults to None.
        """
        super().__init__()

        self.backbone = build_backbone(backbone)
        self.conv = nn.Conv2d(self.backbone.feat_dim, img_enc_dim, 3, 1, 1)
        self.flow_encoder = shallownet8x(output_dim=sparse_enc_dim)
        self.flow_decoder = MotionDecoderPlain(
            input_dim=img_enc_dim+sparse_enc_dim,
            output_dim=output_dim,
            combo=decoder_combo)
        self.loss = build_loss(loss)
            
        self.test_cfg = test_cfg
        self.train_cfg = train_cfg
        
        self.logger = get_root_logger()
        self.pretrained = pretrained

        

    def forward_train(self, imgs, flows, sparse, mask):
        
        sparse = torch.cat([sparse, mask], -1).permute(0, 3, 1, 2).float()
        image = imgs[: , 0, :, 0]
                
        sparse_enc = self.flow_encoder(sparse)
        img_enc = self.conv(self.backbone(image))
        flow_dec = self.flow_decoder(torch.cat((img_enc, sparse_enc), dim=1))
        
        losses = {}
        losses['cmp_loss'] = self.loss(flow_dec, flows[: , 0, :2, 0])
        return losses