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



    
@MODELS.register_module()
class Memory_Tracker_Custom(BaseModel):
    def __init__(self,
                 backbone,
                 loss_weight=dict(l1_loss=1),
                 per_ref=True,
                 head=None,
                 downsample_rate=[4,],
                 radius=[12,],
                 temperature=1,
                 feat_size=[64,],
                 conc_loss=None,
                 forward_backward_t=-1,
                 drop_last=True,
                 scaling=True,
                 upsample=True,
                 weight=20,
                 rec_sampling='stride',
                 test_cfg=None,
                 train_cfg=None,
                 pretrained=None,
                 ):
        """ MAST  (CVPR2020)

        Args:
            backbone ([type]): [description]
            test_cfg ([type], optional): [description]. Defaults to None.
            train_cfg ([type], optional): [description]. Defaults to None.
        """
        super().__init__()

        self.backbone = build_backbone(backbone)
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
        self.drop_last = drop_last
        self.scaling = scaling
        self.upsample = upsample
        self.rec_sampling = rec_sampling
        self.weight = weight
        self.temperature = temperature
        self.forward_backward_t = forward_backward_t
        self.conc_loss = build_loss(conc_loss) if conc_loss is not None else None
        

        self.mask = [ make_mask(feat_size[idx], radius[idx]) for idx, r in enumerate(radius)]
        self.R = radius # radius in previous
        self.radius = radius 
        self.feat_size = feat_size

        self.loss_weight = loss_weight

        assert len(self.feat_size) == len(self.radius) == len(self.downsample_rate) 


    
    def forward_train(self, images_lab, imgs=None):
            
        bsz, _, n, c, h, w = images_lab.shape

        images_lab_gt = [images_lab[:,0,i].clone() for i in range(n)]
        images_lab = [images_lab[:,0,i] for i in range(n)]
        _, ch = self.dropout2d_lab(images_lab)
        
        # forward to get feature
        fs = self.backbone(torch.stack(images_lab,1).flatten(0,1))
        if self.head is not None:
            fs = self.head(fs)
        
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
        
        vis_results = dict(err=None, imgs=imgs[0,0])

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
            num_samples=len(data_batch['images_lab']),
            vis_results=vis_results,
        )

        return outputs    
    
    def frame_reconstruction(self, gts, att, ch, feat_size=32, downsample_rate=8):
        bsz = att.shape[0]
        ref_gt = [self.prep(gt[:,ch],downsample_rate=downsample_rate) for gt in gts[:-1]]
        outputs = frame_transform(att, ref_gt, flatten=False, per_ref=self.per_ref)
        if self.per_ref:
            outputs = outputs[:,0].permute(0,2,1).reshape(bsz, -1, feat_size, feat_size)     
        else:
            outputs = outputs.permute(0,2,1).reshape(bsz, -1, feat_size, feat_size)    
        return outputs

    def dropout2d_lab(self, arr): # drop same layers for all images
        if not self.training:
            return arr

        drop_ch_num = int(np.random.choice(np.arange(1, 2), 1))
        drop_ch_ind = np.random.choice(np.arange(1,3), drop_ch_num, replace=False)

        for idx, a in enumerate(arr):
            if idx == len(arr) - 1 and not self.drop_last:
                continue
            for dropout_ch in drop_ch_ind:
                a[:, dropout_ch] = 0
            a *= (3 / (3 - drop_ch_num))

        return arr, drop_ch_ind # return channels not masked
    
    def compute_lphoto(self, images_lab_gt, ch, outputs, upsample=True, mask=None, gt_idx=-1):
        b, c, h, w = images_lab_gt[0].size()

        tar_y = images_lab_gt[gt_idx][:,ch]  # y4

        if upsample:
            outputs = F.interpolate(outputs, (h, w), mode='bilinear')
            loss = F.smooth_l1_loss(outputs * self.weight, tar_y * self.weight, reduction='mean')
        else:
            tar_y = self.prep(images_lab_gt[-1], self.downsample_rate[-1])[:,ch]
            loss = F.smooth_l1_loss(outputs * self.weight, tar_y * self.weight, reduction='none')
            if mask == None:
                loss = loss.mean()
            else:
                loss = (loss * mask).sum() / (mask.sum() + 1e-9)

        err_maps = torch.abs(outputs - tar_y).sum(1).detach()

        return loss, err_maps
    
    def prep(self, image, downsample_rate=8):
        _,c,_,_ = image.size()

        if self.rec_sampling == 'stride':
            x = image.float()[:,:,::downsample_rate,::downsample_rate]
        elif self.rec_sampling == 'centered':
            x = image.float()[:,:,downsample_rate//2::downsample_rate, downsample_rate//2::downsample_rate]
        else:
            raise NotImplementedError

        return x

    def forward_backward_check(self, att):
        
        if self.mask is not None:
            att = torch.exp(att)
            att.masked_fill_(~self.mask[0].bool(), 0) # for normalized inner product

        bsz = att.shape[0]
        # forward backward consistency
        aff_i = torch.max(att, dim=-1, keepdim=True)[0]
        aff_j = torch.max(att, dim=-2, keepdim=True)[0]
        Q = (att * att) / (torch.matmul(aff_i, aff_j))
        Q = Q[:,0].max(dim=-1)[0]
        M = (Q >= self.forward_backward_t).reshape(bsz, 1, int(att.shape[-1] ** 0.5), -1)

        return M

@MODELS.register_module()
class Memory_Tracker_Custom_Feat_Rec(Memory_Tracker_Custom):
    def __init__(self, backbone_f, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.backbone_f = build_backbone(backbone_f)

    def forward_train(self, images_lab, imgs=None):
        bsz, _, n, c, h, w = images_lab.shape
        
        images = [imgs[:,0,i].clone() for i in range(n)]
        images_lab_gt = [images_lab[:,0,i].clone() for i in range(n)]
        images_lab = [images_lab[:,0,i] for i in range(n)]
        _, ch = self.dropout2d_lab(images_lab)
        
        # forward to get feature
        with torch.no_grad():
            self.backbone_f.eval()
            fs_f = self.backbone_f(torch.stack(images,1).flatten(0,1))
            fs_f = fs_f.reshape(bsz, n, *fs_f.shape[-3:])
            tar_f, refs_f = fs_f[:, -1], fs_f[:, :-1]

        fs = self.backbone(torch.stack(images_lab,1).flatten(0,1))
        if self.head is not None:
            fs = self.head(fs)
        fs = fs.reshape(bsz, n, *fs.shape[-3:])
        tar, refs = fs[:, -1], fs[:, :-1]
        
        # get correlation attention map      
        _, att = non_local_attention(tar, refs, mask=None, scaling=self.scaling, per_ref=self.per_ref, temprature=self.temperature)    

        if self.forward_backward_t != -1:
            att_ = non_local_attention(tar, refs, att_only=True, norm=True)    
            m = self.forward_backward_check(att_)
        else:
            m = None
        
        losses = {}
        if self.conc_loss !=  None:
            att_cos = non_local_attention(tar, refs, att_only=True)
            losses['conc_loss'] = self.conc_loss(att_cos)
        
        tar_f = tar_f.flatten(-2).permute(0,2,1).flatten(0,1)
        rec_f = frame_transform(att, refs_f.flatten(-2).permute(0,1,3,2))
        losses['ft_rec_loss'] = build_loss(dict(type='MSELoss'))(rec_f, tar_f.detach())

        vis_results = dict(err=None, imgs=imgs[0,0])

        return losses, vis_results


@MODELS.register_module()
class Memory_Tracker_Custom_Pyramid(Memory_Tracker_Custom):
    def __init__(self,
                loss=None,
                pool_type='mean',
                bilinear_downsample=True,
                reverse=True,
                num_stage=2,
                detach=False,
                *args,
                **kwargs
                 ):
        """ MAST  (CVPR2020)

        Args:
            backbone ([type]): [description]
            test_cfg ([type], optional): [description]. Defaults to None.
            train_cfg ([type], optional): [description]. Defaults to None.
        """
        super().__init__(*args, **kwargs)
        self.reverse = reverse
        self.num_stage = num_stage

        self.loss = build_loss(loss) if loss is not None else None
        self.pool_type = pool_type
        self.bilinear_downsample = bilinear_downsample
        self.detach = detach


    def forward_train(self, imgs, images_lab=None):
            
        bsz, num_clips, t, c, h, w = images_lab.shape
        
        images_lab_gt = [images_lab[:,0,i,:].clone() for i in range(t)]
        images_lab = [images_lab[:,0,i,:] for i in range(t)]
        _, ch = self.dropout2d_lab(images_lab)
        
        # forward to get feature
        fs = self.backbone(torch.stack(images_lab,1).flatten(0,1))
        fs = [f.reshape(bsz, t, *f.shape[-3:]) for f in fs]
        
        tar_pyramid, refs_pyramid = [f[:, -1] for f in fs], [ f[:, :-1] for f in fs]
        
        losses = {}
        
        atts = []
        for idx, (tar, refs) in enumerate(zip(tar_pyramid, refs_pyramid)):
            # get correlation attention map            
            _, att = non_local_attention(tar, refs, mask=self.mask[idx], scaling=True)            
            # for mast l1_loss
            outputs = self.frame_reconstruction(images_lab_gt, att, ch, self.feat_size[idx], self.downsample_rate[idx])    
            losses[f'stage{idx}_l1_loss'], err_map = self.compute_lphoto(images_lab_gt, ch, outputs)
                
            atts.append(att)
        
        if self.bilinear_downsample:
            if not self.reverse:
                atts[0] = atts[0].permute(0,1,3,2)
            if self.pool_type == 'mean':
                att_ = atts[0].reshape(bsz, -1, *fs[0].shape[-2:])
                att_ = F.avg_pool2d(att_, 2, stride=2).flatten(-2).permute(0,2,1).reshape(bsz, -1, *fs[0].shape[-2:])
                target = F.avg_pool2d(att_, 2, stride=2).flatten(-2).permute(0,2,1)    
            elif self.pool_type == 'max':
                att_ = atts[0].reshape(bsz, -1, *fs[0].shape[-2:])
                att_ = F.max_pool2d(att_, 2, stride=2).flatten(-2).permute(0,2,1).reshape(bsz, -1, *fs[0].shape[-2:])
                target = F.max_pool2d(att_, 2, stride=2).flatten(-2).permute(0,2,1)

            if not self.reverse:
                target = target.permute(0,2,1)
        
            losses['dist_loss'] = self.loss(atts[-1][:,0], target.detach()) if self.detach else \
                self.loss(atts[-1][:,0], target) 
        else:
            if not self.reverse:
                atts[1] = atts[1].permute(0,1,3,2)
    
            att_ = atts[1].reshape(bsz, -1, *fs[1].shape[-2:])
            att_ = F.interpolate(att_, size=fs[0].shape[-2:],mode="bilinear").flatten(-2).permute(0,2,1).reshape(bsz, -1, *fs[1].shape[-2:])
            att_ = F.interpolate(att_, size=fs[0].shape[-2:],mode="bilinear").flatten(-2).permute(0,2,1)    

            if not self.reverse:
                att_ = att_.permute(0,2,1)
                
            losses['dist_loss'] = self.loss(att_, atts[0][:,0].detach()) if self.detach else \
                self.loss(att_, atts[0][:,0]) 
            
        vis_results = dict(err=err_map[0], imgs=imgs[0,0])

        return losses, vis_results
        
    
    def prep(self, image, downsample_rate):
        _,c,_,_ = image.size()
        x = image.float()[:,:,::downsample_rate,::downsample_rate]

        return x
            


@MODELS.register_module()
class Memory_Tracker_Custom_V2(Memory_Tracker_Custom):
    def __init__(self,
                 *args, **kwargs
                 ):
        """ MAST  (CVPR2020) using MMCV Correlation Module

        Args:
            backbone ([type]): [description]
            test_cfg ([type], optional): [description]. Defaults to None.
            train_cfg ([type], optional): [description]. Defaults to None.
        """
        super().__init__(*args, **kwargs)
        from mmcv.ops import Correlation
    
        self.corr = [ Correlation(max_displacement=R) for R in self.R ]
        
    def forward_train(self, images_lab, imgs=None):
            
        bsz, _, n, c, h, w = images_lab.shape
        
        images_lab_gt = [images_lab[:,0,i].clone() for i in range(n)]
        images_lab = [images_lab[:,0,i] for i in range(n)]
        _, ch = self.dropout2d_lab(images_lab)
        
        # forward to get feature
        fs = self.backbone(torch.stack(images_lab,1).flatten(0,1))
        if self.head is not None:
            fs = self.head(fs)
        
        fs = fs.reshape(bsz, n, *fs.shape[-3:])

        tar, refs = fs[:, -1], fs[:, :-1]
        
        # get correlation attention map      
        corr = self.corr[0](tar, refs[:,0]) 
        if self.scaling:
            corr = corr / torch.sqrt(torch.tensor(tar.shape[1]).float()) 
        corr = corr.flatten(1,2).softmax(1)
        
        losses = {}
        
        # for mast l1_loss
        ref_gt = self.prep(images_lab_gt[0][:,ch], self.downsample_rate[0])
        ref_gt = F.unfold(ref_gt, self.R[0] * 2 + 1, padding=self.R[0])
        
        corr = corr.reshape(bsz, -1, tar.shape[-1]**2)
        outputs = (corr * ref_gt).sum(1).reshape(bsz, -1, *fs.shape[-2:])        
        losses['l1_loss'], err_map = self.compute_lphoto(images_lab_gt, ch, outputs, upsample=self.upsample)
        
        vis_results = dict(err=err_map[0], imgs=imgs[0,0])

        return losses, vis_results



@MODELS.register_module()
class Memory_Tracker_Custom_Iterative(Memory_Tracker_Custom):

    def forward_train(self, images_lab, imgs=None):
            
        bsz, _, n, c, h, w = images_lab.shape

        assert n >= 3, "video frames must be larger than 3"
        # x = tensor2img(imgs[0,0], norm_mode='mean-std')
        
        images_lab_gt = [images_lab[:,0,i].clone() for i in range(n)]
        images_lab = [images_lab[:,0,i] for i in range(n)]
        _, ch = self.dropout2d_lab(images_lab)
        
        # forward to get feature
        fs = self.backbone(torch.stack(images_lab,1).flatten(0,1))
        if self.head is not None:
            fs = self.head(fs)
        
        fs = fs.reshape(bsz, n, *fs.shape[-3:])

        pre_out = []
        losses = {}

        for i in range(n-1):

            tar, refs = fs[:, i+1], fs[:, i:i+1]
            # get correlation attention map      
            _, att = non_local_attention(tar, refs, mask=self.mask[0], scaling=self.scaling, per_ref=self.per_ref, temprature=self.temperature)    

            if self.forward_backward_t != -1:
                att_ = non_local_attention(tar, refs, att_only=True, norm=True)    
                m = self.forward_backward_check(att_)
            else:
                m = None
            
            if self.conc_loss !=  None:
                att_cos = non_local_attention(tar, refs, att_only=True)
                losses['conc_loss'] = self.conc_loss(att_cos)
            
            # for mast l1_loss
            outputs = self.frame_reconstruction(images_lab_gt, att, ch, pre_out, feat_size=self.feat_size[0], downsample_rate=self.downsample_rate[0]) 
            losses[f'frame{i}_l1_loss'] = self.loss_weight['l1_loss'] * self.compute_lphoto(images_lab_gt, ch, outputs, upsample=self.upsample, mask=m, gt_idx=i+1)[0]
            pre_out.append(outputs)

        vis_results = dict(err=None, imgs=imgs[0,0])

        return losses, vis_results

    def frame_reconstruction(self, gts, att, ch, pre_out=None, feat_size=32, downsample_rate=8):
        bsz = att.shape[0]

        # value are concat of gt and predict
        if len(pre_out) > 0:
            # refs = [self.prep(gts[0][:,ch],downsample_rate=downsample_rate)]
            # refs.extend(pre_out)
            refs = pre_out
        else:
            refs = [self.prep(gt[:,ch],downsample_rate=downsample_rate) for gt in gts[:1]]

        outputs = frame_transform(att, refs, flatten=False, per_ref=self.per_ref)
        if self.per_ref:
            outputs = outputs[:,0].permute(0,2,1).reshape(bsz, -1, feat_size, feat_size)     
        else:
            outputs = outputs.permute(0,2,1).reshape(bsz, -1, feat_size, feat_size)    
        return outputs



@MODELS.register_module()
class Memory_Tracker_Custom_Pyramid_Cmp_V2(Memory_Tracker_Custom):
    def __init__(self,
                loss=None,
                pool_type='mean',
                bilinear_downsample=True,
                reverse=True,
                T=0.2,
                num_stage=2,
                feat_size=[64, 32],
                radius=[12, 6],
                downsample_rate=[4, 8],
                detach=False,
                *args,
                **kwargs
                 ):
        """ MAST  (CVPR2020) with CMP (CVPR2019)

        Args:
            backbone ([type]): [description]
            test_cfg ([type], optional): [description]. Defaults to None.
            train_cfg ([type], optional): [description]. Defaults to None.
        """
        super().__init__(*args, **kwargs)
        from mmcv.ops import Correlation
        
        self.reverse = reverse
        self.num_stage = num_stage
        self.feat_size = feat_size
        self.downsample_rate = downsample_rate
        self.loss = build_loss(loss) if loss is not None else None
        self.pool_type = pool_type
        self.bilinear_downsample = bilinear_downsample
        self.detach = detach
        self.radius = radius
        self.T = T
        
        self.mask = [ make_mask(feat_size[i], radius[i]) for i in range(len(radius)) if radius[i] != -1]        
        self.corr = [Correlation(max_displacement=R) for R in radius ]
        
        
        
    def forward_train(self, imgs, images_lab=None):
            
        bsz, num_clips, t, c, h, w = images_lab.shape
        
        images_lab_gt = [images_lab[:,0,i,:].clone() for i in range(t)]
        images_lab = [images_lab[:,0,i,:] for i in range(t)]
        _, ch = self.dropout2d_lab(images_lab)
        
        # forward to get feature
        fs = self.backbone(torch.stack(images_lab,1).flatten(0,1))
        fs = [f.reshape(bsz, t, *f.shape[-3:]) for f in fs]
        
        tar_pyramid, refs_pyramid = [f[:, -1] for f in fs], [ f[:, :-1] for f in fs]
        
        losses = {}
        
        atts = []
        corrs = []
        for idx, (tar, refs) in enumerate(zip(tar_pyramid, refs_pyramid)):
            # get correlation attention map            
            _, att = non_local_attention(tar, refs, mask=self.mask[idx], scaling=True)            
            # for mast l1_loss
            ref_gt = [self.prep(gt[:,ch], downsample_rate=self.downsample_rate[idx]) for gt in images_lab_gt[:-1]]
            outputs = frame_transform(att, ref_gt, flatten=False)
            outputs = outputs[:,0].permute(0,2,1).reshape(bsz, -1, self.feat_size[idx], self.feat_size[idx])     
            losses[f'stage{idx}_l1_loss'], err_map = self.compute_lphoto(images_lab_gt, ch, outputs)
            
            corr = self.corr[idx](tar, refs[:,0])
            atts.append(att)
            corrs.append(corr)
        
        # for cmp weight
        corr_feat = corrs[-1].reshape(bsz, -1, self.feat_size[-1], self.feat_size[-1])
        corr = corr_feat.softmax(1)
        corr_en = (-torch.log(corr+1e-12)).sum(1).flatten(-2)
        corr_sorted, _ = torch.sort(corr_en, dim=-1, descending=True)
        idx = int(corr_en.shape[-1] * self.T) - 1
        T = corr_sorted[:, idx:idx+1]
        sparse_mask = (corr_en > T).reshape(bsz, 1, *corr_feat.shape[-2:]).float().detach()
            
        if self.bilinear_downsample:
            if not self.reverse:
                atts[0] = atts[0].permute(0,1,3,2)
            if self.pool_type == 'mean':
                att_ = atts[0].reshape(bsz, -1, *fs[0].shape[-2:])
                att_ = F.avg_pool2d(att_, 2, stride=2).flatten(-2).permute(0,2,1).reshape(bsz, -1, *fs[0].shape[-2:])
                target = F.avg_pool2d(att_, 2, stride=2).flatten(-2).permute(0,2,1)    
            elif self.pool_type == 'max':
                att_ = atts[0].reshape(bsz, -1, *fs[0].shape[-2:])
                att_ = F.max_pool2d(att_, 2, stride=2).flatten(-2).permute(0,2,1).reshape(bsz, -1, *fs[0].shape[-2:])
                target = F.max_pool2d(att_, 2, stride=2).flatten(-2).permute(0,2,1)

            if not self.reverse:
                target = target.permute(0,2,1)
            
            weight = sparse_mask.flatten(-2).permute(0,2,1).repeat(1, 1, target.shape[-1])
            
            losses['dist_loss'] = self.loss(atts[-1][:,0], target.detach(), weight=weight) if self.detach else \
                self.loss(atts[-1][:,0], target, weight=weight) 
            
        else:
            if not self.reverse:
                atts[1] = atts[1].permute(0,1,3,2)
    
            att_ = atts[1].reshape(bsz, -1, *fs[1].shape[-2:])
            att_ = F.interpolate(att_, size=fs[0].shape[-2:],mode="bilinear").flatten(-2).permute(0,2,1).reshape(bsz, -1, *fs[1].shape[-2:])
            att_ = F.interpolate(att_, size=fs[0].shape[-2:],mode="bilinear").flatten(-2).permute(0,2,1)    

            if not self.reverse:
                att_ = att_.permute(0,2,1)
                
            losses['dist_loss'] = self.loss(att_, atts[0][:,0].detach()) if self.detach else \
                self.loss(att_, atts[0][:,0]) 
        
            
        vis_results = dict(mask=sparse_mask[0,0], imgs=imgs[0,0])

        return losses, vis_results


    def prep(self, image, downsample_rate):
        _,c,_,_ = image.size()
        x = image.float()[:,:,::downsample_rate,::downsample_rate]

        return x