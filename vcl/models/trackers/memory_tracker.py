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

from vcl.models.common.correlation import *
from vcl.models.common.hoglayer import *

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
                 per_ref=True,
                 head=None,
                 downsample_rate=4,
                 radius=12,
                 temperature=1,
                 feat_size=64,
                 cos_loss=None,
                 scaling=True,
                 upsample=True,
                 weight=20,
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
        self.scaling = scaling
        self.upsample = upsample
        self.weight = weight
        self.temperature = temperature
        self.cos_loss = build_loss(cos_loss) if cos_loss is not None else None
        
        if isinstance(radius, list):
            masks = []
            for r in radius:
                masks.append(make_mask(feat_size, r))
            self.mask = torch.stack(masks, 0)
        else:
            self.mask = make_mask(feat_size, radius)
        
        self.R = radius # radius
        
        self.init_weights()

    def init_weights(self):
        self.backbone.init_weights()
        if self.pretrained != None:
            _ = load_checkpoint(self, self.pretrained, strict=False, map_location='cpu')
    
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
        _, att = non_local_attention(tar, refs, mask=self.mask, scaling=self.scaling, per_ref=self.per_ref, temprature=self.temperature)        
        
        losses = {}
        if self.cos_loss !=  None:
            att_cos = non_local_attention(tar, refs, att_only=True)
            losses['cos_loss'] = self.cos_loss(att_cos)
        
        # for mast l1_loss
        ref_gt = [self.prep(gt[:,ch]) for gt in images_lab_gt[:-1]]
        outputs = frame_transform(att, ref_gt, flatten=False, per_ref=self.per_ref)
        if self.per_ref:
            outputs = outputs[:,0].permute(0,2,1).reshape(bsz, -1, *fs.shape[-2:])     
        else:
            outputs = outputs.permute(0,2,1).reshape(bsz, -1, *fs.shape[-2:])     
            
        losses['l1_loss'], err_map = self.compute_lphoto(images_lab_gt, ch, outputs, upsample=self.upsample)
        
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
    
    def prep(self, image, mode='default'):
        _,c,_,_ = image.size()

        x = image.float()[:,:,::self.downsample_rate,::self.downsample_rate]

        return x
    
    
@MODELS.register_module()
class Memory_Tracker_Custom_Pyramid(Memory_Tracker_Custom):
    def __init__(self,
                loss=None,
                pool_type='mean',
                bilinear_downsample=True,
                reverse=True,
                num_stage=2,
                feat_size=[64, 32],
                radius=[12, 6],
                downsample_rate=[4, 8],
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
        self.feat_size = feat_size
        self.downsample_rate = downsample_rate
        self.loss = build_loss(loss) if loss is not None else None
        self.pool_type = pool_type
        self.bilinear_downsample = bilinear_downsample
        self.detach = detach
        
        self.mask = [ make_mask(feat_size[i], radius[i]) for i in range(len(radius)) if radius[i] != -1]


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
            ref_gt = [self.prep(gt[:,ch], downsample_rate=self.downsample_rate[idx]) for gt in images_lab_gt[:-1]]
            outputs = frame_transform(att, ref_gt, flatten=False)
            outputs = outputs[:,0].permute(0,2,1).reshape(bsz, -1, self.feat_size[idx], self.feat_size[idx])     
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
        
        if not isinstance(self.R, list):
            self.corr = Correlation(max_displacement=self.R)
        else:
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
        corr = self.corr(tar, refs[:,0]) 
        if self.scaling:
            corr = corr / torch.sqrt(torch.tensor(tar.shape[1]).float()) 
        corr = corr.flatten(1,2).softmax(1)
        
        losses = {}
        if self.cos_loss !=  None:
            att_cos = non_local_attention(tar, refs, att_only=True)
            losses['cos_loss'] = self.cos_loss(att_cos)
        
        # for mast l1_loss
        ref_gt = self.prep(images_lab_gt[0][:,ch])
        ref_gt = F.unfold(ref_gt, self.R * 2 + 1, padding=self.R)
        
        corr = corr.reshape(bsz, -1, tar.shape[-1]**2)
        outputs = (corr * ref_gt).sum(1).reshape(bsz, -1, *fs.shape[-2:])        
        losses['l1_loss'], err_map = self.compute_lphoto(images_lab_gt, ch, outputs, upsample=self.upsample)
        
        vis_results = dict(err=err_map[0], imgs=imgs[0,0])

        return losses, vis_results
    

@MODELS.register_module()
class Memory_Tracker_Custom_Pyramid_V2(Memory_Tracker_Custom_V2):
    def __init__(self,
                loss=None,
                bilinear_downsample=True,
                reverse=True,
                num_stage=2,
                feat_size=[64, 32],
                downsample_rate=[4, 8],
                detach=False,
                pool_type='mean',
                *args,
                **kwargs
                 ):
        """ MAST  (CVPR2020) Pyramid using MMCV Correlation Module

        Args:
            backbone ([type]): [description]
            test_cfg ([type], optional): [description]. Defaults to None.
            train_cfg ([type], optional): [description]. Defaults to None.
        """
        super().__init__(*args, **kwargs)
        self.reverse = reverse
        self.num_stage = num_stage
        self.feat_size = feat_size
        self.downsample_rate = downsample_rate
        self.loss = build_loss(loss) if loss is not None else None
        self.bilinear_downsample = bilinear_downsample
        self.detach = detach
        self.pool_type = pool_type
        
        if self.pool_type == 'max':
            self.pool = nn.AdaptiveMaxPool2d((self.R[-1] *2 +1, self.R[-1] *2 +1))
        else:
            self.pool = nn.AdaptiveAvgPool2d((self.R[-1] *2 +1, self.R[-1] *2 +1))
    

    def forward_train(self, images_lab, imgs=None):
            
        bsz, _, n, c, h, w = images_lab.shape
        
        images_lab_gt = [images_lab[:,0,i].clone() for i in range(n)]
        images_lab = [images_lab[:,0,i] for i in range(n)]
        _, ch = self.dropout2d_lab(images_lab)
        
        # forward to get feature
        fs = self.backbone(torch.stack(images_lab,1).flatten(0,1))
        fs = [f.reshape(bsz, n, *f.shape[-3:]) for f in fs]
        
        tar_pyramid, refs_pyramid = [f[:, -1] for f in fs], [ f[:, :-1] for f in fs]
        
        losses = {}
        
        corrs = []
        for idx, (tar, refs) in enumerate(zip(tar_pyramid, refs_pyramid)):
            # get correlation attention map      
            corr = self.corr[idx](tar, refs[:,0]) 
            if self.scaling:
                corr = corr / torch.sqrt(torch.tensor(tar.shape[1]).float()) 
            corr = corr.flatten(1,2).softmax(1)
            
            # for mast l1_loss
            ref_gt = self.prep(images_lab_gt[0][:,ch], downsample_rate=self.downsample_rate[idx])
            ref_gt = F.unfold(ref_gt, self.R[idx] * 2 + 1, padding=self.R[idx])
            
            corr = corr.reshape(bsz, -1, self.feat_size[idx]**2)
            outputs = (corr * ref_gt).sum(1).reshape(bsz, -1, self.feat_size[idx], self.feat_size[idx])        
            losses[f'stage{idx}_l1_loss'], err_map = self.compute_lphoto(images_lab_gt, ch, outputs, upsample=self.upsample)
            
            corrs.append(corr) 
            
        if self.bilinear_downsample:
            if not self.reverse:
                corrs[0] = corrs[0].permute(0,1,3,2)
            if self.pool_type == 'mean':
                att_ = corrs[0].permute(0,2,1).reshape(bsz, -1, 2 * self.R[0] + 1, 2 * self.R[0] + 1)
                att_ = self.pool(att_).flatten(-2).permute(0,2,1).reshape(bsz, -1, *fs[0].shape[-2:])
                target = F.avg_pool2d(att_, 2, stride=2).flatten(-2)   
            elif self.pool_type == 'max':
                att_ = corrs[0].permute(0,2,1).reshape(bsz, -1, 2 * self.R[0] + 1, 2 * self.R[0] + 1)
                att_ = self.pool(att_).flatten(-2).permute(0,2,1).reshape(bsz, -1, *fs[0].shape[-2:])
                target = F.max_pool2d(att_, 2, stride=2).flatten(-2)

            if not self.reverse:
                target = target.permute(0,2,1)
                
            losses['dist_loss'] = self.loss(corrs[-1], target.detach()) if self.detach else \
                self.loss(corrs[-1][:,0], target) 
        else:
            raise NotImplemented
            
        vis_results = dict(err=err_map[0], imgs=imgs[0,0])

        return losses, vis_results
    
    def prep(self, image, downsample_rate):
        _,c,_,_ = image.size()
        x = image.float()[:,:,::downsample_rate,::downsample_rate]

        return x
    
    
@MODELS.register_module()
class Memory_Tracker_Custom_Pyramid_Cmp(Memory_Tracker_Custom):
    def __init__(self,
                loss=None,
                cmp_loss=None,
                pool_type='mean',
                bilinear_downsample=True,
                reverse=True,
                T=0.2,
                num_stage=2,
                feat_size=[64, 32],
                radius=[12, 6],
                downsample_rate=[4, 8],
                output_dim=-1,
                detach=False,
                rand_mask=False,
                inpaint_only=False,
                mode='classification',
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
        self.rand_mask = rand_mask
        self.radius = radius
        self.ouput_dim = (2 * radius[-1] + 1) ** 2 if output_dim == -1 else output_dim
        self.T = T
        self.inpaint_only = inpaint_only
        self.mode = mode
        
        self.mask = [ make_mask(feat_size[i], radius[i]) for i in range(len(radius)) if radius[i] != -1]        
        self.cmp_loss = build_loss(cmp_loss) if cmp_loss != None else None
        self.flow_decoder = MotionDecoderPlain(
        input_dim=self.backbone.feat_dim+self.ouput_dim+1,
        output_dim=self.ouput_dim,
        combo=[1,2,4])
        
        self.corr = [Correlation(max_displacement=R) for R in radius ]
        
        if self.pool_type == 'max':
            self.pool = nn.AdaptiveMaxPool2d((radius[-1] *2 +1, radius[-1] *2 +1))
        else:
            self.pool = nn.AdaptiveAvgPool2d((radius[-1] *2 +1, radius[-1] *2 +1))
        
        
    def forward_train(self, imgs, images_lab=None):
            
        bsz, num_clips, t, c, h, w = images_lab.shape
        
        images_lab_gt = [images_lab[:,0,i,:].clone() for i in range(t)]
        images_lab = [images_lab[:,0,i,:] for i in range(t)]
        _, ch = self.dropout2d_lab(images_lab)
        
        # forward to get feature
        fs = self.backbone(torch.stack(images_lab,1).flatten(0,1))
        fs = [f.reshape(bsz, t, *f.shape[-3:]) for f in fs]
        tar_cmp = self.backbone(images_lab_gt[-1])[-1]
        
        tar_pyramid, refs_pyramid = [f[:, -1] for f in fs], [ f[:, :-1] for f in fs]
        
        losses = {}
        
        atts = []
        corrs = []
        for idx, (tar, refs) in enumerate(zip(tar_pyramid, refs_pyramid)):
            if idx == len(tar_pyramid) - 1: break
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
        
        # for cmp loss
        corr_feat = corrs[-1].reshape(bsz, -1, self.feat_size[-1], self.feat_size[-1])
        
        if not self.rand_mask:
            corr = corr_feat.softmax(1)
            corr_en = (-torch.log(corr+1e-12)).sum(1).flatten(-2)
            corr_sorted, _ = torch.sort(corr_en, dim=-1, descending=True)
            idx = int(corr_en.shape[-1] * self.T) - 1
            T = corr_sorted[:, idx:idx+1]
            sparse_mask = (corr_en > T).reshape(bsz, 1, *corr_feat.shape[-2:]).float()
            
        else:
            sample_idx = torch.randint(0, self.feat_size[-1] ** 2 -1, (int(self.feat_size[-1] ** 2 * self.T),)).cuda()
            sparse_mask = torch.zeros(bsz, self.feat_size[-1] ** 2).cuda()
            sparse_mask[:, sample_idx] = 1
            sparse_mask = sparse_mask.reshape(bsz, 1, *corr_feat.shape[-2:])
        
        
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
        
        concat_input = torch.cat([tar_cmp, corr_feat*sparse_mask, sparse_mask], 1)
        corr_predict = self.flow_decoder(concat_input).flatten(-2)
        
        # for cmp loss target
        if self.mode == 'classification':
            corr_predict = corr_predict.permute(0,2,1).reshape(-1, self.ouput_dim)
            if not self.inpaint_only:
                target = corrs[0].reshape(bsz, -1, self.feat_size[0], self.feat_size[0])
                target = F.avg_pool2d(target, 2, stride=2).flatten(-2).permute(0,2,1).reshape(bsz, -1, 2*self.radius[0]+1, 2*self.radius[0]+1)
                target = self.pool(target).reshape(-1, self.ouput_dim)
                target = target.max(-1)[1].detach().reshape(-1, 1)
            else:
                target = corrs[1].reshape(bsz, -1, self.feat_size[-1] ** 2).permute(0,2,1)
                target = target.max(-1)[1].reshape(-1, 1)
        
            losses['cmp_loss'] = self.cmp_loss(corr_predict, target)
        elif self.mode == 'rec':
            ref_gt = self.prep(images_lab_gt[0][:,ch], self.downsample_rate[-1])
            ref_gt = F.unfold(ref_gt, self.radius[-1] * 2 + 1, padding=self.radius[-1])
            corr_predict = corr_predict.reshape(bsz, tar.shape[-1]**2, -1).permute(0,2,1)
            outputs = (corr_predict * ref_gt).sum(1).reshape(bsz, -1, *tar_cmp.shape[-2:])        
            losses['cmp_loss'] = 0.1 * self.compute_lphoto(images_lab_gt, ch, outputs, upsample=self.upsample)[0]
        elif self.mode == 'l1_loss':
            target = corrs[0].reshape(bsz, -1, self.feat_size[0], self.feat_size[0])
            target = F.avg_pool2d(target, 2, stride=2).flatten(-2).permute(0,2,1).reshape(bsz, -1, 2*self.radius[0]+1, 2*self.radius[0]+1)
            target = self.pool(target).detach().flatten(-2).permute(0,2,1)
            losses['cmp_loss'] = self.cmp_loss(corr_predict, target)
        else:
            raise NotImplemented
            
        vis_results = dict(mask=sparse_mask[0,0], imgs=imgs[0,0])

        return losses, vis_results


    def prep(self, image, downsample_rate):
        _,c,_,_ = image.size()
        x = image.float()[:,:,::downsample_rate,::downsample_rate]

        return x