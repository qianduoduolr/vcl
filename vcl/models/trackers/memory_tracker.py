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
        self.weight = weight
        self.temperature = temperature
        self.forward_backward_t = forward_backward_t
        self.conc_loss = build_loss(conc_loss) if conc_loss is not None else None
        
        if len(radius) > 1:
            self.mask = []
            for idx, r in enumerate(radius):
                self.mask.append(make_mask(feat_size[idx], r))
        else:
            self.mask = [ make_mask(feat_size[0], radius[0]) ]
        
        self.R = radius # radius in previous
        self.radius = radius 
        self.feat_size = feat_size

        self.loss_weight = loss_weight

        assert len(self.feat_size) == len(self.radius) == len(self.downsample_rate) 
        
        self.init_weights()

    def init_weights(self):

        logger = get_root_logger()
        if self.pretrained is not None:
            _ = load_checkpoint(self, self.pretrained, logger=logger, map_location='cpu')
        else:
            pass

    
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
    
    def compute_lphoto(self, images_lab_gt, ch, outputs, upsample=True, mask=None):
        b, c, h, w = images_lab_gt[0].size()

        tar_y = images_lab_gt[-1][:,ch]  # y4

        if upsample:
            outputs = F.interpolate(outputs, (h, w), mode='bilinear')
            loss = F.smooth_l1_loss(outputs * self.weight, tar_y * self.weight, reduction='mean')
        else:
            tar_y = self.prep(images_lab_gt[-1])[:,ch]
            loss = F.smooth_l1_loss(outputs * self.weight, tar_y * self.weight, reduction='none')
            if mask == None:
                loss = loss.mean()
            else:
                loss = (loss * mask).sum() / (mask.sum() + 1e-9)

        err_maps = torch.abs(outputs - tar_y).sum(1).detach()

        return loss, err_maps
    
    def prep(self, image, downsample_rate=8, mode='default'):
        _,c,_,_ = image.size()

        x = image.float()[:,:,::downsample_rate,::downsample_rate]

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
class Memory_Tracker_Custom_Crop(Memory_Tracker_Custom):

    def __init__(self, backbone_a, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.backbone_a = build_backbone(backbone_a)
        self.corr = Correlation(max_displacement=self.radius[0])

    
    def forward_train(self, images_lab, images_lab_crop, affine_imgs_crop):
            
        losses = {}

        # main branch
        loss_main, corr_main = self.forward_frame_reconstruction(self.backbone_a, images_lab, 'main')
        # crop branch
        loss_crop, corr_crop = self.forward_frame_reconstruction(self.backbone, images_lab_crop, 'crop')

        grid = F.affine_grid(affine_imgs_crop[:,0], corr_main.size())
        corr_gt = F.grid_sample(corr_main, grid).flatten(-2).permute(0,2,1).reshape_as(corr_crop)
        corr_gt = F.grid_sample(corr_gt, grid).flatten(-2).permute(0,2,1).reshape_as(corr_crop).detach()


        losses['main_l1_loss'] = self.loss_weight['main_l1_loss'] * loss_main
        losses['crop_l1_loss'] = self.loss_weight['crop_l1_loss'] * loss_crop
        losses['reg_loss'] = self.loss_weight['reg_loss'] * build_loss(dict(type='MSELoss'))(corr_crop, corr_gt)

        vis_results = dict(err=None, imgs=images_lab[0,0])

        return losses, vis_results

    def forward_frame_reconstruction(self, backbone, images_lab, mode='main'):

        bsz, _, n, c, h, w = images_lab.shape

        images_lab_gt = [images_lab[:,0,i].clone() for i in range(n)]
        images_lab = [images_lab[:,0,i] for i in range(n)]
        _, ch = self.dropout2d_lab(images_lab)

        # forward to get feature
        fs = backbone(torch.stack(images_lab,1).flatten(0,1))
        if self.head is not None:
            fs = self.head(fs)
        
        fs = fs.reshape(bsz, n, *fs.shape[-3:])

        tar, refs = fs[:, -1], fs[:, :-1]
        
        # get correlation attention map      
        _, att = non_local_attention(tar, refs, mask=self.mask[0], scaling=self.scaling, per_ref=self.per_ref, temprature=self.temperature)    

        # corr = self.corr(tar, refs[0]).flatten(1,2)
        
        if self.forward_backward_t != -1:
            m = self.forward_backward_check(att)
        else:
            m = None
        
        losses = {}
        if self.conc_loss !=  None:
            att_cos = non_local_attention(tar, refs, att_only=True)
            losses['conc_loss'] = self.conc_loss(att_cos)
        
        # for mast l1_loss
        outputs = self.frame_reconstruction(images_lab_gt, att, ch, feat_size=self.feat_size[0], downsample_rate=self.downsample_rate[0]) 
        loss = self.compute_lphoto(images_lab_gt, ch, outputs, upsample=self.upsample, mask=m)[0]

        return loss, att[:,0].permute(0,2,1).reshape(bsz, -1, *tar.shape[-2:])

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
class Memory_Tracker_Custom_Cmp(Memory_Tracker_Custom):
    def __init__(self,
                motion_estimator,
                loss=None,
                cmp_loss=None,
                pool_type='mean',
                bilinear_downsample=True,
                reverse=True,
                T=-1,
                vae_var=1,
                num_stage=2,
                output_dim=-1,
                detach=False,
                rand_mask=False,
                inpaint_only=False,
                mp_only=False,
                boundary_r=-1,
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
        
        self.motion_estimator = build_backbone(motion_estimator)

        self.reverse = reverse
        self.num_stage = num_stage
        self.loss = build_loss(loss) if loss is not None else None
        self.pool_type = pool_type
        self.bilinear_downsample = bilinear_downsample
        self.detach = detach
        self.ouput_dim = (2 * self.radius[-1] + 1) ** 2 if output_dim == -1 else output_dim
        self.T = T
        self.mode = mode
        self.vae_var = vae_var
        self.mp_only = mp_only
        
        self.cmp_loss = build_loss(cmp_loss) if cmp_loss != None else None
        
        # flow decoder
        if self.mode.find('vae') != -1:
            self.flow_decoder = MotionDecoderPlain(
            input_dim=self.backbone.feat_dim,
            output_dim=self.ouput_dim,
            combo=[1,2,4])

            self.flow_decoder_m = MotionDecoderPlain(
            input_dim=self.ouput_dim // 2,
            output_dim=self.ouput_dim // 2,
            combo=[1,2,4])

        else:
            self.flow_decoder = MotionDecoderPlain(
            input_dim=self.backbone.feat_dim,
            output_dim=self.ouput_dim,
            combo=[1,2,4])

        self.corr = [Correlation(max_displacement=R) for R in self.radius ]
        
        if self.pool_type == 'max':
            self.pool = nn.AdaptiveMaxPool2d((self.radius[-1] *2 +1, self.radius[-1] *2 +1))
        else:
            self.pool = nn.AdaptiveAvgPool2d((self.radius[-1] *2 +1, self.radius[-1] *2 +1))

        if boundary_r is not -1:
            self.boundary_mask = torch.zeros((self.feat_size[-1], self.feat_size[-1])).cuda()
            self.boundary_mask[boundary_r:-boundary_r, boundary_r:-boundary_r] = 1
            self.boundary_mask = self.boundary_mask.reshape(1,1,-1)
        else:
            self.boundary_mask = None
        
        
    def forward_train(self, imgs, images_lab=None):
            
        bsz, num_clips, t, c, h, w = images_lab.shape
        
        losses = {}
        atts = []
        corrs = []

        images_lab_gt = [images_lab[:,0,i,:].clone() for i in range(t)]
        images_lab = [images_lab[:,0,i,:] for i in range(t)]
        _, ch = self.dropout2d_lab(images_lab)
        
        ################ Feature/Correlation Extraction #####################
        # estimate motion
        with torch.no_grad():
            fs = self.motion_estimator(torch.stack(images_lab_gt,1).flatten(0,1))
            fs = fs.reshape(bsz, t, *fs.shape[-3:])
            
            if fs.shape[-1] > 32:
                # apply corr at higher resolution
                corr_idx_t = 0
            else:
                corr_idx_t = -1
            
            _, att = non_local_attention(fs[:,-1], fs[:, :-1], mask=self.mask[corr_idx_t], scaling=True)   
            corr = self.corr[corr_idx_t](fs[:,-1], fs[:, 0])

            if corr_idx_t == 0:
                att = self.corr_downsample(att, bsz, mode='custom')
                corr = self.corr_downsample(corr, bsz, mode='mmcv')
                
            atts.append(att.detach())
            corrs.append(corr.detach())

        # forward to get feature
        fs = self.backbone(torch.stack(images_lab,1).flatten(0,1))[:-1]
        tar_cmp = self.backbone(images_lab_gt[-1])[-1] # to fix

        if isinstance(fs, tuple): fs = tuple(fs)

        ######################### Frame Reconstruction #############################
        # pyramid (optional) frame reconstruction
        if not self.mp_only:
            for idx, f in enumerate(fs):
                f = f.reshape(bsz, t, *f.shape[-3:])
                # get correlation attention map            
                _, att = non_local_attention(f[:,-1], f[:,:-1], mask=self.mask[idx], scaling=True)            
                # for mast l1_loss
                outputs = self.frame_reconstruction(images_lab_gt, att, ch, self.feat_size[idx], self.downsample_rate[idx])   
                losses[f'stage{idx}_l1_loss'] = self.loss_weight[f'stage{idx}_l1_loss'] * self.compute_lphoto(images_lab_gt, ch, outputs)[0]

                if idx == len(self.mask) - 1:
                    corr = self.corr[idx](f[:,-1], f[:,0])
                    atts.append(att)
                    corrs.append(corr)
                else:
                    att_hr = self.corr_downsample(att, bsz, mode='custom').detach()

        # exp corr loss
        if self.loss_weight.get('corr_loss', 0) != 0:
            if corr_idx_t == 0:
                losses['corr_loss'] = self.loss_weight['corr_loss'] * build_loss(dict(type='L1Loss'))(atts[-1][:,0], atts[0])
            else:
                losses['corr_loss'] = self.loss_weight['corr_loss'] * build_loss(dict(type='L1Loss'))(corrs[-1], corrs[0])

        # local distillation loss
        if self.T != -1:
            corr_feat = corrs[-1].reshape(bsz, -1, self.feat_size[-1], self.feat_size[-1])
            corr = corr_feat.softmax(1)
            corr_en = (-torch.log(corr+1e-12)).sum(1).flatten(-2)
            corr_sorted, _ = torch.sort(corr_en, dim=-1, descending=True)
            idx = int(corr_en.shape[-1] * self.T) - 1
            T = corr_sorted[:, idx:idx+1]
            sparse_mask = (corr_en > T).reshape(bsz, 1, *corr_feat.shape[-2:]).float().detach()
            weight = sparse_mask.flatten(-2).permute(0,2,1).repeat(1, 1, atts[-1].shape[-1])
            weight_mp = sparse_mask.repeat(1, self.ouput_dim // 2, 1, 1)

        else:
            weight = None
            sparse_mask = None
            weight_mp = None

        if self.loss_weight.get('local_corr_dist_loss', 0) != 0 and len(self.mask) > 1:
            losses['local_corr_dist_loss'] = self.loss_weight['local_corr_dist_loss'] * build_loss(dict(type='MSELoss'))(att[:,0], att_hr, weight)


        #################### Motion Prediction #######################
        # for cmp loss   
        if self.mode.find('vae') == -1:     
            corr_predict = self.flow_decoder(tar_cmp).flatten(-2)
        
        # for cmp loss target 
        if self.mode == 'exp':
            losses.update(self.forward_exp(corrs, corr_predict))
        elif self.mode.find('vae') != -1:
            losses.update(self.forward_vae(corrs, tar_cmp, images_lab_gt, ch, weight=weight_mp))
        elif self.mode == 'regression':
            losses.update(self.forward_regression(corr_predict, atts))

        vis_results = dict(mask=sparse_mask[0,0] if sparse_mask != None else None, imgs=imgs[0,0])

        return losses, vis_results

    def prep(self, image, downsample_rate):
        _,c,_,_ = image.size()
        x = image.float()[:,:,::downsample_rate,::downsample_rate]

        return x

    def reparameterise(self, mu, logvar):
        epsilon = torch.randn_like(mu)
        return mu + epsilon * torch.exp(logvar / 2)

    def corr_downsample(self, corr, bsz, pool_type='mean', mode='custom'):
        if mode == 'custom':

            if pool_type == 'mean':
                corr = corr.reshape(bsz, -1, self.feat_size[0], self.feat_size[0])
                corr = F.avg_pool2d(corr, 2, stride=2).flatten(-2).permute(0,2,1).reshape(bsz, -1, self.feat_size[0], self.feat_size[0])
                corr_down = F.avg_pool2d(corr, 2, stride=2).flatten(-2).permute(0,2,1)    
            elif pool_type == 'max':
                corr = corr.reshape(bsz, -1, self.feat_size[0], self.feat_size[0])
                corr = F.max_pool2d(corr, 2, stride=2).flatten(-2).permute(0,2,1).reshape(bsz, -1, self.feat_size[0], self.feat_size[0])
                corr_down = F.max_pool2d(corr, 2, stride=2).flatten(-2).permute(0,2,1)

        elif mode == 'mmcv':
            corr = corr.reshape(bsz, (2*self.radius[0]+1) ** 2, -1)

            if self.pool_type == 'mean':
                corr = corr.permute(0,2,1).reshape(bsz, -1, 2 * self.radius[0] + 1, 2 * self.radius[0] + 1)
                corr = self.pool(corr).flatten(-2).permute(0,2,1).reshape(bsz, -1, self.feat_size[0], self.feat_size[0])
                corr_down = F.avg_pool2d(corr, 2, stride=2).flatten(-2)   
            elif self.pool_type == 'max':
                corr = corr.permute(0,2,1).reshape(bsz, -1, 2 * self.radius[0] + 1, 2 * self.radius[0] + 1)
                corr = self.pool(corr).flatten(-2).permute(0,2,1).reshape(bsz, -1, self.feat_size[0], self.feat_size[0])
                corr_down = F.max_pool2d(corr, 2, stride=2).flatten(-2)

            
            corr_down = corr_down.reshape(bsz, (2*self.radius[-1]+1), -1, self.feat_size[-1], self.feat_size[-1])
        
        return corr_down

    def forward_vae(self, corrs, tar_cmp, images_lab_gt, ch, weight):

        bsz = tar_cmp.shape[0]
        corr_predict = self.flow_decoder(tar_cmp)
        # VAE prior loss
        mu_pred = corr_predict[:,:self.ouput_dim // 2]
        logvar_pred = corr_predict[:,self.ouput_dim // 2:]

        losses = {}

        if self.mode.find('learnt_prior') != -1:
            mu_tar = corrs[0].reshape(bsz, (2*self.radius[-1]+1) ** 2, *tar_cmp.shape[-2:])
            sampled_corr = self.reparameterise(mu_pred, logvar_pred)

            corr_predict = self.flow_decoder_m(sampled_corr)
            
        elif self.mode.find('fix_prior') != -1:
            mu_tar = torch.zeros_like(mu_pred)
            sampled_corr = self.reparameterise(mu_pred, logvar_pred)
            corr_predict = self.flow_decoder_m(sampled_corr)

            # additional exp loss
            losses['cmp_loss'] = self.loss_weight['cmp_loss'] *  build_loss(dict(type='L1Loss'))(corr_predict.flatten(-2), corrs[0].reshape(bsz, (2*self.radius[0]+1) ** 2, -1))

        ref_gt = self.prep(images_lab_gt[0][:,ch], self.downsample_rate[-1])
        ref_gt = F.unfold(ref_gt, self.radius[-1] * 2 + 1, padding=self.radius[-1])
        outputs = (corr_predict.flatten(-2) * ref_gt).sum(1).reshape(bsz, -1, *tar_cmp.shape[-2:]) 

        logvar_tar = torch.log(torch.ones_like(logvar_pred) * self.vae_var) # var set to 1

        losses['vae_kl_loss'] = self.loss_weight['vae_kl_loss'] * build_loss(dict(type='Kl_Loss_Gaussion'))([mu_pred, logvar_pred], [mu_tar, logvar_tar], weight=weight)

        losses['vae_rec_loss'] = self.loss_weight['vae_rec_loss'] * self.compute_lphoto(images_lab_gt, ch, outputs, upsample=self.upsample)[0]

        return losses


    def forward_classification(self, corr_predict, corrs):
        bsz = corr_predict.shape[0]
        corr_predict = corr_predict.permute(0,2,1).reshape(-1, self.ouput_dim)
        target = corrs[0].reshape(bsz, (2*self.radius[-1]+1) ** 2, -1).permute(0, 2, 1).reshape(-1, self.ouput_dim)

        losses = {}
        if self.mode.find('soft') == -1:
            target = target.max(-1)[1].detach().reshape(-1, 1)
        else:
            target = target.detach()
    
        losses['cmp_loss'] = self.loss_weight['cmp_loss'] * self.cmp_loss(corr_predict, target)

        return losses

    def forward_regression(self, corr_predict, atts):
        losses = {}

        h = w = self.feat_size[1]
        target = atts[0]
        off = att2flow(h, w, target)
        corr_predict = corr_predict.permute(0,2,1)
        
        losses['cmp_loss'] = self.loss_weight['cmp_loss'] * F.smooth_l1_loss(corr_predict, off, reduction='mean')
        return losses

    def forward_exp(self, corrs, corr_predict):
        bsz = corrs.shape[0]
        losses = {}
        target = corrs[0].reshape(bsz, (2*self.radius[-1]+1) ** 2, -1)
        losses['cmp_loss'] = self.loss_weight['cmp_loss'] * self.cmp_loss(corr_predict, target, weight=self.boundary_mask)
        return losses