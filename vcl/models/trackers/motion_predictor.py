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

from .base import BaseModel
from ..builder import build_backbone, build_components, build_loss
from ..registry import MODELS
from vcl.utils import *

import torch.nn as nn
import torch.nn.functional as F
from .modules import *
from .memory_tracker import *

@MODELS.register_module()
class Memory_Tracker_Custom_Cmp(Memory_Tracker_Custom):
    def __init__(self,
                motion_estimator=None,
                loss=None,
                cmp_loss=None,
                pool_type='mean',
                bilinear_downsample=True,
                reverse=True,
                temperature_t=1.0,
                scale_t=True,
                norm_t=False,
                pre_softmax=False,
                T=-1,
                vae_var=1,
                num_stage=2,
                output_dim=-1,
                detach=False,
                mp_only=False,
                boundary_r=-1,
                mode='classification',
                modality='LAB',
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
        
        self.motion_estimator = build_backbone(motion_estimator) if motion_estimator is not None else None

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
        self.modality = modality
        self.cmp_loss = build_loss(cmp_loss) if cmp_loss != None else None

        self.temperature_t = temperature_t
        self.norm_t = norm_t
        self.scale_t = scale_t
        self.pre_softmax = pre_softmax
        
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

        elif self.mode.find('exp') != -1:
            self.flow_decoder = MotionDecoderPlain(
            input_dim=self.backbone.feat_dim,
            output_dim=self.ouput_dim,
            combo=[1,2,4])

            self.flow_decoder_m = MotionDecoderPlain(
            input_dim=self.ouput_dim,
            output_dim=self.ouput_dim,
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

        images_lab_gt = [images_lab[:,0,i,:].clone() for i in range(t)]
        images_lab = [images_lab[:,0,i,:] for i in range(t)]
        _, ch = self.dropout2d_lab(images_lab)
        
        ################ Feature/Correlation Extraction #####################
        # estimate motion 
        with torch.no_grad():
            fs = self.motion_estimator(torch.stack(images_lab_gt,1).flatten(0,1))
            fs = fs.reshape(bsz, t, *fs.shape[-3:])
            if self.norm_t:
                fs = F.normalize(fs, dim=2)
            
            if fs.shape[-1] > self.feat_size[-1]:
                # apply corr at higher resolution
                corr_idx_t = 0
                mask = make_mask(self.feat_size[-1] * 2, self.radius[-1] * 2)
            else:
                corr_idx_t = -1
                mask = self.mask[corr_idx_t]
            
            att_m = non_local_attention(fs[:,-1], fs[:, :-1], mask=mask, scaling=self.scale_t, \
                temprature=self.temperature_t)[-1].detach()
            corr_m = self.corr[corr_idx_t](fs[:,-1], fs[:, 0]).detach()

            if self.pre_softmax:
                corr_m = torch.div(corr_m, self.temperature_t).softmax(1)

            if corr_idx_t == 0:
                att_m = self.corr_downsample(att_m, bsz, mode='custom')
                corr_m = self.corr_downsample(corr_m, bsz, mode='mmcv')
                
        # forward to get feature
        fs = self.backbone(torch.stack(images_lab,1).flatten(0,1))[:-1]

        if self.modality == 'LAB':
            tar_cmp = self.backbone(images_lab_gt[-1])[-1] # to fix
        else:
            tar_cmp = self.backbone(imgs[:,0,-1])[-1]

        if isinstance(fs, tuple): fs = tuple(fs)
        else: fs = [fs]

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
                else:
                    att_hr = self.corr_downsample(att, bsz, mode='custom').detach()
        else:
            f = fs[-1].reshape(bsz, t, *fs[-1].shape[-3:])
            corr = self.corr[-1](f[:,-1], f[:,0])


        # local distillation loss
        if self.T != -1:
            corr_feat = corr.reshape(bsz, -1, self.feat_size[-1], self.feat_size[-1])
            corr = corr_feat.softmax(1)
            corr_en = (-torch.log(corr+1e-12)).sum(1).flatten(-2)
            corr_sorted, _ = torch.sort(corr_en, dim=-1, descending=True)
            idx = int(corr_en.shape[-1] * self.T) - 1
            T = corr_sorted[:, idx:idx+1]
            sparse_mask = (corr_en > T).reshape(bsz, 1, *corr_feat.shape[-2:]).float().detach()
            weight = sparse_mask.flatten(-2).permute(0,2,1).repeat(1, 1, att.shape[-1])
            weight_mp = sparse_mask.repeat(1, self.ouput_dim // 2, 1, 1)

        else:
            weight = None
            sparse_mask = None
            weight_mp = None

        if self.loss_weight.get('local_corr_dist_loss', 0) != 0 and len(self.mask) > 1:
            losses['local_corr_dist_loss'] = self.loss_weight['local_corr_dist_loss'] * build_loss(dict(type='MSELoss'))(att[:,0], att_hr, weight)

        #################### Motion Prediction #######################        
        # for cmp loss target 
        if self.mode == 'exp':
            target = corr_m.reshape(bsz, (2*self.radius[-1]+1) ** 2, -1)
            losses.update(self.forward_exp(target, images_lab_gt, ch, tar_cmp))
        elif self.mode.find('vae') != -1:
            losses.update(self.forward_vae(corr_m, tar_cmp, images_lab_gt, ch, weight=weight_mp))
        elif self.mode == 'regression':
            losses.update(self.forward_regression(tar_cmp, [att_m, att]))

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

    def forward_vae(self, target, tar_cmp, images_lab_gt, ch, weight):

        bsz = tar_cmp.shape[0]
        corr_predict = self.flow_decoder(tar_cmp)
        # VAE prior loss
        mu_pred = corr_predict[:,:self.ouput_dim // 2]
        logvar_pred = corr_predict[:,self.ouput_dim // 2:]

        if weight == None and self.boundary_mask is not None:
            weight_kl  = self.boundary_mask.reshape(1, 1, *tar_cmp.shape[-2:])
            weight_kl_prior = weight_kl.repeat(bsz, self.ouput_dim // 2, 1, 1)
            weight_rec = weight_kl.repeat(bsz, 1, 1, 1)
        else:
            weight_kl_prior = weight
            weight_rec = None

        losses = {}

        if self.mode.find('learnt_prior') != -1:
            mu_tar = target.reshape(bsz, (2*self.radius[-1]+1) ** 2, *tar_cmp.shape[-2:])
            sampled_corr = self.reparameterise(mu_pred, logvar_pred)
            corr_predict = self.flow_decoder_m(sampled_corr)
            
        elif self.mode.find('fix_prior') != -1:
            mu_tar = torch.zeros_like(mu_pred)
            sampled_corr = self.reparameterise(mu_pred, logvar_pred)
            corr_predict = self.flow_decoder_m(sampled_corr)

            # additional exp loss
            losses['cmp_loss'] = self.loss_weight['cmp_loss'] *  build_loss(dict(type='L1Loss'))(corr_predict.flatten(-2), target.reshape(bsz, (2*self.radius[0]+1) ** 2, -1))

        ref_gt = self.prep(images_lab_gt[0][:,ch], self.downsample_rate[-1])
        ref_gt = F.unfold(ref_gt, self.radius[-1] * 2 + 1, padding=self.radius[-1])
        outputs = (corr_predict.flatten(-2) * ref_gt).sum(1).reshape(bsz, -1, *tar_cmp.shape[-2:]) 

        logvar_tar = torch.log(torch.ones_like(logvar_pred) * self.vae_var) # var set to 1
        losses['vae_kl_loss'] = self.loss_weight['vae_kl_loss'] * build_loss(dict(type='Kl_Loss_Gaussion'))([mu_pred, logvar_pred], [mu_tar, logvar_tar], weight=weight_kl_prior)
        losses['vae_rec_loss'] = self.loss_weight['vae_rec_loss'] * self.compute_lphoto(images_lab_gt, ch, outputs, upsample=self.upsample, mask=weight_rec)[0]

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
        off = compute_flow_v2(h, w, target)
        corr_predict = corr_predict.permute(0,2,1)
        
        losses['cmp_loss'] = self.loss_weight['cmp_loss'] * F.smooth_l1_loss(corr_predict, off, reduction='mean')
        return losses

    def forward_exp(self, target, images_lab_gt, ch, tar_cmp):
        bsz = target.shape[0]

        corr_predict = self.flow_decoder(tar_cmp)
        losses = {}
        losses['cmp_loss'] = self.loss_weight['cmp_loss'] * self.cmp_loss(corr_predict.flatten(-2), target, weight=self.boundary_mask)

        if self.loss_weight.get('rec_loss', None) is not None:
            corr_predict = self.flow_decoder_m(corr_predict).flatten(-2)
            ref_gt = self.prep(images_lab_gt[0][:,ch], self.downsample_rate[-1])
            ref_gt = F.unfold(ref_gt, self.radius[-1] * 2 + 1, padding=self.radius[-1])
            outputs = (corr_predict * ref_gt).sum(1).reshape(bsz, -1, *tar_cmp.shape[-2:]) 
            losses['rec_loss'] = self.loss_weight['rec_loss'] * self.compute_lphoto(images_lab_gt, ch, outputs, upsample=self.upsample)[0]
        return losses

    
    def forward_test(self, imgs, images_lab=None, **kwargs):

        bsz, num_clips, t, c, h, w = images_lab.shape
        losses = {}

        images_lab_gt = [images_lab[:,0,i,:].clone() for i in range(t)]
        imgs = [imgs[:,0,i,:].clone() for i in range(t)]


        tar_cmp = self.backbone(images_lab_gt[-1])[-1]

        corr_predict = self.flow_decoder(tar_cmp)
        # VAE prior loss
        mu_pred = corr_predict[:,:self.ouput_dim // 2]
        logvar_pred = corr_predict[:,self.ouput_dim // 2:]

        outs = []
        for i in range(3):
            sampled_corr = self.reparameterise(mu_pred, torch.tensor(5).cuda())
            corr_predict = self.flow_decoder_m(sampled_corr)

            ref_gt = self.prep(imgs[0], self.downsample_rate[-1])
            ref_gt = F.unfold(ref_gt, self.radius[-1] * 2 + 1, padding=self.radius[-1]).view(bsz, 3, -1, ref_gt.shape[-1] ** 2)

            outputs = (corr_predict.flatten(-2).unsqueeze(1) * ref_gt).sum(2).reshape(bsz, -1, *tar_cmp.shape[-2:]) 
            out = F.interpolate(outputs, size=(h,w), mode='bilinear')
            outs.append(out)

        o = torch.cat(outs, 0)
        g = tensor2img(o, norm_mode='mean-std')
        imgs_gt_f1 = tensor2img(imgs[0], norm_mode='mean-std')
        imgs_gt_f2 = tensor2img(imgs[1], norm_mode='mean-std')




        