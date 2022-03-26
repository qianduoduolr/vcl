from builtins import isinstance, list
import os.path as osp
from collections import *
from pickle import NONE

import mmcv
import tempfile
from mmcv.runner import auto_fp16, load_checkpoint
from dall_e  import map_pixels, unmap_pixels, load_model
from torch import unsqueeze

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
class Memory_Tracker(BaseModel):
    def __init__(self,
                 backbone,
                 post_convolution=dict(in_c=256,out_c=64, ks=3, pad=1),
                 downsample_rate=4,
                 radius=12,
                 test_cfg=None,
                 train_cfg=None
                 ):
        """ MAST  (CVPR2020)

        Args:
            backbone ([type]): [description]
            test_cfg ([type], optional): [description]. Defaults to None.
            train_cfg ([type], optional): [description]. Defaults to None.
        """
        super().__init__()
        # Model options
        self.p = 0.3
        self.C = 7

        # self.backbone = build_backbone(backbone)
        self.backbone = build_backbone(backbone)
        self.post_convolution = nn.Conv2d(post_convolution['in_c'], post_convolution['out_c'], post_convolution['ks'], 1, post_convolution['pad'])
        self.D = downsample_rate

        self.test_cfg = test_cfg
        self.train_cfg = train_cfg
        
        self.test_cfg.ref = 2

        self.R = radius # radius

        self.colorizer = Colorizer(self.D, self.R, self.C)
        
    def forward_colorization(self, rgb_r, quantized_r, rgb_t, ref_index=None,current_ind=None):
        feats_r = [self.post_convolution(self.backbone(rgb)) for rgb in rgb_r]
        feats_t = self.post_convolution(self.backbone(rgb_t))

        quantized_t = self.colorizer(feats_r, feats_t, quantized_r, ref_index, current_ind)
        return quantized_t
    

    def forward_test(self, imgs, ref_seg_map, img_meta,
                save_image=False,
                save_path=None,
                iteration=None):
        num_frame = imgs.shape[3]
        
        imgs = [imgs[:,0,:,i] for i in range(num_frame)]
        annotations = [ref_seg_map[:,None,:]]

        all_seg_preds = []
        seg_preds = []
        
        N = len(imgs)
        outputs = [annotations[0].contiguous()]

        for i in range(N-1):
            mem_gap = 2
            # ref_index = [i]
            if self.test_cfg.ref == 0:
                ref_index = list(filter(lambda x: x <= i, [0, 5])) + list(filter(lambda x:x>0,range(i,i-mem_gap*3,-mem_gap)))[::-1]
                ref_index = sorted(list(set(ref_index)))
            elif self.test_cfg.ref == 1:
                ref_index = [0] + list(filter(lambda x: x > 0, range(i, i - mem_gap * 3, -mem_gap)))[::-1]
            elif self.test_cfg.ref == 2:
                ref_index = [i]
            else:
                raise NotImplementedError

            rgb_0 = [imgs[ind] for ind in ref_index]
            rgb_1 = imgs[i+1]

            anno_0 = [outputs[ind] for ind in ref_index]

            _, _, h, w = anno_0[0].size()

            with torch.no_grad():
                _output = self.forward_colorization(rgb_0, anno_0, rgb_1, ref_index, i+1)
                _output = F.interpolate(_output, (h,w), mode='bilinear')

                output = torch.argmax(_output, 1, keepdim=True).float()
                outputs.append(output)

            ###
            pad =  ((0,0), (0,0))
            if i == 0:
                # output first mask
                out_img = anno_0[0][0, 0].cpu().numpy().astype(np.uint8)
                out_img = np.pad(out_img, pad, 'edge').astype(np.uint8)
                seg_preds.append(out_img[None,:])
                
            out_img = output[0, 0].cpu().numpy().astype(np.uint8)
            out_img = np.pad(out_img, pad, 'edge').astype(np.uint8)
            seg_preds.append(out_img[None,:])
        
        
        seg_preds = np.stack(seg_preds, axis=1)
        all_seg_preds.append(seg_preds)
        
        if self.test_cfg.get('save_np', False):
            if len(all_seg_preds) > 1:
                return [all_seg_preds]
            else:
                return [all_seg_preds[0]]
        else:
            if len(all_seg_preds) > 1:
                all_seg_preds = np.stack(all_seg_preds, axis=1)
            else:
                all_seg_preds = all_seg_preds[0]
            # unravel batch dim
            return list(all_seg_preds)

    def forward_train(self, images_lab, imgs):
        
        bsz, n, c, t, h, w = imgs.shape
        
        images_lab_gt = [images_lab[:,0,:,i].clone() for i in range(t)]
        images_lab = [images_lab[:,0,:,i] for i in range(t)]
        imgs = [imgs[:,0,:,i] for i in range(t)]
        
        _, ch = self.dropout2d_lab(images_lab)
        
        losses = {}
        sum_loss, err_maps = self.compute_lphoto(images_lab, images_lab_gt, ch)
        losses['l1_loss'] = sum_loss
        
        return losses

        

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
    
    def compute_lphoto(self, image_lab, images_lab_gt, ch):
        b, c, h, w = image_lab[0].size()

        ref_x = [lab for lab in image_lab[:-1]]   # [im1, im2, im3]
        ref_y = [rgb[:,ch] for rgb in images_lab_gt[:-1]]  # [y1, y2, y3]
        tar_x = image_lab[-1]  # im4
        tar_y = images_lab_gt[-1][:,ch]  # y4


        outputs = self.forward_colorization(ref_x, ref_y, tar_x, [0,2], 4)   # only train with pairwise data

        outputs = F.interpolate(outputs, (h, w), mode='bilinear')
        loss = F.smooth_l1_loss(outputs*20, tar_y*20, reduction='mean')

        err_maps = torch.abs(outputs - tar_y).sum(1).detach()

        return loss, err_maps
    
    

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
        
        self.pretrained = pretrained
        self.scaling = scaling
        self.upsample = upsample
        self.weight = weight
        self.temperature = temperature
        
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
        
        # for mast l1_loss
        ref_gt = [self.prep(gt[:,ch]) for gt in images_lab_gt[:-1]]
        outputs = frame_transform(att, ref_gt, flatten=False, per_ref=self.per_ref)
        if self.per_ref:
            outputs = outputs[:,0].permute(0,2,1).reshape(bsz, -1, *fs.shape[-2:])     
        else:
            outputs = outputs.permute(0,2,1).reshape(bsz, -1, *fs.shape[-2:])     
            
        losses['l1_loss'], err_map = self.compute_lphoto(images_lab_gt, ch, outputs, upsample=self.upsample)

        return losses
        
    
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
                num_stage=2,
                feat_size=[64, 32],
                radius=[12, 6],
                downsample_rate=[4, 8],
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
        
        self.num_stage = num_stage
        self.feat_size = feat_size
        self.downsample_rate = downsample_rate


        self.mask = [ make_mask(feat_size[i], radius[i]) for i in range(len(radius)) ]

    def forward_train(self, imgs, images_lab=None):
            
        bsz, num_clips, t, c, h, w = imgs.shape
        
        images_lab_gt = [images_lab[:,0,i,:].clone() for i in range(t)]
        images_lab = [images_lab[:,0,i,:] for i in range(t)]
        _, ch = self.dropout2d_lab(images_lab)
        
        # forward to get feature
        fs = self.backbone(torch.stack(images_lab,1).flatten(0,1))
        fs = [f.reshape(bsz, t, *f.shape[-3:]) for f in fs]
        
        tar_pyramid, refs_pyramid = [f[:, -1] for f in fs], [ f[:, :-1] for f in fs]
        
        losses = {}
        
        
        for idx, (tar, refs) in enumerate(zip(tar_pyramid, refs_pyramid)):
            # get correlation attention map            
            _, att = non_local_attention(tar, refs, mask=self.mask[idx], scaling=True)            
            # for mast l1_loss
            ref_gt = [self.prep(gt[:,ch], downsample_rate=self.downsample_rate[idx]) for gt in images_lab_gt[:-1]]
            outputs = frame_transform(att, ref_gt, flatten=False)
            outputs = outputs[:,0].permute(0,2,1).reshape(bsz, -1, self.feat_size[idx], self.feat_size[idx])     
            losses[f'stage{idx}_l1_loss'], _ = self.compute_lphoto(images_lab_gt, ch, outputs)
        
        return losses
    
    def prep(self, image, downsample_rate):
        _,c,_,_ = image.size()
        x = image.float()[:,:,::downsample_rate,::downsample_rate]

        return x


@MODELS.register_module()
class Memory_Tracker_Custom_Hog(Memory_Tracker_Custom):
    def __init__(self,
                kernel=8,
                l2_loss=None,
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
        
        self.hog_layer = HOGLayer(kernel=kernel, pool=self.downsample_rate)
        self.l2_loss = build_loss(l2_loss)
    
    def forward_train(self, images_gray, images_lab=None):
        bsz, _, n, c, h, w = images_lab.shape
        
        images_gray_gt = [images_gray[:,i].clone() for i in range(n)]
        
        images_lab = [images_lab[:,0,i] for i in range(n)]
        _, ch = self.dropout2d_lab(images_lab)
        
        # forward to get feature
        fs = self.backbone(torch.stack(images_lab,1).flatten(0,1))
        if self.post_convolution is not None:
            fs = self.post_convolution(fs)
        
        fs = fs.reshape(bsz, n, *fs.shape[-3:])

        tar, refs = fs[:, -1], fs[:, :-1]
        
        # get correlation attention map            
        _, att = non_local_attention(tar, refs, mask=self.mask, scaling=True, per_ref=self.per_ref)
        
        
        losses = {}
        
        # for mast l1_loss
        ref_gt = [self.prep(gt).float() for gt in images_gray_gt[:-1]]
        outputs = frame_transform(att, ref_gt, flatten=False, per_ref=self.per_ref)
        if self.per_ref:
            outputs = outputs[:,0].permute(0,2,1).reshape(bsz, -1, *fs.shape[-2:])     
        else:
            outputs = outputs.permute(0,2,1).reshape(bsz, -1, *fs.shape[-2:])     
            
        losses['l2_loss'] = self.compute_l2_distance(images_gray_gt, outputs)
    
        
        return losses

    
    def prep(self, image):
        _,c,_,_ = image.size()
        x = self.hog_layer(image)

        return x

    def compute_l2_distance(self, images_gray_gt, outputs):
        b, c, h, w = images_gray_gt[0].size()

        tar = self.prep(images_gray_gt[-1])

        outputs = outputs.permute(0, 2, 3, 1).flatten(0,2)
        tar = tar.permute(0, 2, 3, 1).flatten(0,2)
        
        loss = self.l2_loss(outputs, tar)

        return loss
    
    def train_step(self, data_batch, optimizer, progress_ratio):
        """Abstract method for one training step.

        All subclass should overwrite it.
        """
        # parser loss
        losses = self(**data_batch, test_mode=False)
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
            num_samples=len(data_batch['images_gray'])
        )

        return outputs
    


@MODELS.register_module()
class Memory_Tracker_Custom_Inter(Memory_Tracker_Custom):
    
    def __init__(self,
                inter_w=5.0,
                *args,
                **kwargs
                 ):
       
        super().__init__(*args, **kwargs)
        
        self.inter_w = inter_w
    
    def forward_train(self, imgs, images_lab=None):
            
        bsz, _, n, c, h, w = imgs.shape
        
        images_lab_gt = [images_lab[:,0,i].clone() for i in range(n)]
        images_lab = [images_lab[:,0,i] for i in range(n)]
        _, ch = self.dropout2d_lab(images_lab)
        
        # forward to get feature
        fs = self.backbone(torch.stack(images_lab,1).flatten(0,1))
        if self.head is not None:
            fs = self.head(fs)
        
        fs = fs.reshape(bsz, n, *fs.shape[-3:])

        tar, refs = fs[:, -1], fs[:, :-1]
        refs_fold = (refs[:, -1])[None].repeat(bsz, 1, 1, 1, 1)
        
        # get correlation attention map            
        _, att = non_local_attention(tar, refs, mask=self.mask, scaling=True, per_ref=self.per_ref)
        _, att_inter = non_local_attention(tar, refs_fold, mask=self.mask, scaling=True, per_ref=False)
        
        losses = {}
        
        # for mast l1_loss
        ref_gt = [self.prep(gt[:,ch]) for gt in images_lab_gt[:-1]]
        outputs = frame_transform(att, ref_gt, flatten=False, per_ref=self.per_ref)        
        if self.per_ref:
            outputs = outputs[:,0].permute(0,2,1).reshape(bsz, -1, *fs.shape[-2:])     
        else:
            outputs = outputs.permute(0,2,1).reshape(bsz, -1, *fs.shape[-2:])     
        
        refs_gt_fold = torch.stack(ref_gt, 1)
        refs_gt_fold = refs_gt_fold.flatten(3).permute(0, 1, 3, 2)
        refs_gt_fold = refs_gt_fold[:, 0].repeat(bsz, 1, 1, 1)
        outputs_inter = frame_transform(att_inter, refs_gt_fold, flatten=False, per_ref=False)
        outputs_inter = outputs_inter[:,:,0].reshape(bsz, -1, *fs.shape[-2:])     
                   
        losses['l1_loss'], outputs = self.compute_lphoto(images_lab_gt, ch, outputs)
        losses['inter_loss'] = self.inter_w * self.compute_inter_intra_consistency(outputs, outputs_inter, images_lab_gt)
    
        return losses
    
    def compute_lphoto(self, images_lab_gt, ch, outputs):
        b, c, h, w = images_lab_gt[0].size()

        tar_y = images_lab_gt[-1][:,ch]  # y4

        outputs = F.interpolate(outputs, (h, w), mode='bilinear')
        loss = F.smooth_l1_loss(outputs*20, tar_y*20, reduction='mean')

        err_maps = torch.abs(outputs - tar_y).sum(1).detach()

        return loss, outputs
    
    def compute_inter_intra_consistency(self, outputs, outputs_inter, images_lab_gt):
        b, c, h, w = images_lab_gt[0].size()
        
        outputs_inter = F.interpolate(outputs_inter, (h,w), mode='bilinear')
        loss = F.smooth_l1_loss(outputs_inter*20, outputs*20, reduction='mean')
        
        return loss
    


@MODELS.register_module()
class Memory_Tracker_Cycle(Memory_Tracker_Custom):
    
    def __init__(self, T, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.T = T
    
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
        
        # forward backward consistency
        aff_i = torch.max(att, dim=-1, keepdim=True)[0]
        aff_j = torch.max(att, dim=-2, keepdim=True)[0]
        Q = (att * att) / (torch.matmul(aff_i, aff_j))
        Q = Q[:,0].max(dim=-1)[0]
        Q_sorted, _ = torch.sort(Q, dim=-1, descending=True)
        idx = int(Q.shape[-1] * self.T) - 1
        T = Q_sorted[:, idx:idx+1]
        M = (Q >= T).reshape(bsz, 1, *tar.shape[-2:])
        
        losses = {}
                
        # for mast l1_loss
        ref_gt = [self.prep(gt[:,ch]) for gt in images_lab_gt[:-1]]
        outputs = frame_transform(att, ref_gt, flatten=False, per_ref=self.per_ref)
        if self.per_ref:
            outputs = outputs[:,0].permute(0,2,1).reshape(bsz, -1, *fs.shape[-2:])     
        else:
            outputs = outputs.permute(0,2,1).reshape(bsz, -1, *fs.shape[-2:])     
            
        losses['l1_loss'], err_map = self.compute_lphoto(images_lab_gt, ch, outputs, M, upsample=self.upsample)

        return losses
    
    def compute_lphoto(self, images_lab_gt, ch, outputs, mask, upsample=True):
        b, c, h, w = images_lab_gt[0].size()

        tar_y = images_lab_gt[-1][:,ch]  # y4

        if upsample:
            outputs = F.interpolate(outputs, (h, w), mode='bilinear')
            loss = F.smooth_l1_loss(outputs * self.weight, tar_y * self.weight, reduction='none')
        else:
            tar_y = self.prep(images_lab_gt[-1])[:,ch]
            loss = F.smooth_l1_loss(outputs * self.weight, tar_y * self.weight, reduction='none')

        err_maps = torch.abs(outputs - tar_y).sum(1).detach()
        
        loss = (loss * mask).sum() / (mask.sum() + 1e-12)

        return loss, err_maps