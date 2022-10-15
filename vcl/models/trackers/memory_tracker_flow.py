import enum
import os.path as osp
import tempfile
from builtins import isinstance, list
from collections import *
from pickle import NONE
from re import A
from turtle import forward

import mmcv
import torch.nn as nn
import torch.nn.functional as F
from dall_e import load_model, map_pixels, unmap_pixels
from mmcv.ops import Correlation
from mmcv.runner import CheckpointLoader, auto_fp16, load_checkpoint
from torch import bilinear, unsqueeze
from tqdm import tqdm

from vcl.models.common import (images2video, masked_attention_efficient,
                               non_local_attention, pil_nearest_interpolate,
                               spatial_neighbor, video2images)
from vcl.models.common.correlation import *
from vcl.models.common.hoglayer import *
from vcl.models.common.local_attention import (
    flow_guided_attention_efficient, flow_guided_attention_efficient_v2)
from vcl.models.losses.losses import l1_loss
from vcl.utils import *

from ..builder import (build_backbone, build_components, build_loss,
                       build_model, build_operators)
from ..registry import MODELS
from .base import BaseModel
from .memory_tracker import *
from .modules import *


@MODELS.register_module()
class Memory_Tracker_Flow(Memory_Tracker_Custom):
    def __init__(self,
                 decoder,
                 cxt_backbone,
                 loss,
                 corr_op_cfg,
                 corr_op_cfg_infer,
                 num_levels: int,
                 cxt_channels: int,
                 h_channels: int,
                 corr_op_cfg_sample = None,
                 target_model: dict = None,
                 flow_clamp: int = -1,
                 flow_detach: bool = False,
                 freeze_bn: bool = False,
                 dense_rec: bool = False,
                 drop_ch: bool = False,
                 warp_op_cfg: dict = None,
                 prior_loss: dict = None,                
                 *args,
                 **kwargs):
        super().__init__(*args, **kwargs)

        self.decoder = build_components(decoder)
        self.target_model = build_model(target_model) if target_model is not None else None

        self.num_levels = num_levels
        self.context = build_backbone(cxt_backbone)
        self.h_channels = h_channels
        self.cxt_channels = cxt_channels
        # self.warp = build_operators(warp)
        self.dense_rec = dense_rec
        self.drop_ch = drop_ch
        self.flow_clamp = flow_clamp
        self.flow_detach = flow_detach
        self.loss = build_loss(loss)
        self.prior_loss = build_loss(prior_loss) if prior_loss is not None else None
        
        if corr_op_cfg_sample == None: corr_op_cfg_sample = corr_op_cfg
        self.corr_lookup = build_operators(corr_op_cfg)
        self.corr_lookup_inference = build_operators(corr_op_cfg_infer)
        self.corr_lookup_sample = build_operators(corr_op_cfg_sample)
        
        self.warp = build_operators(warp_op_cfg) if warp_op_cfg is not None else None

        assert self.num_levels == self.decoder.num_levels
        assert self.h_channels == self.decoder.h_channels
        assert self.cxt_channels == self.decoder.cxt_channels
        assert self.h_channels + self.cxt_channels == self.context.out_channels
    
    def extract_feat(
        self, imgs: torch.Tensor):
    
        """
        Extract features from images.
        Args:
            imgs (Tensor): The concatenated input images.
        Returns:
            Tuple[Tensor, Tensor, Tensor, Tensor]: The feature from the first
                image, the feature from the second image, the hidden state
                feature for GRU cell and the contextual feature.
        """
        bsz, p, c, t, h, w = imgs.shape
        img1 = imgs[:, 0, :, 0, ...]
        img2 = imgs[:, 0, :, 1, ...]
        
        feat1 = self.backbone(img1)
        feat2 = self.backbone(img2)
        cxt_feat = self.context(img1)

        h_feat, cxt_feat = torch.split(
            cxt_feat, [self.h_channels, self.cxt_channels], dim=1)
        h_feat = torch.tanh(h_feat)
        cxt_feat = torch.relu(cxt_feat)

        return feat1, feat2, h_feat, cxt_feat
    
    def reparameterise(self, mu, logvar):
        epsilon = torch.randn_like(mu)
        return mu + epsilon * torch.exp(logvar / 2)

    def dropout2d_lab(self, arr): # drop same layers for all images
        if not self.training:
            return arr

        drop_ch_num = int(np.random.choice(np.arange(1, 2), 1))
        drop_ch_ind = np.random.choice(np.arange(1,3), drop_ch_num, replace=False)

        arr[:,:,drop_ch_ind[0]] = 0
        arr *= (3 / (3 - drop_ch_num))

        return arr, drop_ch_ind # return channels not masked


    def forward_train(self, images_lab, imgs=None, flow_init=None):
            
        bsz, _, c, n, h, w = images_lab.shape
        images_lab_gt = images_lab.clone()
        losses = {}

        if self.drop_ch:
            _, ch = self.dropout2d_lab(images_lab)
        else:
            ch = np.random.choice(np.arange(1,3), 1, replace=False)

        # forward to get feature
        feat1, feat2, h_feat, cxt_feat = self.extract_feat(images_lab)
        B, _, H, W = feat1.shape
        
        # B x HW x HW
        if self.flow_detach:
            corr = non_local_attention(feat1, [feat2], flatten=False, temprature=1.0, mask=None, scaling=True, att_only=True)
            corr = corr.reshape(bsz * H * W, 1, H, W)
            feat1 = feat1.detach()
            feat2 = feat2.detach()
        
        # get correlation attention map      
        if flow_init is None:
            flow_init = torch.zeros((B, 2, H, W), device=feat1.device)

        
        if self.prior_loss is not None:
            preds, _, corr_, logvar_pred  = self.decoder(
                False,
                feat1,
                feat2,
                flow=flow_init,
                h_feat=h_feat,
                cxt_feat=cxt_feat,
            )
        else:
            # interative update the flow B x 2 x H x W
            preds, _, corr_  = self.decoder(
                False,
                feat1,
                feat2,
                flow=flow_init,
                h_feat=h_feat,
                cxt_feat=cxt_feat,
            )

        if self.flow_clamp != -1:
            preds = [torch.clamp(pred, -self.flow_clamp, self.flow_clamp) for pred in preds]
        
        
        if self.loss_weight['flow_rec_loss'] > 0:
            if self.dense_rec:
                pass
            else:
                recs = []
                gt = images_lab_gt[:,0,ch,0] * 20
                ref_v = images_lab_gt[:,0,ch,1].unsqueeze(1).repeat(1, H*W, 1, 1, 1).flatten(0,1)
                preds_lr = [pred[:,:,::self.downsample_rate[0],::self.downsample_rate[0]] / self.downsample_rate[0] for pred in preds]
                for pred_hr, pred_lr in zip(preds, preds_lr):
                    update_v = self.corr_lookup_sample([ref_v], pred_hr).reshape(bsz, 1, -1, H, W)
                    if self.flow_detach:
                        corr_v = self.corr_lookup([corr], pred_lr).reshape(bsz, 1, -1, H, W)
                    else:
                        corr_v = self.corr_lookup(corr_[:1], pred_lr).reshape(bsz, 1, -1, H, W)
                        
                    rec = (corr_v.softmax(2) * update_v).sum(2)
                    rec = F.interpolate(rec, (h,w), mode='bilinear')
                    recs.append(rec * 20)
            
                # for mast l1_loss
                losses['flow_rec_loss'] = self.loss_weight['flow_rec_loss'] * self.loss(recs, gt)

        
        if self.prior_loss is not None:
            mu_tar = torch.zeros_like(preds_lr[-1])
            sampled_pred = self.reparameterise(preds_lr[-1], logvar_pred)
            var_tar = torch.log(torch.ones_like(logvar_pred))
            losses['vae_prior_loss'] = self.loss_weight['vae_prior_loss'] * self.prior_loss([preds_lr[-1], logvar_pred], [mu_tar, var_tar])
            corr_v = self.corr_lookup(corr_[:1], sampled_pred).reshape(bsz, 1, -1, H, W)
            rec = (corr_v.softmax(2) * update_v).sum(2)
            rec = F.interpolate(rec, (h,w), mode='bilinear')
            losses['vae_rec_loss'] = self.loss_weight['vae_rec_loss'] * build_loss(dict(type='SmoothL1Loss'))(rec, gt)

        if self.target_model is not None:
            self.target_model.eval()
            with torch.no_grad():
                target = self.target_model(test_mode=True, imgs=imgs)[0][-1]
            losses['raft_gt_loss'] = self.loss_weight['raft_gt_loss'] * self.loss(preds, target)
        
        losses['abs_mv'] = preds_lr[-1].abs().mean()
        
        imgs = tensor2img(imgs[0,0].permute(1,0,2,3), norm_mode='mean-std')
        flow = preds[-1][0].permute(1,2,0).detach().cpu().numpy()
        flow = mmcv.flow2rgb(flow) * 255
        # flow_gt = mmcv.flow2rgb(target[0].permute(1,2,0).detach().cpu().numpy()) * 255
        vis_results = dict(flow=flow, imgs=imgs)

        return losses, vis_results

    def extract_feats(self, imgs, return_enc=False):
        clip_len = imgs.size(2)
        imgs = video2images(imgs)

        feats = self.backbone(imgs)
        feats = images2video(feats, clip_len)
        
        if return_enc:
            return feats
        
        cxt_feats = self.context(imgs)

        h_feats, cxt_feats = torch.split(
            cxt_feats, [self.h_channels, self.cxt_channels], dim=1)
        h_feats = torch.tanh(h_feats)
        cxt_feats = torch.relu(cxt_feats)

        cxt_feats = images2video(cxt_feats, clip_len)
        h_feats = images2video(h_feats, clip_len)

        return feats, h_feats, cxt_feats

    def forward_test(self, **kwargs):
        if self.test_cfg.get('use_raft', False):
            if not self.test_cfg.get('warp', False):
                return self.forward_test_raft(**kwargs)
            else:
                assert self.test_cfg.get('zero_flow', False) == True
                return self.forward_test_raft_warp(**kwargs)
        elif self.test_cfg.get('raft_prop', False):
            return self.forward_test_raft_prop(**kwargs)
        else:
            return self.forward_test_vanilla(**kwargs)
    
    def forward_test_vanilla(self, imgs, ref_seg_map, img_meta, 
                    images_lab=None,
                    flows=None,
                    save_image=False,
                    save_path=None,
                    iteration=None):
        
        imgs = imgs.reshape((-1, ) + imgs.shape[2:])
        clip_len = imgs.size(2)

        feat_bank, h_feat_bank, cxt_feat_bank  = self.extract_feats(imgs)
        feat_shape = feat_bank[-1].shape
        H, W = feat_shape[-2:]

        if self.test_cfg.get('with_norm', True):
            feat_bank_norm = feat_bank.pow(2).sum(dim=1, keepdim=True).sqrt()

        input_onehot = ref_seg_map.ndim == 4
        if not input_onehot:
            resized_seg_map = pil_nearest_interpolate(
                ref_seg_map.unsqueeze(1),
                size=feat_shape[2:]).squeeze(1).long()
            resized_seg_map = F.one_hot(resized_seg_map).permute(
                0, 3, 1, 2).float()
            ref_seg_map = F.interpolate(
                ref_seg_map.unsqueeze(1).float(),
                size=img_meta[0]['original_shape'][:2],
                mode='nearest').squeeze(1)
        else:
            resized_seg_map = F.interpolate(
                ref_seg_map,
                size=feat_shape[2:],
                mode='bilinear',
                align_corners=False).float()
            ref_seg_map = F.interpolate(
                ref_seg_map,
                size=img_meta[0]['original_shape'][:2],
                mode='bilinear',
                align_corners=False)

        seg_bank = []
        all_seg_preds = []

        seg_preds = [ref_seg_map.detach().cpu().numpy()]
        seg_bank.append(resized_seg_map.cpu())

        for frame_idx in tqdm(range(1, clip_len), total=clip_len-1):
            key_start = max(0, frame_idx - self.test_cfg.precede_frames)

            query_feat = feat_bank[:, :,frame_idx].to(imgs.device)
            query_cxt_feat = cxt_feat_bank[:, :,frame_idx].to(imgs.device)
            query_h_feat = h_feat_bank[:, :,frame_idx].to(imgs.device)
            key_feat = feat_bank[:, :, key_start:frame_idx].to(
                imgs.device)
            key_feat_norm = feat_bank_norm[:, :,key_start:frame_idx]
            value_logits = torch.stack(
                seg_bank[key_start:frame_idx], dim=2).to(imgs.device)

            if self.test_cfg.get('with_first', True):
                key_feat = torch.cat([feat_bank[:, :, 0:1].to(imgs.device),key_feat], dim=2)
                key_feat_norm = torch.cat([feat_bank_norm[:, :, 0:1].to(imgs.device),key_feat_norm], dim=2)
                value_logits = cat([seg_bank[0].unsqueeze(2).to(imgs.device), value_logits], dim=2)

            L = key_feat.shape[2]
            C = resized_seg_map.shape[1]

            if self.test_cfg.get('warm_up', False):
                pass
            else:
                value_logits = value_logits.transpose(1,2).flatten(0,1)
                query_feat = query_feat.repeat(L, 1, 1, 1)
                key_feat = key_feat.transpose(1,2).flatten(0,1)
                key_feat_norm = key_feat_norm.transpose(1,2).flatten(0,1)

                query_h_feat = query_h_feat.repeat(L, 1, 1, 1)
                query_cxt_feat = query_cxt_feat.repeat(L, 1, 1, 1)
                
                t_step = self.test_cfg.get('t_step', 4)
                
                preds = []
                corrs = []

                for ptr in range(0, L, t_step):

                    t = min(L-ptr, t_step)
                    flow_init = torch.zeros((t, 2, *query_feat.shape[-2:]), device=query_feat.device)

                    # flow decoder
                    flow, corr = self.decoder(
                            True,
                            query_feat[ptr:ptr+t_step],
                            key_feat[ptr:ptr+t_step],
                            flow=flow_init,
                            h_feat=query_h_feat[ptr:ptr+t_step],
                            cxt_feat=query_cxt_feat[ptr:ptr+t_step],
                            return_lr=True
                        )

                    if not self.test_cfg.get('with_norm', True):
                        pass
                    else:
                        C_ = query_feat.shape[1]
                        query_feat_norm = feat_bank_norm[:, :,frame_idx]
                        # L'HW x 1 x H x W
                        norm_ = torch.einsum('bci,tck->bitk', [query_feat_norm.flatten(-2), key_feat_norm[ptr:ptr+t_step].flatten(-2)]).permute(2,1,0,3).reshape(t*H*W, 1, H, W)
                        corr = (corr * torch.sqrt(torch.tensor(C_).float())/ norm_) / self.test_cfg.get('temperature')

                    if self.flow_clamp != -1:
                        flow = torch.clamp(flow, -self.flow_clamp, self.flow_clamp)
                    
                    preds.append(flow)
                    corrs.append(corr)


                if self.test_cfg.get('eval_mode', 'v2') == 'v2':
                    pred = torch.cat(preds, 0)
                    corr = torch.cat(corrs, 0)
                    del preds, corrs

                    seg_logit = flow_guided_attention_efficient_v2(
                                                                  corr, 
                                                                  value_logits, 
                                                                  pred, 
                                                                  sample_fn=self.corr_lookup_inference, 
                                                                  topk=self.test_cfg.get('topk', 10),
                                                                  zero_flow=self.test_cfg.get('zero_flow', False),
                                                                  ).reshape_as(resized_seg_map)

                    # a = F.interpolate(
                    #             seg_logit,
                    #             size=img_meta[0]['original_shape'][:2],
                    #             mode='bilinear',
                    #             align_corners=False).argmax(dim=1).permute(1,2,0).cpu()
                    # a = F.one_hot(a)[:,:,0].numpy()

                else:

                    seg_logit = flow_guided_attention_efficient(
                                                                preds,
                                                                corrs,
                                                                value_logits,
                                                                self.corr_lookup_inference,
                                                                topk=10,
                                                                step=32,
                                                                radius=6,
                                                                temperature=0.07,
                                                                mode='softmax',
                                                                ).reshape_as(resized_seg_map)

                    del preds, corrs

                seg_bank.append(seg_logit.cpu())

                seg_pred = F.interpolate(
                    seg_logit,
                    size=img_meta[0]['original_shape'][:2],
                    mode='bilinear',
                    align_corners=False)

                if not input_onehot:
                    seg_pred_min = seg_pred.view(*seg_pred.shape[:2], -1).min(
                        dim=-1)[0].view(*seg_pred.shape[:2], 1, 1)
                    seg_pred_max = seg_pred.view(*seg_pred.shape[:2], -1).max(
                        dim=-1)[0].view(*seg_pred.shape[:2], 1, 1)
                    normalized_seg_pred = (seg_pred - seg_pred_min) / (
                        seg_pred_max - seg_pred_min + 1e-12)
                    seg_pred = torch.where(seg_pred_max > 0,
                                           normalized_seg_pred, seg_pred)
                    seg_pred = seg_pred.argmax(dim=1)
                    seg_pred = F.interpolate(
                        seg_pred.float().unsqueeze(1),
                        size=img_meta[0]['original_shape'][:2],
                        mode='nearest').squeeze(1)
                seg_preds.append(seg_pred.detach().cpu().numpy())

                
        seg_preds = np.stack(seg_preds, axis=1)
        if self.test_cfg.get('save_np', False):
            assert seg_preds.shape[0] == 1
            eval_dir = '.eval'
            mmcv.mkdir_or_exist(eval_dir)
            temp_file = tempfile.NamedTemporaryFile(
                dir=eval_dir, suffix='.npy', delete=False)
            file_path = osp.join(eval_dir, temp_file.name)
            np.save(file_path, seg_preds[0])
            all_seg_preds.append(file_path)
        else:
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
        
    def forward_test_raft(self, imgs, ref_seg_map, img_meta, 
                     images_lab=None,
                    flows=None,
                    save_image=False,
                    save_path=None,
                    iteration=None):
        
        # images_lab used for feature matching, RGB imgs used for optical flow
        imgs_ = imgs.reshape((-1, ) + imgs.shape[2:])
        imgs = images_lab.reshape((-1, ) + images_lab.shape[2:])
        
        clip_len = imgs.size(2)

        feat_bank = self.extract_feats(imgs, return_enc=True)
        feat_shape = feat_bank[-1].shape
        H, W = feat_shape[-2:]

        if self.test_cfg.get('with_norm', True):
            feat_bank_norm = feat_bank.pow(2).sum(dim=1, keepdim=True).sqrt()

        input_onehot = ref_seg_map.ndim == 4
        if not input_onehot:
            resized_seg_map = pil_nearest_interpolate(
                ref_seg_map.unsqueeze(1),
                size=feat_shape[2:]).squeeze(1).long()
            resized_seg_map = F.one_hot(resized_seg_map).permute(
                0, 3, 1, 2).float()
            ref_seg_map = F.interpolate(
                ref_seg_map.unsqueeze(1).float(),
                size=img_meta[0]['original_shape'][:2],
                mode='nearest').squeeze(1)
        else:
            resized_seg_map = F.interpolate(
                ref_seg_map,
                size=feat_shape[2:],
                mode='bilinear',
                align_corners=False).float()
            ref_seg_map = F.interpolate(
                ref_seg_map,
                size=img_meta[0]['original_shape'][:2],
                mode='bilinear',
                align_corners=False)

        seg_bank = []
        all_seg_preds = []

        seg_preds = [ref_seg_map.detach().cpu().numpy()]
        seg_bank.append(resized_seg_map.cpu())

        for frame_idx in tqdm(range(1, clip_len), total=clip_len-1):
            key_start = max(0, frame_idx - self.test_cfg.precede_frames)
            query_feat = feat_bank[:, :,frame_idx].to(imgs.device)
            key_feat = feat_bank[:, :, key_start:frame_idx].to(
                imgs.device)
            key_feat_norm = feat_bank_norm[:, :,key_start:frame_idx]
            value_logits = torch.stack(
                seg_bank[key_start:frame_idx], dim=2).to(imgs.device)
            imgs_flow = imgs_[:, :, key_start:frame_idx]

            if self.test_cfg.get('with_first', True):
                key_feat = torch.cat([feat_bank[:, :, 0:1].to(imgs.device),key_feat], dim=2)
                key_feat_norm = torch.cat([feat_bank_norm[:, :, 0:1].to(imgs.device),key_feat_norm], dim=2)
                value_logits = cat([seg_bank[0].unsqueeze(2).to(imgs.device), value_logits], dim=2)
                imgs_flow = torch.cat([imgs_[:,:,0:1], imgs_flow], dim=2) 
                

            L = key_feat.shape[2]
            C = resized_seg_map.shape[1]

            if self.test_cfg.get('warm_up', False):
                pass
            else:
                value_logits = value_logits.transpose(1,2).flatten(0,1)
                query_feat = query_feat.repeat(L, 1, 1, 1)
                key_feat = key_feat.transpose(1,2).flatten(0,1)
                key_feat_norm = key_feat_norm.transpose(1,2).flatten(0,1)
                t_step = self.test_cfg.get('t_step', 4)
                
                preds = []
                corrs = []

                for ptr in range(0, L, t_step):

                    t = min(L-ptr, t_step)
                    flow_init = torch.zeros((t, 2, *query_feat.shape[-2:]), device=query_feat.device)
                    
                    ref_imgs = imgs_flow[:, :, ptr:ptr+t_step].permute(2, 0, 1, 3, 4)
                    tar_imgs = imgs_[:, :, frame_idx:frame_idx+1].permute(2, 0, 1, 3, 4).repeat(ref_imgs.shape[0], 1, 1, 1, 1)
                    flow_imgs = torch.stack([tar_imgs, ref_imgs], 3)
                    
                    corr = self.decoder(
                            True,
                            query_feat[ptr:ptr+t_step],
                            key_feat[ptr:ptr+t_step],
                            flow=flow_init,
                            h_feat=None,
                            cxt_feat=None,
                            return_corr=True
                        )
                    
                    flow = self.target_model(test_mode=True, imgs=flow_imgs, return_lr=True)[0]
                    
                    # flow = self.prep(flow, downsample_rate=8) / 8
            
                    # if torch.distributed.get_rank() == 0:
                    #     print(ptr, flow.max().item(), flow.min().item(), flow.abs().mean().item())

                    if not self.test_cfg.get('with_norm', True):
                        pass
                    else:
                        C_ = query_feat.shape[1]
                        query_feat_norm = feat_bank_norm[:, :,frame_idx]
                        # L'HW x 1 x H x W
                        norm_ = torch.einsum('bci,tck->bitk', [query_feat_norm.flatten(-2), key_feat_norm[ptr:ptr+t_step].flatten(-2)]).permute(2,1,0,3).reshape(t*H*W, 1, H, W)
                        corr = (corr * torch.sqrt(torch.tensor(C_).float())/ norm_) / self.test_cfg.get('temperature')

                    if self.flow_clamp != -1:
                        flow = torch.clamp(flow, -self.flow_clamp, self.flow_clamp)
                    
                    preds.append(flow)
                    corrs.append(corr)
                    del flow_imgs, ref_imgs, tar_imgs


                if self.test_cfg.get('eval_mode', 'v2') == 'v2':
                    pred = torch.cat(preds, 0)
                    corr = torch.cat(corrs, 0)
                    del preds, corrs

                    seg_logit = flow_guided_attention_efficient_v2(
                                                                  corr, 
                                                                  value_logits, 
                                                                  pred, 
                                                                  sample_fn=self.corr_lookup_inference, 
                                                                  topk=self.test_cfg.get('topk', 10),
                                                                  zero_flow=self.test_cfg.get('zero_flow', False),
                                                                  ).reshape_as(resized_seg_map)

                    a = F.interpolate(
                                seg_logit,
                                size=img_meta[0]['original_shape'][:2],
                                mode='bilinear',
                                align_corners=False).argmax(dim=1).permute(1,2,0).cpu()
                    a = F.one_hot(a)[:,:,0].numpy()

                else:

                    seg_logit = flow_guided_attention_efficient(
                                                                preds,
                                                                corrs,
                                                                value_logits,
                                                                self.corr_lookup_inference,
                                                                topk=10,
                                                                step=32,
                                                                radius=6,
                                                                temperature=0.07,
                                                                mode='softmax',
                                                                ).reshape_as(resized_seg_map)

                    del preds, corrs

                seg_bank.append(seg_logit.cpu())

                seg_pred = F.interpolate(
                    seg_logit,
                    size=img_meta[0]['original_shape'][:2],
                    mode='bilinear',
                    align_corners=False)

                if not input_onehot:
                    seg_pred_min = seg_pred.view(*seg_pred.shape[:2], -1).min(
                        dim=-1)[0].view(*seg_pred.shape[:2], 1, 1)
                    seg_pred_max = seg_pred.view(*seg_pred.shape[:2], -1).max(
                        dim=-1)[0].view(*seg_pred.shape[:2], 1, 1)
                    normalized_seg_pred = (seg_pred - seg_pred_min) / (
                        seg_pred_max - seg_pred_min + 1e-12)
                    seg_pred = torch.where(seg_pred_max > 0,
                                           normalized_seg_pred, seg_pred)
                    seg_pred = seg_pred.argmax(dim=1)
                    seg_pred = F.interpolate(
                        seg_pred.float().unsqueeze(1),
                        size=img_meta[0]['original_shape'][:2],
                        mode='nearest').squeeze(1)
                seg_preds.append(seg_pred.detach().cpu().numpy())

                
        seg_preds = np.stack(seg_preds, axis=1)
        if self.test_cfg.get('save_np', False):
            assert seg_preds.shape[0] == 1
            eval_dir = '.eval'
            mmcv.mkdir_or_exist(eval_dir)
            temp_file = tempfile.NamedTemporaryFile(
                dir=eval_dir, suffix='.npy', delete=False)
            file_path = osp.join(eval_dir, temp_file.name)
            np.save(file_path, seg_preds[0])
            all_seg_preds.append(file_path)
        else:
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
        
    def forward_test_raft_warp(self, imgs, ref_seg_map, img_meta, 
                     images_lab=None,
                    flows=None,
                    save_image=False,
                    save_path=None,
                    iteration=None):
        
        # images_lab used for feature matching, RGB imgs used for optical flow
        imgs_ = imgs.reshape((-1, ) + imgs.shape[2:])
        imgs = images_lab.reshape((-1, ) + images_lab.shape[2:])
        
        clip_len = imgs.size(2)

        feat_bank = self.extract_feats(imgs, return_enc=True)
        feat_shape = feat_bank[-1].shape
        H, W = feat_shape[-2:]

        input_onehot = ref_seg_map.ndim == 4
        if not input_onehot:
            resized_seg_map = pil_nearest_interpolate(
                ref_seg_map.unsqueeze(1),
                size=feat_shape[2:]).squeeze(1).long()
            resized_seg_map = F.one_hot(resized_seg_map).permute(
                0, 3, 1, 2).float()
            ref_seg_map = F.interpolate(
                ref_seg_map.unsqueeze(1).float(),
                size=img_meta[0]['original_shape'][:2],
                mode='nearest').squeeze(1)
        else:
            resized_seg_map = F.interpolate(
                ref_seg_map,
                size=feat_shape[2:],
                mode='bilinear',
                align_corners=False).float()
            ref_seg_map = F.interpolate(
                ref_seg_map,
                size=img_meta[0]['original_shape'][:2],
                mode='bilinear',
                align_corners=False)

        seg_bank = []
        all_seg_preds = []

        seg_preds = [ref_seg_map.detach().cpu().numpy()]
        seg_bank.append(resized_seg_map.cpu())

        for frame_idx in tqdm(range(1, clip_len), total=clip_len-1):
            key_start = max(0, frame_idx - self.test_cfg.precede_frames)
            query_feat = feat_bank[:, :,frame_idx].to(imgs.device)
            key_feat = feat_bank[:, :, key_start:frame_idx].to(
                imgs.device)
            value_logits = torch.stack(
                seg_bank[key_start:frame_idx], dim=2).to(imgs.device)
            imgs_flow = imgs_[:, :, key_start:frame_idx]

            if self.test_cfg.get('with_first', True):
                key_feat = torch.cat([feat_bank[:, :, 0:1].to(imgs.device),key_feat], dim=2)
                value_logits = cat([seg_bank[0].unsqueeze(2).to(imgs.device), value_logits], dim=2)
                imgs_flow = torch.cat([imgs_[:,:,0:1], imgs_flow], dim=2) 
                

            L = key_feat.shape[2]
            C = resized_seg_map.shape[1]

            if self.test_cfg.get('warm_up', False):
                pass
            else:
                value_logits = value_logits.transpose(1,2).flatten(0,1)
                query_feat = query_feat.repeat(L, 1, 1, 1)
                key_feat = key_feat.transpose(1,2).flatten(0,1)
                t_step = self.test_cfg.get('t_step', 4)
                
                preds = []
                corrs = []

                for ptr in range(0, L, t_step):

                    t = min(L-ptr, t_step)
                    flow_init = torch.zeros((t, 2, *query_feat.shape[-2:]), device=query_feat.device)
                    
                    ref_imgs = imgs_flow[:, :, ptr:ptr+t_step].permute(2, 0, 1, 3, 4)
                    tar_imgs = imgs_[:, :, frame_idx:frame_idx+1].permute(2, 0, 1, 3, 4).repeat(ref_imgs.shape[0], 1, 1, 1, 1)
                    flow_imgs = torch.stack([tar_imgs, ref_imgs], 3)
                    
                    flow = self.target_model(test_mode=True, imgs=flow_imgs, return_lr=True)[0]
                    
                    if self.flow_clamp != -1:
                        flow = torch.clamp(flow, -self.flow_clamp, self.flow_clamp)
                    else:
                        flow[:, 0] = torch.clamp(flow[:, 0], -W, W)
                        flow[:, 1] = torch.clamp(flow[:, 1], -H, H)
                        
                    key_feat_warped = self.warp(key_feat[ptr:ptr+t_step], flow)
                    value_logits[ptr:ptr+t_step] = self.warp(value_logits[ptr:ptr+t_step], flow)
                    
                    corr = self.decoder(
                            True,
                            query_feat[ptr:ptr+t_step],
                            key_feat_warped,
                            flow=flow_init,
                            h_feat=None,
                            cxt_feat=None,
                            return_corr=True
                        )
            
                    # a = F.interpolate(
                    #     value_logits[-1:],
                    #     size=img_meta[0]['original_shape'][:2],
                    #     mode='bilinear',
                    #     align_corners=False).argmax(dim=1).permute(1,2,0).cpu()
                    # a = F.one_hot(a)[:,:,0].numpy()
                    
                    # flow = self.prep(flow, downsample_rate=8) / 8

                    if not self.test_cfg.get('with_norm', True):
                        pass
                    else:
                        C_ = query_feat.shape[1]
                        query_feat_norm = query_feat[:1].pow(2).sum(dim=1, keepdim=True).sqrt()
                        key_feat_norm = key_feat_warped.pow(2).sum(dim=1, keepdim=True).sqrt() + 1e-7
                        # L'HW x 1 x H x W
                        norm_ = torch.einsum('bci,tck->bitk', [query_feat_norm.flatten(-2), key_feat_norm.flatten(-2)]).permute(2,1,0,3).reshape(t*H*W, 1, H, W)
                        corr = (corr * torch.sqrt(torch.tensor(C_).float())/ norm_) / self.test_cfg.get('temperature')

                    
                    preds.append(flow)
                    corrs.append(corr)
                    del flow_imgs, ref_imgs, tar_imgs


                if self.test_cfg.get('eval_mode', 'v2') == 'v2':
                    pred = torch.cat(preds, 0)
                    corr = torch.cat(corrs, 0)
                    del preds, corrs

                    seg_logit = flow_guided_attention_efficient_v2(
                                                                  corr, 
                                                                  value_logits, 
                                                                  pred, 
                                                                  sample_fn=self.corr_lookup_inference, 
                                                                  topk=self.test_cfg.get('topk', 10),
                                                                  zero_flow=self.test_cfg.get('zero_flow', False),
                                                                  ).reshape_as(resized_seg_map)

                    # a = F.interpolate(
                    #             seg_logit,
                    #             size=img_meta[0]['original_shape'][:2],
                    #             mode='bilinear',
                    #             align_corners=False).argmax(dim=1).permute(1,2,0).cpu()
                    # a = F.one_hot(a)[:,:,0].numpy()

                else:

                    seg_logit = flow_guided_attention_efficient(
                                                                preds,
                                                                corrs,
                                                                value_logits,
                                                                self.corr_lookup_inference,
                                                                topk=10,
                                                                step=32,
                                                                radius=6,
                                                                temperature=0.07,
                                                                mode='softmax',
                                                                ).reshape_as(resized_seg_map)

                    del preds, corrs

                seg_bank.append(seg_logit.cpu())

                seg_pred = F.interpolate(
                    seg_logit,
                    size=img_meta[0]['original_shape'][:2],
                    mode='bilinear',
                    align_corners=False)

                if not input_onehot:
                    seg_pred_min = seg_pred.view(*seg_pred.shape[:2], -1).min(
                        dim=-1)[0].view(*seg_pred.shape[:2], 1, 1)
                    seg_pred_max = seg_pred.view(*seg_pred.shape[:2], -1).max(
                        dim=-1)[0].view(*seg_pred.shape[:2], 1, 1)
                    normalized_seg_pred = (seg_pred - seg_pred_min) / (
                        seg_pred_max - seg_pred_min + 1e-12)
                    seg_pred = torch.where(seg_pred_max > 0,
                                           normalized_seg_pred, seg_pred)
                    seg_pred = seg_pred.argmax(dim=1)
                    seg_pred = F.interpolate(
                        seg_pred.float().unsqueeze(1),
                        size=img_meta[0]['original_shape'][:2],
                        mode='nearest').squeeze(1)
                seg_preds.append(seg_pred.detach().cpu().numpy())

                
        seg_preds = np.stack(seg_preds, axis=1)
        if self.test_cfg.get('save_np', False):
            assert seg_preds.shape[0] == 1
            eval_dir = '.eval'
            mmcv.mkdir_or_exist(eval_dir)
            temp_file = tempfile.NamedTemporaryFile(
                dir=eval_dir, suffix='.npy', delete=False)
            file_path = osp.join(eval_dir, temp_file.name)
            np.save(file_path, seg_preds[0])
            all_seg_preds.append(file_path)
        else:
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
    
    
    def forward_test_raft_prop(self, imgs, ref_seg_map, img_meta, 
            flows=None,
            save_image=False,
            save_path=None,
            iteration=None):
        
        # images_lab used for feature matching, RGB imgs used for optical flow
        imgs = imgs.reshape((-1, ) + imgs.shape[2:])
        
        assert self.test_cfg.get('precede_frames', 1) == 1
        
        B, C, clip_len, H, W = imgs.shape

        feat_shape = [H, W]

        input_onehot = ref_seg_map.ndim == 4
        if not input_onehot:
            resized_seg_map = pil_nearest_interpolate(
                ref_seg_map.unsqueeze(1),
                size=feat_shape).squeeze(1).long()
            resized_seg_map = F.one_hot(resized_seg_map).permute(
                0, 3, 1, 2).float()
            ref_seg_map = F.interpolate(
                ref_seg_map.unsqueeze(1).float(),
                size=img_meta[0]['original_shape'][:2],
                mode='nearest').squeeze(1)
        else:
            resized_seg_map = F.interpolate(
                ref_seg_map,
                size=feat_shape,
                mode='bilinear',
                align_corners=False).float()
            ref_seg_map = F.interpolate(
                ref_seg_map,
                size=img_meta[0]['original_shape'][:2],
                mode='bilinear',
                align_corners=False)

        seg_bank = []
        all_seg_preds = []

        seg_preds = [ref_seg_map.detach().cpu().numpy()]
        seg_bank.append(resized_seg_map.cpu())

        for frame_idx in tqdm(range(1, clip_len), total=clip_len-1):
            key_start = max(0, frame_idx - self.test_cfg.precede_frames)
            value_logits = torch.stack(
                seg_bank[key_start:frame_idx], dim=2).to(imgs.device)
            imgs_flow = imgs[:, :, key_start:frame_idx]
                
            L = imgs_flow.shape[2]
            C = resized_seg_map.shape[1]

            if self.test_cfg.get('warm_up', False):
                pass
            else:
                value_logits = value_logits.transpose(1,2).flatten(0,1)
                t_step = self.test_cfg.get('t_step', 4)

                for ptr in range(0, L, t_step):
                    
                    ref_imgs = imgs_flow[:, :, ptr:ptr+t_step].permute(2, 0, 1, 3, 4)
                    tar_imgs = imgs[:, :, frame_idx:frame_idx+1].permute(2, 0, 1, 3, 4).repeat(ref_imgs.shape[0], 1, 1, 1, 1)
                    flow_imgs = torch.stack([tar_imgs, ref_imgs], 3)
                    
                    flow = self.target_model(test_mode=True, imgs=flow_imgs)[0][-1]
                    
                    if self.flow_clamp != -1:
                        flow = torch.clamp(flow, -self.flow_clamp, self.flow_clamp)
                    else:
                        flow[:, 0] = torch.clamp(flow[:, 0], -feat_shape[1], feat_shape[1])
                        flow[:, 1] = torch.clamp(flow[:, 1], -feat_shape[0], feat_shape[0])
                        
                    seg_logit = self.warp(value_logits[ptr:ptr+t_step], flow)
                    

                    del flow_imgs, ref_imgs, tar_imgs


                seg_bank.append(seg_logit.cpu())

                seg_pred = F.interpolate(
                    seg_logit,
                    size=img_meta[0]['original_shape'][:2],
                    mode='bilinear',
                    align_corners=False)

                if not input_onehot:
                    seg_pred_min = seg_pred.view(*seg_pred.shape[:2], -1).min(
                        dim=-1)[0].view(*seg_pred.shape[:2], 1, 1)
                    seg_pred_max = seg_pred.view(*seg_pred.shape[:2], -1).max(
                        dim=-1)[0].view(*seg_pred.shape[:2], 1, 1)
                    normalized_seg_pred = (seg_pred - seg_pred_min) / (
                        seg_pred_max - seg_pred_min + 1e-12)
                    seg_pred = torch.where(seg_pred_max > 0,
                                           normalized_seg_pred, seg_pred)
                    seg_pred = seg_pred.argmax(dim=1)
                    seg_pred = F.interpolate(
                        seg_pred.float().unsqueeze(1),
                        size=img_meta[0]['original_shape'][:2],
                        mode='nearest').squeeze(1)
                seg_preds.append(seg_pred.detach().cpu().numpy())

                
        seg_preds = np.stack(seg_preds, axis=1)
        if self.test_cfg.get('save_np', False):
            assert seg_preds.shape[0] == 1
            eval_dir = '.eval'
            mmcv.mkdir_or_exist(eval_dir)
            temp_file = tempfile.NamedTemporaryFile(
                dir=eval_dir, suffix='.npy', delete=False)
            file_path = osp.join(eval_dir, temp_file.name)
            np.save(file_path, seg_preds[0])
            all_seg_preds.append(file_path)
        else:
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
    