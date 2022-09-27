from builtins import isinstance, list
import enum
import os.path as osp
from collections import *
from pickle import NONE
from re import A
from turtle import forward

import mmcv
import tempfile
from tqdm import tqdm
from mmcv.runner import auto_fp16, load_checkpoint
from dall_e  import map_pixels, unmap_pixels, load_model
from torch import bilinear, unsqueeze
from mmcv.runner import CheckpointLoader
from mmcv.ops import Correlation


from vcl.models.common.correlation import *
from vcl.models.common.hoglayer import *
from vcl.models.common import images2video, masked_attention_efficient, pil_nearest_interpolate, spatial_neighbor, video2images
from vcl.models.common.local_attention import flow_guided_attention_efficient
from vcl.models.losses.losses import l1_loss

from .base import BaseModel
from ..builder import build_backbone, build_components, build_loss, build_operators
from ..registry import MODELS
from vcl.utils import *

import torch.nn as nn
import torch.nn.functional as F
from .modules import *
from .memory_tracker import *

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
                 freeze_bn: bool = False,
                 dense_rec: bool = False,
                 drop_ch: bool = False,
                 *args,
                 **kwargs):
        super().__init__(*args, **kwargs)

        self.decoder = build_components(decoder)

        self.num_levels = num_levels
        self.context = build_backbone(cxt_backbone)
        self.h_channels = h_channels
        self.cxt_channels = cxt_channels
        # self.warp = build_operators(warp)
        self.dense_rec = dense_rec
        self.drop_ch = drop_ch
        self.loss = build_loss(loss)
        self.corr_lookup = build_operators(corr_op_cfg)
        self.corr_lookup_inference = build_operators(corr_op_cfg_infer)

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

        if self.drop_ch:
            _, ch = self.dropout2d_lab(images_lab)
        else:
            ch = np.random.choice(np.arange(1,3), 1, replace=False)

        # forward to get feature
        feat1, feat2, h_feat, cxt_feat = self.extract_feat(images_lab)
        B, _, H, W = feat1.shape
        
        
        # get correlation attention map      
        if flow_init is None:
            flow_init = torch.zeros((B, 2, H, W), device=feat1.device)

        # interative update the flow B x 2 x H x W
        preds, preds_lr, corr_pyramid = self.decoder(
            False,
            feat1,
            feat2,
            flow=flow_init,
            h_feat=h_feat,
            cxt_feat=cxt_feat,
            return_lr=True
        )

        if self.dense_rec:
            gt = imgs[:,0,:,0]
            ref = imgs[:,0,:,1]
            for pred in preds:
                warp_frame = 1
        else:
            recs = []
            gt = images_lab_gt[:,0,ch,0] * 20
            ref_v = self.prep(images_lab_gt[:,0,ch,1]).unsqueeze(1).repeat(1, H*W, 1, 1, 1).flatten(0,1)
            for pred_lr in preds_lr:
                # pred_lr = torch.zeros_like(pred_lr).cuda()
                update_v = self.corr_lookup([ref_v], pred_lr).reshape(bsz, 1, -1, H, W)
                corr_v = self.corr_lookup(corr_pyramid[:1], pred_lr).reshape(bsz, 1, -1, H, W)
                rec = (corr_v.softmax(2) * update_v).sum(2)
                rec = F.interpolate(rec, (h,w), mode='bilinear')
                recs.append(rec * 20)
        
            # for mast l1_loss
            losses = {}
            losses['flow_rec_loss'] = self.loss_weight['flow_rec_loss'] * self.loss(recs, gt)
        
        imgs = tensor2img(imgs[0,0].permute(1,0,2,3), norm_mode='mean-std')
        flow = preds[-1][0].permute(1,2,0).detach().cpu().numpy()
        flow = mmcv.flow2rgb(flow) * 255
        vis_results = dict(flow=flow, imgs=imgs)

        return losses, vis_results

    def extract_feats(self, imgs):
        clip_len = imgs.size(2)
        imgs = video2images(imgs)

        feats = self.backbone(imgs)
        cxt_feats = self.context(imgs)

        h_feats, cxt_feats = torch.split(
            cxt_feats, [self.h_channels, self.cxt_channels], dim=1)
        h_feats = torch.tanh(h_feats)
        cxt_feats = torch.relu(cxt_feats)

        feats = images2video(feats, clip_len)
        cxt_feats = images2video(cxt_feats, clip_len)
        h_feats = images2video(h_feats, clip_len)

        return feats, h_feats, cxt_feats

    def forward_test(self, imgs, ref_seg_map, img_meta,
                    save_image=False,
                    save_path=None,
                    iteration=None):

        imgs = imgs.reshape((-1, ) + imgs.shape[2:])
        clip_len = imgs.size(2)

        feat_bank, h_feat_bank, cxt_feat_bank  = self.extract_feats(imgs)
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
            query_cxt_feat = cxt_feat_bank[:, :,frame_idx].to(imgs.device)
            query_h_feat = h_feat_bank[:, :,frame_idx].to(imgs.device)
            key_feat = feat_bank[:, :, key_start:frame_idx].to(
                imgs.device)
            value_logits = torch.stack(
                seg_bank[key_start:frame_idx], dim=2).to(imgs.device)

            if self.test_cfg.get('with_first', True):
                key_feat = torch.cat([feat_bank[:, :, 0:1].to(imgs.device),key_feat], dim=2)
                value_logits = cat([seg_bank[0].unsqueeze(2).to(imgs.device), value_logits], dim=2)

            L = key_feat.shape[2]
            C = resized_seg_map.shape[1]

            if self.test_cfg.get('warm_up', False):
                pass
            else:

                key_feat = key_feat.transpose(1,2).flatten(0,1)
                value_logits = value_logits.transpose(1,2).flatten(0,1)
            

                query_feat = query_feat.repeat(key_feat.shape[0], 1, 1, 1)
                query_h_feat = query_h_feat.repeat(key_feat.shape[0], 1, 1, 1)
                query_cxt_feat = query_cxt_feat.repeat(key_feat.shape[0], 1, 1, 1)

                # seg_logit = flow_guided_attention_efficient(
                #                                             query_feat, 
                #                                             key_feat, 
                #                                             value_logits, 
                #                                             query_h_feat,
                #                                             query_cxt_feat,
                #                                             self.corr_lookup_inference, 
                #                                             topk=self.test_cfg.get('topk', 10), 
                #                                             decoder=self.decoder, 
                #                                             ).reshape(1, C, H, W)

                t_step = 21
                step = 512
                L, C, H, W = value_logits.shape
                C_ = query_feat.shape[1]
                value_feat = value_logits

                if step is None:
                    step = H * W

                seg_logit = torch.zeros(1, C,
                                    H * W).to(value_feat)

                for ptr in range(0, H*W, step):
                    s = min(H*W - ptr, step)
                    affinity = torch.zeros(L, (6*2 +1)**2,
                                        s).to(value_feat)
                    value_ = []


                    for ptr_t in range(0, L, t_step):
                        
                        t = min(L- ptr_t, t_step) 

                        flow_init = torch.zeros((t, 2, *query_feat.shape[-2:]), device=query_feat.device)
                        value_feat = value_feat[ptr_t:ptr_t+t_step]

                        pred, corr = self.decoder(
                                True,
                                query_feat[ptr_t:ptr_t+t_step],
                                key_feat[ptr_t:ptr_t+t_step],
                                flow=flow_init,
                                h_feat=query_h_feat[ptr_t:ptr_t+t_step],
                                cxt_feat=query_cxt_feat[ptr_t:ptr_t+t_step],
                            )

                    #     if not with_norm:
                    #         pass
                    #     else:
                    #         query_feat = query[ptr_t:ptr_t+t_step].pow(2).sum(dim=1, keepdim=True).sqrt()
                    #         key_feat = key[ptr_t:ptr_t+t_step].pow(2).sum(dim=1, keepdim=True).sqrt()
                    #         # L'HW x 1 x H x W
                    #         norm_ = torch.matmul(query_feat.flatten(-2).permute(0,2,1), key_feat.flatten(-2)).reshape(t*H*W, 1, H, W)
                    #         corr = (corr * torch.sqrt(torch.tensor(C_).float())/ norm_) / temperature
                    #         corr = corr.reshape(t, -1, 1, H, W)
                        
                    #     xx = torch.arange(0, W, device=pred.device)
                    #     yy = torch.arange(0, H, device=pred.device)
                    #     # L' x 2 x H x W
                    #     grid = coords_grid(t, xx, yy) + pred
                    #     grid = grid.permute(0, 2, 3, 1)
                    #     grid = grid.flatten(1,2)

                    #     # L' x S x 2
                    #     g = grid[:, ptr:ptr + step]
                    #     c = corr[:, ptr:ptr + step]
                    #     # L'S x 1 x H x W
                    #     c = c.flatten(0,1)

                    #     # L' x r^2 x S
                    #     cur_affinity = sample_fn([c], flow=None, grid=g)
                    #     affinity[ptr_t:ptr_t+t_step, ...] = cur_affinity

                    #     # for valune
                    #     # L'S x C x H x W 
                    #     v = value_feat.repeat(1, s, 1, 1, 1).flatten(0,1)
                    #     # 1 x L' x C x r^2 x S
                    #     ref_v = sample_fn([v], flow=None, mode='nearest', grid=g).reshape(t, C, -1, s).unsqueeze(0)
                    #     value_.append(ref_v)

                    # if topk is not None:
                    #     # 1 x (L' x r^2) x S
                    #     affinity = affinity.reshape(1, -1, s)

                    #     # [1, topk, S]
                    #     topk_affinity, topk_indices = affinity.topk(k=topk, dim=1)
                        
                    #     # 1 x C x (L x r^2) x S
                    #     value_ = torch.cat(value_, 1).transpose(1,2).flatten(2,3)

                    #     topk_value = value_.transpose(0, 1).reshape(
                    #         C, -1).index_select(
                    #             dim=1, index=topk_indices.reshape(-1))
                    #     # [N, C, topk, step]
                    #     topk_value = topk_value.reshape(C,
                    #                                     *topk_indices.shape).transpose(
                    #                                         0, 1)
                    #     if mode == 'softmax':
                    #         topk_affinity = topk_affinity.softmax(dim=1)
                    #     elif mode == 'cosine':
                    #         topk_affinity = topk_affinity.clamp(min=0)**2
                    #     else:
                    #         raise ValueError

                    #     cur_output = torch.einsum('bcks,bks->bcs', topk_value,
                    #                             topk_affinity)
                    # print(time.time()- start)
                    seg_logit[...,ptr:ptr+step] = 0

                seg_logit = seg_logit.reshape(1, C, H, W)

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
    