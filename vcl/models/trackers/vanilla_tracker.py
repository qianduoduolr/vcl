import os.path as osp
import tempfile

import mmcv
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.runner import auto_fp16

from ..builder import build_backbone, build_loss, build_components
from ..backbones import ResNet
from ..common import (cat, images2video, masked_attention_efficient,
                      pil_nearest_interpolate, spatial_neighbor, video2images)
from ..registry import MODELS
from ..base import BaseModel
from ...utils import *

@MODELS.register_module()
class BaseTracker(BaseModel):
    """Base class for recognizers.

    All recognizers should subclass it.
    All subclass should overwrite:

    - Methods:``forward_train``, supporting to forward when training.
    - Methods:``forward_test``, supporting to forward when testing.

    Args:
        backbone (dict): Backbone modules to extract feature.
        cls_head (dict): Classification head to process feature.
        train_cfg (dict): Config for training. Default: None.
        test_cfg (dict): Config for testing. Default: None.
    """

    def __init__(self, backbone, head=None, train_cfg=None, test_cfg=None):
        super().__init__()
        self.backbone = build_backbone(backbone)
        if head is not None:
            self.head = build_components(head)
        else:
            self.head = None

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.init_weights()

        self.fp16_enabled = False
        self.register_buffer('iteration', torch.tensor(0, dtype=torch.float))

    def init_weights(self):
        """Initialize the model network weights."""
        # self.backbone.init_weights()
        pass

    @auto_fp16()
    def extract_feat(self, imgs):
        """Extract features through a backbone.

        Args:
            imgs (torch.Tensor): The input images.

        Returns:
            torch.tensor: The extracted features.
        """
        x = self.backbone(imgs)
        
        if self.head is not None:
            try:
                x = self.head(x)
            except Exception as e:
                return x
        return x

@MODELS.register_module()
class VanillaTracker(BaseTracker):
    """Pixel Tracker framework."""

    def __init__(self, *args, **kwargs):
        super(VanillaTracker, self).__init__(*args, **kwargs)
        self.save_np = self.test_cfg.get('save_np', False)
        self.hard_prop = self.test_cfg.get('hard_prop', False)
        self.norm_mask = self.test_cfg.get('norm_mask', True)

    @property
    def stride(self):
        assert isinstance(self.backbone, ResNet)
        end_index = self.backbone.original_out_indices[0]
        return np.prod(self.backbone.strides[:end_index + 1]) * 4

    def extract_feat_test(self, imgs):
        outs = []
        if self.test_cfg.get('all_blocks', False):
            assert isinstance(self.backbone, ResNet)
            x = self.backbone.conv1(imgs)
            x = self.backbone.maxpool(x)
            outs = []
            for i, layer_name in enumerate(self.backbone.res_layers):
                res_layer = getattr(self.backbone, layer_name)
                if i in self.test_cfg.out_indices:
                    for block in res_layer:
                        x = block(x)
                        outs.append(x)
                else:
                    x = res_layer(x)
            return tuple(outs)
        return self.extract_feat(imgs)

    def extract_single_feat(self, imgs, idx):
        feats = self.extract_feat_test(imgs)
        if isinstance(feats, (tuple, list)):
            return feats[idx]
        else:
            return feats

    def get_feats(self, imgs, num_feats):
        assert imgs.shape[0] == 1
        batch_step = self.test_cfg.get('batch_step', 10)
        feat_bank = [[] for _ in range(num_feats)]
        clip_len = imgs.size(2)
        imgs = video2images(imgs)
        for batch_ptr in range(0, clip_len, batch_step):
            feats = self.extract_feat_test(imgs[batch_ptr:batch_ptr +
                                                batch_step])
            if isinstance(feats, tuple):
                assert len(feats) == len(feat_bank)
                for i in range(len(feats)):
                    feat_bank[i].append(feats[i].cpu())
            else:
                feat_bank[0].append(feats.cpu())
        for i in range(num_feats):
            feat_bank[i] = images2video(
                torch.cat(feat_bank[i], dim=0), clip_len)
            assert feat_bank[i].size(2) == clip_len

        return feat_bank

    def forward_train(self, imgs, labels=None):
        raise NotImplementedError

    def forward_test(self, imgs, ref_seg_map, img_meta,
                    save_image=False,
                    save_path=None,
                    iteration=None):
        """Defines the computation performed at every call when evaluation and
        testing."""

        imgs = imgs.reshape((-1, ) + imgs.shape[2:])
        clip_len = imgs.size(2)
        # get target shape
        dummy_feat = self.extract_feat_test(imgs[0:1, :, 0])
        if isinstance(dummy_feat, (list, tuple)):
            feat_shapes = [_.shape for _ in dummy_feat]
        else:
            feat_shapes = [dummy_feat.shape]
        all_seg_preds = []
        feat_bank = self.get_feats(imgs, len(dummy_feat))
        for feat_idx, feat_shape in enumerate(feat_shapes):
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

            seg_preds = [ref_seg_map.detach().cpu().numpy()]
            neighbor_range = self.test_cfg.get('neighbor_range', None)
            if neighbor_range is not None:
                spatial_neighbor_mask = spatial_neighbor(
                    feat_shape[0],
                    *feat_shape[2:],
                    neighbor_range=neighbor_range,
                    device=imgs.device,
                    dtype=imgs.dtype,
                    mode='circle')
            else:
                spatial_neighbor_mask = None

            seg_bank.append(resized_seg_map.cpu())
            for frame_idx in range(1, clip_len):
                key_start = max(0, frame_idx - self.test_cfg.precede_frames)
                query_feat = feat_bank[feat_idx][:, :,
                                                 frame_idx].to(imgs.device)
                key_feat = feat_bank[feat_idx][:, :, key_start:frame_idx].to(
                    imgs.device)
                value_logits = torch.stack(
                    seg_bank[key_start:frame_idx], dim=2).to(imgs.device)
                if self.test_cfg.get('with_first', True):
                    key_feat = torch.cat([
                        feat_bank[feat_idx][:, :, 0:1].to(imgs.device),
                        key_feat
                    ],
                                         dim=2)
                    value_logits = cat([
                        seg_bank[0].unsqueeze(2).to(imgs.device), value_logits
                    ],
                                       dim=2)
                seg_logit = masked_attention_efficient(
                    query_feat,
                    key_feat,
                    value_logits,
                    spatial_neighbor_mask,
                    temperature=self.test_cfg.temperature,
                    topk=self.test_cfg.topk,
                    normalize=self.test_cfg.get('with_norm', True),
                    non_mask_len=0 if self.test_cfg.get(
                        'with_first_neighbor', True) else 1)
                
                if not self.hard_prop:
                    seg_bank.append(seg_logit.cpu())
                else:
                    seg_logit_hard = seg_logit.argmax(1,keepdim=True)
                    seg_logit_hard = F.one_hot(seg_logit_hard)[:,0].permute(0,3,1,2).float()
                    seg_bank.append(seg_logit_hard.cpu())

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
            if self.save_np:
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
        if self.save_np:
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
   

@MODELS.register_module()
class VanillaTracker_V2(VanillaTracker):
    """Follow MAMP."""

    def __init__(self, pad_divisible=8, *args, **kwargs):
        super(VanillaTracker_V2, self).__init__(*args, **kwargs)
        self.divisible = pad_divisible

    def forward_test(self, imgs, ref_seg_map, img_meta,
                    save_image=False,
                    save_path=None,
                    iteration=None):
        """Defines the computation performed at every call when evaluation and
        testing."""

        imgs = imgs.reshape((-1, ) + imgs.shape[2:])
        clip_len = imgs.size(2)
        
        imgs = [ self.align_pad(imgs[:, :, i])[0] for i in range(clip_len) ]
        imgs = torch.stack(imgs, dim=2)
        
        ori_h, ori_w = img_meta[0]['original_shape'][:2]
        
        # get target shape
        dummy_feat = self.extract_feat_test(imgs[0:1, :, 0])
        if isinstance(dummy_feat, (list, tuple)):
            feat_shapes = [_.shape for _ in dummy_feat]
        else:
            feat_shapes = [dummy_feat.shape]
        all_seg_preds = []
        feat_bank = self.get_feats(imgs, len(dummy_feat))
        for feat_idx, feat_shape in enumerate(feat_shapes):
            input_onehot = ref_seg_map.ndim == 4
            if not input_onehot:
                ref_seg_map, H, W = self.align_pad(ref_seg_map.unsqueeze(1))
                resized_seg_map = self.prep(ref_seg_map)[:,0].long()
                resized_seg_map = F.one_hot(resized_seg_map).permute(
                    0, 3, 1, 2).float()
                ref_seg_map = F.interpolate(
                    ref_seg_map.float(),
                    size=(H,W),
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

            seg_preds = [ref_seg_map[:,:ori_h,:ori_w].detach().cpu().numpy()]
            neighbor_range = self.test_cfg.get('neighbor_range', None)
            if neighbor_range is not None:
                spatial_neighbor_mask = spatial_neighbor(
                    feat_shape[0],
                    *feat_shape[2:],
                    neighbor_range=neighbor_range,
                    device=imgs.device,
                    dtype=imgs.dtype,
                    mode='circle')
            else:
                spatial_neighbor_mask = None

            seg_bank.append(resized_seg_map.cpu())
            for frame_idx in range(1, clip_len):
                key_start = max(0, frame_idx - self.test_cfg.precede_frames)
                query_feat = feat_bank[feat_idx][:, :,
                                                 frame_idx]
                key_feat = feat_bank[feat_idx][:, :, key_start:frame_idx].to(
                    imgs.device)
                value_logits = torch.stack(
                    seg_bank[key_start:frame_idx], dim=2)
                if self.test_cfg.get('with_first', True):
                    key_feat = torch.cat([
                        feat_bank[feat_idx][:, :, 0:1],
                        key_feat
                    ],
                                         dim=2)
                    value_logits = cat([
                        seg_bank[0].unsqueeze(2), value_logits
                    ],
                                       dim=2)
                seg_logit = masked_attention_efficient(
                    query_feat,
                    key_feat,
                    value_logits,
                    spatial_neighbor_mask,
                    temperature=self.test_cfg.temperature,
                    topk=self.test_cfg.topk,
                    normalize=self.test_cfg.get('with_norm', True),
                    non_mask_len=0 if self.test_cfg.get(
                        'with_first_neighbor', True) else 1)
                
                if not self.hard_prop:
                    seg_bank.append(seg_logit.cpu())
                else:
                    seg_logit_hard = seg_logit.argmax(1,keepdim=True)
                    seg_logit_hard = F.one_hot(seg_logit_hard)[:,0].permute(0,3,1,2).float()
                    seg_bank.append(seg_logit_hard.cpu())

                seg_pred = F.interpolate(
                    seg_logit,
                    size=(H,W),
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
                        size=(H,W),
                        mode='nearest').squeeze(1)
                seg_pred = seg_pred[:,:ori_h,:ori_w]
                seg_preds.append(seg_pred.detach().cpu().numpy())

            seg_preds = np.stack(seg_preds, axis=1)
            if self.save_np:
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
        if self.save_np:
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
        
    def align_pad(self, img):
        cur_b, cur_c, cur_h, cur_w = img.shape
        pad_h = 0 if (cur_h % self.divisible) == 0 else self.divisible - (cur_h % self.divisible)
        pad_w = 0 if (cur_w % self.divisible) == 0 else self.divisible - (cur_w % self.divisible)

        if (pad_h + pad_w) != 0:
            pad = nn.ZeroPad2d(padding=(0, pad_w, 0, pad_h))
            image = pad(img)
            final_h = cur_h + pad_h 
            final_w = cur_w + pad_w
            return image, final_h, final_w
        else:
            return img, cur_h, cur_w
        
    def prep(self, image, mode='default'):
        _,c,_,_ = image.size()

        x = image.float()[:,:,::self.divisible,::self.divisible]

        return x

@MODELS.register_module()
class VanillaTracker_Fusion(VanillaTracker):
    """Pixel Tracker framework."""

    def __init__(self, spatial_modality='LAB', backbone_=None, *args, **kwargs):
        super(VanillaTracker_Fusion, self).__init__(*args, **kwargs)
        self.spatial_modality = spatial_modality
        if backbone_ is not None:
            # backbone_.strides = self.backbone.strides
            backbone_.out_indices = self.backbone.out_indices
            self.backbone_ = build_backbone(backbone_) 
            self.gama = self.test_cfg.get('fusion_gama', 1)
            self.fusion_mode = self.test_cfg.get('fusion_mode', 'early_concat')

    @auto_fp16()
    def extract_feat(self, imgs, imgs_lab=None, mode='ST'):
        """Extract features through a backbone.

        Args:
            imgs (torch.Tensor): The input images.

        Returns:
            torch.tensor: The extracted features.
        """
        imgs_s = imgs_lab if self.spatial_modality == 'LAB' else imgs
        if mode == 'S':
            x = self.backbone(imgs_s)
        elif mode == 'T':
            x = self.backbone_(imgs_lab)
        elif mode == 'ST':
            x = self.backbone(imgs_s)
            x_ = self.backbone_(imgs_lab)

        if self.head is not None:
            try:
                x = self.head(x)
                if mode == 'ST':
                    x_ = self.head_(x_)
            except Exception as e:
                return x
        
        if mode == 'ST':
            if self.fusion_mode.find('concat') != -1:
                x = torch.cat([self.gama * F.normalize(x, p=2, dim=1), F.normalize(x_, p=2, dim=1)], 1)
            elif self.fusion_mode.find('add') != -1:
                x = self.gama * F.normalize(x, p=2, dim=1) + F.normalize(x_, p=2, dim=1)

        return x


    def get_feats(self, imgs, num_feats, imgs_lab=None, mode='ST'):
        assert imgs.shape[0] == 1
        batch_step = self.test_cfg.get('batch_step', 10)
        feat_bank = [[] for _ in range(num_feats)]
        clip_len = imgs.size(2)
        imgs = video2images(imgs)
        imgs_lab = video2images(imgs_lab)

        for batch_ptr in range(0, clip_len, batch_step):
            imgs_ = imgs_lab[batch_ptr:batch_ptr + batch_step]
            feats = self.extract_feat(imgs[batch_ptr:batch_ptr +
                                                batch_step], imgs_, mode=mode)
            if isinstance(feats, tuple):
                assert len(feats) == len(feat_bank)
                for i in range(len(feats)):
                    feat_bank[i].append(feats[i].cpu())
            else:
                feat_bank[0].append(feats.cpu())
        for i in range(num_feats):
            feat_bank[i] = images2video(
                torch.cat(feat_bank[i], dim=0), clip_len)
            assert feat_bank[i].size(2) == clip_len

        return feat_bank

    def forward_test(self, imgs, ref_seg_map, img_meta, imgs_lab=None,
                    save_image=False,
                    save_path=None,
                    iteration=None):
        """Defines the computation performed at every call when evaluation and
        testing."""

        imgs = imgs.reshape((-1, ) + imgs.shape[2:])
        imgs_lab = imgs_lab.reshape((-1, ) + imgs_lab.shape[2:])
        clip_len = imgs.size(2)
        # get target shape
        dummy_feat = self.extract_feat(imgs[0:1, :, 0], imgs_lab[0:1, :, 0], mode='T')
        if isinstance(dummy_feat, (list, tuple)):
            feat_shapes = [_.shape for _ in dummy_feat]
        else:
            feat_shapes = [dummy_feat.shape]
        all_seg_preds = []

        for feat_idx, feat_shape in enumerate(feat_shapes):
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

            seg_preds = [ref_seg_map.detach().cpu().numpy()]
            neighbor_range = self.test_cfg.get('neighbor_range', None)
            if neighbor_range is not None:
                spatial_neighbor_mask = spatial_neighbor(
                    feat_shape[0],
                    *feat_shape[2:],
                    neighbor_range=neighbor_range,
                    device=imgs.device,
                    dtype=imgs.dtype,
                    mode='circle').cuda()
            else:
                spatial_neighbor_mask = None

            seg_bank.append(resized_seg_map.cpu())

            if self.fusion_mode.find('early') != -1:
                feat_banks = list([self.get_feats(imgs, len(dummy_feat), imgs_lab)])
            else:
                feats_s = self.get_feats(imgs, len(dummy_feat), imgs_lab, mode='S')
                feats_t = self.get_feats(imgs, len(dummy_feat), imgs_lab, mode='T')
                feat_banks = list([feats_s,feats_t])
 

            for frame_idx in range(1, clip_len):
                key_start = max(0, frame_idx - self.test_cfg.precede_frames)
                seg_logit = self.inference_per_frame(feat_banks, feat_idx, frame_idx, \
                    key_start, seg_bank, imgs, spatial_neighbor_mask)
                  
                if not self.hard_prop:
                    seg_bank.append(seg_logit.cpu())
                else:
                    seg_logit_hard = seg_logit.argmax(1,keepdim=True)
                    seg_logit_hard = F.one_hot(seg_logit_hard)[:,0].permute(0,3,1,2).float()
                    seg_bank.append(seg_logit_hard.cpu())

                seg_pred = F.interpolate(
                    seg_logit,
                    size=img_meta[0]['original_shape'][:2],
                    mode='bilinear',
                    align_corners=False)
                if not input_onehot:
                    if self.norm_mask:
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
            if self.save_np:
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
        if self.save_np:
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


    def inference_per_frame(self, feat_banks, feat_idx, frame_idx, key_start, \
                            seg_bank, imgs, n_mask):
        seg_logits = []
        for feat_bank in feat_banks:
            query_feat = feat_bank[feat_idx][:, :,
                                                frame_idx].to(imgs.device)
            key_feat = feat_bank[feat_idx][:, :, key_start:frame_idx].to(
                imgs.device)
            value_logits = torch.stack(
                seg_bank[key_start:frame_idx], dim=2).to(imgs.device)
            if self.test_cfg.get('with_first', True):
                key_feat = torch.cat([
                    feat_bank[feat_idx][:, :, 0:1].to(imgs.device),
                    key_feat
                ],
                                        dim=2)
                value_logits = cat([
                    seg_bank[0].unsqueeze(2).to(imgs.device), value_logits
                ],
                                    dim=2)
            seg_logit = masked_attention_efficient(
                query_feat,
                key_feat,
                value_logits,
                n_mask,
                temperature=self.test_cfg.temperature,
                topk=self.test_cfg.topk,
                normalize=self.test_cfg.get('with_norm', True),
                non_mask_len=0 if self.test_cfg.get(
                    'with_first_neighbor', True) else 1)

            seg_logits.append(seg_logit)
        
        if len(seg_logits) == 1:
            return seg_logits[0]
        else:
            seg_logit = self.gama * seg_logits[0] + (1-self.gama) * seg_logits[1]
            return seg_logit
