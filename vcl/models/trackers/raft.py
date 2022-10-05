# Copyright (c) OpenMMLab. All rights reserved.
from lib2to3.pytree import Base
from typing import Dict, Optional, Sequence, Tuple

import torch
import torch.nn as nn
from numpy import ndarray

from ..builder import MODELS, build_backbone, build_components, build_loss
from .base import BaseModel


@MODELS.register_module()
class RAFT(BaseModel):
    """RAFT model.
    Args:
        num_levels (int): Number of levels in .
        radius (int): Number of radius in  .
        cxt_channels (int): Number of channels of context feature.
        h_channels (int): Number of channels of hidden feature in .
        cxt_encoder (dict): Config dict for building context encoder.
        freeze_bn (bool, optional): Whether to freeze batchnorm layer or not.
            Default: False.
    """

    def __init__(self,
                 backbone, 
                 decoder,
                 cxt_backbone,
                 loss,
                 num_levels: int,
                 radius: int,
                 cxt_channels: int,
                 h_channels: int,
                 freeze_bn: bool = False,
                 *args,
                 **kwargs):
        super().__init__(*args, **kwargs)

        self.encoder = build_backbone(backbone)
        self.decoder = build_components(decoder)

        self.num_levels = num_levels
        self.radius = radius
        self.context = build_backbone(cxt_backbone)
        self.h_channels = h_channels
        self.cxt_channels = cxt_channels

        self.loss = build_loss(loss)

        assert self.num_levels == self.decoder.num_levels
        assert self.radius == self.decoder.radius
        assert self.h_channels == self.decoder.h_channels
        assert self.cxt_channels == self.decoder.cxt_channels
        assert self.h_channels + self.cxt_channels == self.context.out_channels

        if freeze_bn:
            self.freeze_bn()

    def freeze_bn(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    def extract_feat(
        self, imgs: torch.Tensor
    ):
    
        """Extract features from images.
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

        feat1 = self.encoder(img1)
        feat2 = self.encoder(img2)
        cxt_feat = self.context(img1)

        h_feat, cxt_feat = torch.split(
            cxt_feat, [self.h_channels, self.cxt_channels], dim=1)
        h_feat = torch.tanh(h_feat)
        cxt_feat = torch.relu(cxt_feat)

        return feat1, feat2, h_feat, cxt_feat

    def forward_train(
            self,
            imgs: torch.Tensor,
            flows: torch.Tensor,
            valid: torch.Tensor = None,
            flow_init: Optional[torch.Tensor] = None,
            img_metas: Optional[Sequence[dict]] = None
    ):
        """Forward function for RAFT when model training.
        Args:
            imgs (Tensor): The concatenated input images.
            flow_gt (Tensor): The ground truth of optical flow.
                Defaults to None.
            valid (Tensor, optional): The valid mask. Defaults to None.
            flow_init (Tensor, optional): The initialized flow when warm start.
                Default to None.
            img_metas (Sequence[dict], optional): meta data of image to revert
                the flow to original ground truth size. Defaults to None.
        Returns:
            Dict[str, Tensor]: The losses of output.
        """

        feat1, feat2, h_feat, cxt_feat = self.extract_feat(imgs)
        B, _, H, W = feat1.shape

        if flow_init is None:
            flow_init = torch.zeros((B, 2, H, W), device=feat1.device)

        pred, _, _ = self.decoder(
            False,
            feat1,
            feat2,
            flow=flow_init,
            h_feat=h_feat,
            cxt_feat=cxt_feat,
            valid=valid,
            )

        losses = {}

        losses['flow_loss'] = self.loss(pred, flows[:,0,:,0])

        return losses        

    def forward_test(
            self,
            imgs: torch.Tensor,
            flow_init: Optional[torch.Tensor] = None,
            return_lr: bool = False,
            img_metas: Optional[Sequence[dict]] = None):
        """Forward function for RAFT when model testing.
        Args:
            imgs (Tensor): The concatenated input images.
            flow_init (Tensor, optional): The initialized flow when warm start.
                Default to None.
            img_metas (Sequence[dict], optional): meta data of image to revert
                the flow to original ground truth size. Defaults to None.
        Returns:
            Sequence[Dict[str, ndarray]]: the batch of predicted optical flow
                with the same size of images after augmentation.
        """
        train_iter = self.decoder.iters
        if self.test_cfg is not None and self.test_cfg.get(
                'iters') is not None:
            self.decoder.iters = self.test_cfg.get('iters')

        feat1, feat2, h_feat, cxt_feat = self.extract_feat(imgs)
        B, _, H, W = feat1.shape

        if flow_init is None:
            flow_init = torch.zeros((B, 2, H, W), device=feat1.device)

        results = self.decoder(
            test_mode=True,
            feat1=feat1,
            feat2=feat2,
            flow=flow_init,
            h_feat=h_feat,
            cxt_feat=cxt_feat,
            img_metas=img_metas,
            return_lr=return_lr
            )
        # recover iter in train
        self.decoder.iters = train_iter

        return results