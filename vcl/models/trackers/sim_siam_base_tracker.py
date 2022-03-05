from vcl.utils import add_prefix
from .. import builder
from ..common import images2video, video2images
from ..registry import MODELS
from .vanilla_tracker import BaseTracker
from ..vqvae.modules import Quantize
from torch import distributed


@MODELS.register_module()
class SimSiamBaseTracker(BaseTracker):
    """SimSiam framework."""

    def __init__(self, *args, backbone, img_head=None, **kwargs):
        super().__init__(*args, backbone=backbone, **kwargs)
        if img_head is not None:
            self.img_head = builder.build_components(img_head)
        self.init_extra_weights()
        if self.train_cfg is not None:
            self.intra_video = self.train_cfg.get('intra_video', False)
            self.transpose_temporal = self.train_cfg.get(
                'transpose_temporal', False)
        
        self.cts_loss = self.img_head.loss

    @property
    def with_img_head(self):
        """bool: whether the detector has img head"""
        return hasattr(self, 'img_head') and self.img_head is not None

    def init_extra_weights(self):
        if self.with_img_head:
            self.img_head.init_weights()

    def forward_img_head(self, x1, x2, clip_len):
        if isinstance(x1, tuple):
            x1 = x1[-1]
        if isinstance(x2, tuple):
            x2 = x2[-1]
        losses = dict()
        z1, p1 = self.img_head(x1)
        z2, p2 = self.img_head(x2)
        loss_weight = 1. / clip_len if self.intra_video else 1.
        
        
        losses.update(
            add_prefix(
                self.cts_loss(p1, z1, p2, z2, weight=loss_weight),
                prefix='0'))
        
        if self.intra_video:
            z2_v, p2_v = images2video(z2, clip_len), images2video(p2, clip_len)
            for i in range(1, clip_len):
                losses.update(
                    add_prefix(
                        self.cts_loss(
                            p1,
                            z1,
                            video2images(p2_v.roll(i, dims=2)),
                            video2images(z2_v.roll(i, dims=2)),
                            weight=loss_weight),
                        prefix=f'{i}'))
        return losses

    def forward_train(self, imgs, grids=None, label=None):
        # [B, N, C, T, H, W]
        bsz, _, c, _, h, w = imgs.shape     
        imgs = imgs.reshape(bsz, 2, -1, c, h, w).permute(0, 1, 3, 2, 4, 5)   
        
        assert imgs.size(1) == 2
        assert imgs.ndim == 6
        clip_len = imgs.size(3)
        imgs1 = video2images(imgs[:,
                                  0].contiguous().reshape(-1, *imgs.shape[2:]))
        imgs2 = video2images(imgs[:,
                                  1].contiguous().reshape(-1, *imgs.shape[2:]))
        x1 = self.backbone(imgs1)
        x2 = self.backbone(imgs2)
        losses = dict()
        if self.with_img_head:
            loss_img_head = self.forward_img_head(x1, x2, clip_len)
            losses.update(add_prefix(loss_img_head, prefix='img_head'))

        return losses

    def forward_test(self, imgs, **kwargs):
        raise NotImplementedError
    
    
@MODELS.register_module()
class SimaSiam_Vq(SimSiamBaseTracker):

    def __init__(self,
                 embed_dim=128,
                n_embed=4096,
                commitment_cost=0.25,
                decay=0.99,
                *args, 
                **kwargs):
        super().__init__(*args, **kwargs)

        self.commitment_cost = commitment_cost
        self.embed_dim = embed_dim
        self.n_embed = n_embed

        self.quantize = Quantize(embed_dim, n_embed, commitment_cost, decay)  
    
    def train_step(self, data_batch, optimizer, progress_ratio):
    
        # parser loss
        losses, diff = self(**data_batch, test_mode=False)
        loss, log_vars = self.parse_losses(losses)

        # optimizer
        optimizer.zero_grad()
        
        loss.backward()

        optimizer.step()

        log_vars.pop('loss')
        log_vars['diff_item'] = diff
        outputs = dict(
            log_vars=log_vars,
            num_samples=len(data_batch['imgs'])
        )

        return outputs
    
    def encode(self, x):
        """
        Encodes and quantizes an input tensor using VQ-VAE algorithm
        :param x: input tensor
        :return: encoder output, quantized map, commitment loss, codebook indices, codebook embedding
        """
        # Encoding
        enc = self.backbone(x)
        
        bsz, c, _, _ = enc.shape
        
        if c != self.embed_dim:
            q_emb = self.quantize_conv(enc).permute(0, 2, 3, 1)
        else:
            q_emb = enc.permute(0, 2, 3, 1)

        quant, diff, ind, embed = self.quantize(q_emb.contiguous())

        # Converting back the quantized map to B x C x H x W
        quant = quant.permute(0, 3, 1, 2)

        return enc, quant, diff, ind, embed
    
    def forward_train(self, imgs, grids=None, label=None):
            # [B, N, C, T, H, W]
        if self.transpose_temporal:
            imgs = imgs.transpose(1, 3).contiguous()
        assert imgs.size(1) == 2
        assert imgs.ndim == 6
        clip_len = imgs.size(3)
        imgs1 = video2images(imgs[:,
                                  0].contiguous().reshape(-1, *imgs.shape[2:]))
        imgs2 = video2images(imgs[:,
                                  1].contiguous().reshape(-1, *imgs.shape[2:]))
        x1 = self.backbone(imgs1)
        x2 = self.backbone(imgs2)
        
        # vq cluster
        bsz, c, _, _ = x1.shape

        if c != self.embed_dim:
            q_emb = self.quantize_conv(x1)
            k_emb = self.quantize_conv(x2)
        else:
            q_emb = x1
            k_emb = x2
        
        quant, diff, ind, embed = self.quantize(q_emb.permute(0, 2, 3, 1).contiguous())
        
        
        losses = dict()
        if self.with_img_head:
            loss_img_head = self.forward_img_head(x1, x2, clip_len)
            losses.update(add_prefix(loss_img_head, prefix='img_head'))

        return losses, diff