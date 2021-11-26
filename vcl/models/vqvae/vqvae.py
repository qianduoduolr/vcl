import mmcv
from mmcv.runner import auto_fp16, load_checkpoint
from dall_e  import map_pixels, unmap_pixels, load_model

from ..base import BaseModel
from ..components import *
from ..builder import build_backbone, build_loss, build_components
from ..registry import MODELS
from vcl.utils.helpers import *
from vcl.utils import *


import torch.nn as nn
import torch
import torch.nn.functional as F
from roi_align import RoIAlign      # RoIAlign module
from roi_align import CropAndResize # crop_and_resize module

@MODELS.register_module()
class VQVAE(BaseModel):
    """
        Vector Quantized Variational Autoencoder. This networks includes a encoder which maps an
        input image to a discrete latent space, and a decoder to maps the latent map back to the input domain
    """
    def __init__(
        self,
        in_channel=3,
        channel=256,
        n_res_block=2,
        n_res_channel=128,
        downsample=1,
        embed_dim=128,
        n_embed=4096,
        commitment_cost=0.25,
        decay=0.99,
        loss=None,
        newed=False,
        train_cfg=None,
        test_cfg=None,
    ):
        """
        :param in_channel: input channels
        :param channel: convolution channels
        :param n_res_block: number of residual blocks for the encoder and the decoder
        :param n_res_channel: number of intermediate channels of the residual block
        :param downsample: times of downsample and upsample in the encoder and the decoder
        :param embed_dim: embedding dimensions
        :param n_embed: number of embeddings in the codebook
        :param commitment_cost: cost of commitment loss
        :param decay: weight decay for exponential updating
        """
        super().__init__()

        if newed:
            self.enc = Encoder_my(in_channel, channel, n_res_block, n_res_channel, downsample)
            self.dec = Decoder_my(embed_dim, in_channel, channel, n_res_block, n_res_channel, downsample)
        else:
            self.enc = Encoder(in_channel, channel, n_res_block, n_res_channel, downsample)
            self.dec = Decoder(embed_dim, in_channel, channel, n_res_block, n_res_channel, downsample)

        self.quantize_conv = nn.Conv2d(channel, embed_dim, 1)  # Dimension reduction to embedding size
        self.quantize = Quantize(embed_dim, n_embed, commitment_cost, decay)  # Vector quantization

        self.loss = build_loss(loss)


    def encode(self, x):
        """
        Encodes and quantizes an input tensor using VQ-VAE algorithm
        :param x: input tensor
        :return: encoder output, quantized map, commitment loss, codebook indices, codebook embedding
        """
        # Encoding
        enc = self.enc(x)

        # Convolution before quantization and converting to B x H x W x C
        enc = self.quantize_conv(enc).permute(0, 2, 3, 1)

        # Vector quantization
        quant, diff, ind, embed = self.quantize(enc)

        # Converting back the quantized map to B x C x H x W
        quant = quant.permute(0, 3, 1, 2)

        return enc, quant, diff, ind, embed

    def decode(self, quant):
        """
        Decodes quantized map
        :param quant: quantized map
        :return: decoded tensor in input space
        """
        dec = self.dec(quant)  # Decodes to input space

        return dec

    def forward_train(self, imgs, bbox_mask=None, jitter_imgs=None):

        img = imgs[:, 0, 0]

        _, quant, diff, _, _ = self.encode(img)
        dec = self.decode(quant)

        losses = {}

        losses['rec_loss'] = self.loss(dec, img, bbox_mask)
        losses['commit_loss'] = diff

        return losses

    def train_step(self, data_batch, optimizer, progress_ratio):

        # parser loss
        losses = self(**data_batch, test_mode=False)
        loss, log_vars = self.parse_losses(losses)

        # optimizer
        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        log_vars.pop('loss')
        outputs = dict(
            log_vars=log_vars,
            num_samples=len(data_batch['imgs'])
        )

        return outputs


@MODELS.register_module()
class VQCL(BaseModel):
    def __init__(
        self,
        backbone,
        K,
        m=0.999,
        T=0.01,
        embed_dim=128,
        n_embed=4096,
        commitment_cost=0.25,
        decay=0.99,
        loss=None,
        mlp=True,
        train_cfg=None,
        test_cfg=None,
    ):
        super().__init__()

        self.K = K
        self.m = m
        self.T = T
        self.commitment_cost = commitment_cost

        self.quantize = Quantize(embed_dim, n_embed, commitment_cost, decay)  # Vector quantization
        self.backbone = build_backbone(backbone)
        self.backbone_k = build_backbone(backbone)

        crop_height = 4
        crop_width = 4
        self.roi_align = RoIAlign(crop_height, crop_width)
        
        if mlp:  # hack: brute-force replacement
            dim_mlp = self.backbone.feat_dim
            self.backbone.fc = nn.Sequential(nn.Conv2d(dim_mlp, dim_mlp, 1, 1), nn.ReLU(), nn.Conv2d(dim_mlp, embed_dim, 1, 1))
            self.backbone_k.fc = nn.Sequential(nn.Conv2d(dim_mlp, dim_mlp, 1, 1), nn.ReLU(), nn.Conv2d(dim_mlp, embed_dim, 1, 1))

        for param_q, param_k in zip(self.backbone.parameters(), self.backbone_k.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient

        # create the queue
        self.register_buffer("queue", torch.randn(embed_dim, K))
        self.queue = nn.functional.normalize(self.queue, dim=0).cuda()

        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

        self.loss = build_loss(loss)

    
    def forward_train(self, imgs, bboxs=None, jitter_imgs=None):
    
        im_q = imgs[:, 0, 0]
        im_k = imgs[:, 0, 1]

        bbox_q = bboxs[:, 0]
        bbox_k = bboxs[:, 1]

        q = self.backbone(im_q)

        bsz, c, _, _ = q.shape

        bbox_index = torch.arange(bsz,dtype=torch.int).reshape(-1).cuda()

        q = nn.functional.normalize(q, dim=1)

        # Vector quantization
        quant, diff, ind, embed = self.quantize(q.permute(0, 2, 3, 1))

        with torch.no_grad():
            self._momentum_update_key_encoder()

            # shuffle for making use of BN
            im_k, idx_unshuffle = self._batch_shuffle_ddp(im_k.contiguous())

            k = self.backbone_k(im_k)  # keys: NxCxHxW
            k = nn.functional.normalize(k, dim=1)

            # undo shuffle
            k = self._batch_unshuffle_ddp(k, idx_unshuffle)

        qs = self.roi_align(q.contiguous(), bbox_q.float(), bbox_index).reshape(bsz, c, -1)
        ks = self.roi_align(k.contiguous(), bbox_k.float(), bbox_index).reshape(bsz, c, -1)
        
        l_pos = torch.einsum('nci,ncj->nij', [qs, ks]).unsqueeze(-1)
        l_neg = torch.einsum('nci,ck->nik', [qs, self.queue.clone().detach()]).unsqueeze(-2).repeat(1, 1, l_pos.shape[-2], 1)

        logits = torch.cat([l_pos, l_neg], dim=-1).reshape(-1, self.K+1)

        # apply temperature
        logits /= self.T

        # labels: positive key indicators
        labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda().unsqueeze(1)

        # dequeue and enqueue
        self._dequeue_and_enqueue(ks.permute(0,2,1).reshape(-1, c))

        losses = {}

        losses['cts_loss'] = self.loss(logits, labels)
        losses['diff_loss'] = diff * self.commitment_cost

        return losses, diff.item()

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
    

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(self.backbone.parameters(), self.backbone_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        # gather keys before updating queue
        keys = concat_all_gather(keys)

        batch_size = keys.shape[0]

        ptr = int(self.queue_ptr)
        assert self.K % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.queue[:, ptr:ptr + batch_size] = keys.T
        ptr = (ptr + batch_size) % self.K  # move pointer

        self.queue_ptr[0] = ptr

    @torch.no_grad()
    def _batch_shuffle_ddp(self, x):
        """
        Batch shuffle, for making use of BatchNorm.
        *** Only support DistributedDataParallel (DDP) model. ***
        """
        # gather from all gpus
        batch_size_this = x.shape[0]
        x_gather = concat_all_gather(x)
        batch_size_all = x_gather.shape[0]

        num_gpus = batch_size_all // batch_size_this

        # random shuffle index
        idx_shuffle = torch.randperm(batch_size_all).cuda()

        # broadcast to all gpus
        torch.distributed.broadcast(idx_shuffle, src=0)

        # index for restoring
        idx_unshuffle = torch.argsort(idx_shuffle)

        # shuffled index for this gpu
        gpu_idx = torch.distributed.get_rank()
        idx_this = idx_shuffle.view(num_gpus, -1)[gpu_idx]

        return x_gather[idx_this], idx_unshuffle

    @torch.no_grad()
    def _batch_unshuffle_ddp(self, x, idx_unshuffle):
        """
        Undo batch shuffle.
        *** Only support DistributedDataParallel (DDP) model. ***
        """
        # gather from all gpus
        batch_size_this = x.shape[0]
        x_gather = concat_all_gather(x)
        batch_size_all = x_gather.shape[0]

        num_gpus = batch_size_all // batch_size_this

        # restored index for this gpu
        gpu_idx = torch.distributed.get_rank()
        idx_this = idx_unshuffle.view(num_gpus, -1)[gpu_idx]

        return x_gather[idx_this]