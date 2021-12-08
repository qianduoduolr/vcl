import mmcv
from mmcv.runner import auto_fp16, load_checkpoint
from dall_e  import map_pixels, unmap_pixels, load_model
from torch import distributed

from ..base import BaseModel
from ..components import *
from ..builder import build_backbone, build_loss, build_components
from ..registry import MODELS
from vcl.utils.helpers import *
from vcl.utils import *


import torch.nn as nn
import torch
import torch.nn.functional as F
from torchvision.ops import roi_align


class Quantize(nn.Module):
    """
        Vector Quantization module that performs codebook look up
    """

    def __init__(self, embedding_dim, n_embed, commitment_cost, decay=0.99, eps=1e-5):
        """
        Parameters
        ----------
        :param embedding_dim: code dimension
        :param n_embed: number of embeddings
        :param decay: decay value for codebook exponential moving average update
        :param eps: epsilon value to avoid division by 0
        """
        super().__init__()

        assert 0 <= decay <= 1

        self.dim = embedding_dim
        self.n_embed = n_embed
        self.commitment_cost = commitment_cost
        self.decay = decay
        self.eps = eps

        embed = torch.randn(embedding_dim, n_embed)
        self.register_buffer("embed", embed)
        self.register_buffer("cluster_size", torch.zeros(n_embed))
        self.register_buffer("embed_avg", embed.clone())

    def forward(self, z, distributed=False):
        """
        :param z: Encoder output
        :return: Quantized tensor
        """
        if not distributed:
            flatten = z.reshape(-1, self.dim)  # Converting the z input to a [N x D] tensor, where D is embedding dimension
        else:
            flatten = concat_all_gather(z.reshape(-1, self.dim))
            z = concat_all_gather(z)
        dist = (
                flatten.pow(2).sum(1, keepdim=True)
                - 2 * flatten @ self.embed
                + self.embed.pow(2).sum(0, keepdim=True)
        )  # Distance calculation between Ze and codebook.
        _, embed_ind = (-dist).max(1)  # Arg min of closest distances
        embed_onehot = F.one_hot(embed_ind, self.n_embed).type(flatten.dtype)  # Assigns the actual codes according
        # their closest indices, with flattened
        embed_ind = embed_ind.view(*z.shape[:-1])  # B x H x W tensor with the indices of their nearest code

        quantize = F.embedding(embed_ind, self.embed.transpose(0, 1))  # B x H x W x D quantized tensor

        # Exponential decay updating, as a replacement to codebook loss
        if self.training:
            self.cluster_size.data.mul_(self.decay).add_(
                embed_onehot.sum(0), alpha=1 - self.decay
            )
            embed_sum = flatten.transpose(0, 1) @ embed_onehot
            self.embed_avg.data.mul_(self.decay).add_(embed_sum, alpha=1 - self.decay)
            n = self.cluster_size.sum()
            cluster_size = (
                    (self.cluster_size + self.eps) / (n + self.n_embed * self.eps) * n
            )
            embed_normalized = self.embed_avg / cluster_size.unsqueeze(0)
            self.embed.data.copy_(embed_normalized)

        # Commitment loss, used to keep the encoder output close to the codebook
        diff =  (quantize.detach() - z).pow(2).mean()

        quantize = z + (quantize - z).detach()  # This is added to pass the gradients directly from Z. Basically
        # means that quantization operations have no gradient

        return quantize, diff, embed_ind, self.embed


class ResBlock(nn.Module):
    """
        Residual block with two Convolutional layers
    """
    def __init__(self, in_channel, channel):
        """
        :param in_channel: input channels
        :param channel: intermediate channels of residual block
        """
        super().__init__()

        self.conv = nn.Sequential(
            nn.ELU(),
            nn.Conv2d(in_channel, channel, 3, padding=1),
            nn.ELU(),
            nn.Conv2d(channel, in_channel, 1),
        )

    def forward(self, inp):
        out = self.conv(inp)
        out += inp

        return out


class Encoder(nn.Module):
    """
        Encoder network. It consists a set of convolutional layers followed by N residual blocks.
    """
    def __init__(self, in_channel, channel, n_res_block, n_res_channel, downsample):
        """
        :param in_channel: input channels
        :param channel: convolution channels
        :param n_res_block: number of residual blocks
        :param n_res_channel: number of intermediate layers of the residual block
        :param downsample: times of downsample
        """
        super().__init__()

        if downsample == 1:
            blocks = [
                nn.Conv2d(in_channel, channel, 4, stride=2, padding=1)
            ]

        elif downsample == 2:
            blocks = [
                nn.Conv2d(in_channel, channel//2, 4, stride=2, padding=1),
                nn.ELU(),
                nn.Conv2d(channel//2, channel, 4, stride=2, padding=1)
            ]
        elif downsample == 4:
            blocks = [
                nn.Conv2d(in_channel, channel//2, 4, stride=2, padding=1),
                nn.ELU(),
                nn.Conv2d(channel//2, channel//2, 4, stride=2, padding=1),
                nn.ELU(),
                nn.Conv2d(channel//2, channel, 4, stride=2, padding=1)
            ]

        blocks.append(nn.ELU())
        blocks.append(nn.Conv2d(channel, channel, 3, padding=1))

        for i in range(n_res_block):
            blocks.append(ResBlock(channel, n_res_channel))

        blocks.append(nn.ELU())

        self.blocks = nn.Sequential(*blocks)

    def forward(self, x):
        return self.blocks(x)


class Decoder(nn.Module):
    """
        Decoder network. It consists a convolutional layer, N residual blocks and a deconvolution.
    """
    def __init__(self, in_channel, out_channel, channel, n_res_block, n_res_channel, upsample):
        """
        :param in_channel: input channels
        :param out_channel: output channels
        :param channel: convolution channels
        :param n_res_block: number of residual blocks
        :param n_res_channel: number of intermediate layers of the residual block
        :param upsample: times of upsample
        """
        super().__init__()


        blocks = [nn.Conv2d(in_channel, channel, 3, padding=1)]

        for i in range(n_res_block):
            blocks.append(ResBlock(channel, n_res_channel))

        blocks.append(nn.ELU())

        if upsample == 1:
            blocks.append(nn.ConvTranspose2d(channel, out_channel, 4, stride=2, padding=1))

        elif upsample == 2:
            blocks.append(nn.ConvTranspose2d(channel, channel//2, 4, stride=2, padding=1))
            blocks.append(nn.ELU())
            blocks.append(nn.ConvTranspose2d(channel//2, out_channel, 4, stride=2, padding=1))
        elif upsample == 4:
            blocks.append(nn.ConvTranspose2d(channel, channel//2, 4, stride=2, padding=1))
            blocks.append(nn.ELU())
            blocks.append(nn.ConvTranspose2d(channel//2, channel//2, 4, stride=2, padding=1))
            blocks.append(nn.ELU())
            blocks.append(nn.ConvTranspose2d(channel//2, out_channel, 4, stride=2, padding=1))

        self.blocks = nn.Sequential(*blocks)

    def forward(self, z):
        return self.blocks(z)

class Encoder_my(nn.Module):
    """
        Encoder network. It consists a set of convolutional layers followed by N residual blocks.
    """
    def __init__(self, in_channel, channel, n_res_block, n_res_channel, downsample):
        """
        :param in_channel: input channels
        :param channel: convolution channels
        :param n_res_block: number of residual blocks
        :param n_res_channel: number of intermediate layers of the residual block
        :param downsample: times of downsample
        """
        super().__init__()

        if downsample == 1:
            blocks = [
                nn.Conv2d(in_channel, channel // 2, 4, stride=2, padding=1)
            ]

        elif downsample == 2:
            blocks = [
                nn.Conv2d(in_channel, channel//2, 4, stride=2, padding=1),
                nn.ELU(),
                nn.Conv2d(channel//2, channel//2, 4, stride=2, padding=1)
            ]
        elif downsample == 4:
            blocks = [
                nn.Conv2d(in_channel, channel//2, 4, stride=2, padding=1),
                nn.ELU(),
                nn.Conv2d(channel//2, channel//2, 4, stride=2, padding=1),
                nn.ELU(),
                nn.Conv2d(channel//2, channel//2, 4, stride=2, padding=1)
            ]

        blocks.append(nn.ELU())
        blocks.append(nn.Conv2d(channel//2, channel, 3, padding=1))

        for i in range(n_res_block):
            blocks.append(ResBlock(channel, n_res_channel))

        blocks.append(nn.ELU())

        self.blocks = nn.Sequential(*blocks)

    def forward(self, x):
        return self.blocks(x)


class Decoder_my(nn.Module):
    """
        Decoder network. It consists a convolutional layer, N residual blocks and a deconvolution.
    """
    def __init__(self, in_channel, out_channel, channel, n_res_block, n_res_channel, upsample):
        """
        :param in_channel: input channels
        :param out_channel: output channels
        :param channel: convolution channels
        :param n_res_block: number of residual blocks
        :param n_res_channel: number of intermediate layers of the residual block
        :param upsample: times of upsample
        """
        super().__init__()


        blocks = [nn.Conv2d(in_channel, channel, 3, padding=1)]

        for i in range(n_res_block):
            blocks.append(ResBlock(channel, n_res_channel))

        blocks.append(nn.ELU())

        if upsample == 1:
            blocks.append(nn.ConvTranspose2d(channel, out_channel, 4, stride=2, padding=1))

        elif upsample == 2:
            blocks.append(nn.ConvTranspose2d(channel, channel//2, 4, stride=2, padding=1))
            blocks.append(nn.ELU())
            blocks.append(nn.ConvTranspose2d(channel//2, out_channel, 4, stride=2, padding=1))
        elif upsample == 4:
            blocks.append(nn.ConvTranspose2d(channel, channel//2, 4, stride=2, padding=1))
            blocks.append(nn.ELU())
            blocks.append(nn.ConvTranspose2d(channel//2, channel//2, 4, stride=2, padding=1))
            blocks.append(nn.ELU())
            blocks.append(nn.ConvTranspose2d(channel//2, out_channel, 4, stride=2, padding=1))

        self.blocks = nn.Sequential(*blocks)

    def forward(self, z):
        return self.blocks(z)

@MODELS.register_module()
class VQVAE(nn.Module):
    """
        Vector Quantized Variational Autoencoder. This networks includes a encoder which maps an
        input image to a discrete latent space, and a decoder to maps the latent map back to the input domain
    """
    def __init__(
        self,
        in_channel=3,
        channel=128,
        n_res_block=2,
        n_res_channel=64,
        downsample=1,
        embed_dim=64,
        n_embed=4096,
        commitment_cost=0.25,
        decay=0.99,
        newed=False,
        train_cfg=None,
        test_cfg=None
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
        

    def forward(self, x):
        _, quant, diff, _, _ = self.encode(x)
        dec = self.decode(quant)

        return dec, diff

    def encode(self, x, encoding=True):
        """
        Encodes and quantizes an input tensor using VQ-VAE algorithm
        :param x: input tensor
        :return: encoder output, quantized map, commitment loss, codebook indices, codebook embedding
        """
        # Encoding
        if encoding:
            enc = self.enc(x)
            # Convolution before quantization and converting to B x H x W x C
            enc = self.quantize_conv(enc).permute(0, 2, 3, 1)
        else:
            enc = x

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


# @MODELS.register_module()
# class VQVAE(BaseModel):
#     """
#         Vector Quantized Variational Autoencoder. This networks includes a encoder which maps an
#         input image to a discrete latent space, and a decoder to maps the latent map back to the input domain
#     """
#     def __init__(
#         self,
#         in_channel=3,
#         channel=256,
#         n_res_block=2,
#         n_res_channel=128,
#         downsample=1,
#         embed_dim=128,
#         n_embed=4096,
#         commitment_cost=0.25,
#         decay=0.99,
#         loss=None,
#         newed=False,
#         train_cfg=None,
#         test_cfg=None,
#     ):
#         """
#         :param in_channel: input channels
#         :param channel: convolution channels
#         :param n_res_block: number of residual blocks for the encoder and the decoder
#         :param n_res_channel: number of intermediate channels of the residual block
#         :param downsample: times of downsample and upsample in the encoder and the decoder
#         :param embed_dim: embedding dimensions
#         :param n_embed: number of embeddings in the codebook
#         :param commitment_cost: cost of commitment loss
#         :param decay: weight decay for exponential updating
#         """
#         super().__init__()

#         if newed:
#             self.enc = Encoder_my(in_channel, channel, n_res_block, n_res_channel, downsample)
#             self.dec = Decoder_my(embed_dim, in_channel, channel, n_res_block, n_res_channel, downsample)
#         else:
#             self.enc = Encoder(in_channel, channel, n_res_block, n_res_channel, downsample)
#             self.dec = Decoder(embed_dim, in_channel, channel, n_res_block, n_res_channel, downsample)

#         self.quantize_conv = nn.Conv2d(channel, embed_dim, 1)  # Dimension reduction to embedding size
#         self.quantize = Quantize(embed_dim, n_embed, commitment_cost, decay)  # Vector quantization

#         self.loss = build_loss(loss)


#     def encode(self, x):
#         """
#         Encodes and quantizes an input tensor using VQ-VAE algorithm
#         :param x: input tensor
#         :return: encoder output, quantized map, commitment loss, codebook indices, codebook embedding
#         """
#         # Encoding
#         enc = self.enc(x)

#         # Convolution before quantization and converting to B x H x W x C
#         enc = self.quantize_conv(enc).permute(0, 2, 3, 1)

#         # Vector quantization
#         quant, diff, ind, embed = self.quantize(enc)

#         # Converting back the quantized map to B x C x H x W
#         quant = quant.permute(0, 3, 1, 2)

#         return enc, quant, diff, ind, embed

#     def decode(self, quant):
#         """
#         Decodes quantized map
#         :param quant: quantized map
#         :return: decoded tensor in input space
#         """
#         dec = self.dec(quant)  # Decodes to input space

#         return dec

#     def forward_train(self, imgs, bbox_mask=None, jitter_imgs=None):

#         img = imgs[:, 0, 0]

#         _, quant, diff, _, _ = self.encode(img)
#         dec = self.decode(quant)

#         losses = {}

#         losses['rec_loss'] = self.loss(dec, img, bbox_mask)
#         losses['commit_loss'] = diff

#         return losses

#     def train_step(self, data_batch, optimizer, progress_ratio):

#         # parser loss
#         losses = self(**data_batch, test_mode=False)
#         loss, log_vars = self.parse_losses(losses)

#         # optimizer
#         optimizer.zero_grad()

#         loss.backward()

#         optimizer.step()

#         log_vars.pop('loss')
#         outputs = dict(
#             log_vars=log_vars,
#             num_samples=len(data_batch['imgs'])
#         )

#         return outputs


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
        cluster=False,
        train_cfg=None,
        test_cfg=None,
        pretrained=None
    ):
        super().__init__()

        self.K = K
        self.m = m
        self.T = T
        self.commitment_cost = commitment_cost
        self.cluster = cluster

        self.quantize = Quantize(embed_dim, n_embed, commitment_cost, decay)  # Vector quantization
        self.backbone = build_backbone(backbone)
        self.backbone_k = build_backbone(backbone)

        crop_height = 4
        crop_width = 4
        # self.roi_align = roi_align(crop_height, crop_width)
        
        if mlp:  # hack: brute-force replacement
            dim_mlp = self.backbone.feat_dim
            self.backbone.fc = nn.Sequential(nn.Conv2d(dim_mlp, dim_mlp, 1, 1), nn.ReLU(), nn.Conv2d(dim_mlp, embed_dim, 1, 1))
            self.backbone_k.fc = nn.Sequential(nn.Conv2d(dim_mlp, dim_mlp, 1, 1), nn.ReLU(), nn.Conv2d(dim_mlp, embed_dim, 1, 1))

        for param_q, param_k in zip(self.backbone.parameters(), self.backbone_k.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient

        if cluster:
            self.quantize_conv = nn.Conv2d(self.backbone.feat_dim, embed_dim, 1)  # Dimension reduction to embedding size

        # create the queue
        self.register_buffer("queue", torch.randn(embed_dim, K))
        self.queue = nn.functional.normalize(self.queue, dim=0).cuda()

        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

        self.loss = build_loss(loss)

        if pretrained is not None:
            self.init_weights(pretrained)

    
    def init_weights(self, pretrained):
        self.backbone.pretrained = pretrained
        self.backbone.init_weights()

    def forward(self, test_mode, **kwargs):
        """Forward function for base model.

        Args:
            imgs (Tensor): Input image(s).
            labels (Tensor): Ground-truth label(s).
            test_mode (bool): Whether in test mode.
            kwargs (dict): Other arguments.

        Returns:
            Tensor: Forward results.
        """

        if test_mode:
            return self.forward_test(**kwargs)

        if not self.cluster:
            return self.forward_train(**kwargs)
        else:
            return self.forward_cluster(**kwargs)

    def forward_cluster(self, imgs, bboxs=None, jitter_imgs=None):
        im_q = imgs[:, 0, 0]

        with torch.no_grad():
            q = self.backbone(im_q)

        bsz, c, _, _ = q.shape
        # q = self.quantize_conv(q)

        # Vector quantization
        quant, diff, ind, embed = self.quantize(q.permute(0, 2, 3, 1))

        losses = {}

        losses['diff_loss'] = diff * self.commitment_cost

        print(diff.item())

        return losses, diff.item()


    def forward_train(self, imgs, bboxs=None, jitter_imgs=None):
    
        im_q = imgs[:, 0, 0]
        im_k = imgs[:, 0, 1]

        bbox_q = bboxs[:, 0]
        bbox_k = bboxs[:, 1]

        q = self.backbone(im_q)

        bsz, c, _, _ = q.shape

        bbox_index = torch.arange(bsz,dtype=torch.float).reshape(-1,1).cuda()

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

        bboxs_q = torch.cat([bbox_index, bbox_q.float()], dim=1)
        bboxs_k = torch.cat([bbox_index, bbox_k.float()], dim=1)

        qs = roi_align(q.contiguous(), bboxs_q.float(), output_size=(4,4)).reshape(bsz, c, -1)
        ks = roi_align(k.contiguous(), bboxs_k.float(), output_size=(4,4)).reshape(bsz, c, -1)
        
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
        for k,opz in optimizer.items():
            opz.zero_grad()

        loss.backward()
        for k,opz in optimizer.items():
            opz.step()

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

    def encode(self, x):
        """
        Encodes and quantizes an input tensor using VQ-VAE algorithm
        :param x: input tensor
        :return: encoder output, quantized map, commitment loss, codebook indices, codebook embedding
        """
        # Encoding
        enc = self.backbone(x).permute(0, 2, 3, 1)

        # Vector quantization
        quant, diff, ind, embed = self.quantize(enc)

        # Converting back the quantized map to B x C x H x W
        quant = quant.permute(0, 3, 1, 2)

        return enc, quant, diff, ind, embed

@MODELS.register_module()
class VQCL_v2(BaseModel):
    def __init__(
        self,
        backbone,
        embed_dim=128,
        n_embed=4096,
        commitment_cost=0.25,
        decay=0.99,
        sim_siam_head=None,
        loss=None,

        train_cfg=None,
        test_cfg=None,
        pretrained=None
    ):
        super().__init__()

        self.commitment_cost = commitment_cost
        self.embed_dim = embed_dim

        self.quantize = Quantize(embed_dim, n_embed, commitment_cost, decay)  # Vector quantization
        self.backbone = build_backbone(backbone)
        self.quantize_conv = nn.Conv2d(self.backbone.feat_dim, embed_dim, 1)  # Dimension reduction to embedding size

        self.cts_loss = build_loss(loss)

        if sim_siam_head is not None:
            self.head = build_components(sim_siam_head)
            self.head.init_weights()
        else:
            self.head = None

        if pretrained is not None:
            self.init_weights(pretrained)



    def forward_train(self, imgs, bboxs=None, jitter_imgs=None):
    
        im_q = imgs[:, 0, 0]
        im_k = imgs[:, 0, 1]

        q = self.backbone(im_q)
        k = self.backbone(im_k)

        bsz, c, _, _ = q.shape

        q_emb = self.quantize_conv(q).permute(0, 2, 3, 1)
        quant, diff, ind, embed = self.quantize(q_emb.contiguous(), distributed=False)
    
        losses = {}

        losses['cts_loss'] = self.forward_img_head(q, k)
        losses['diff'] = diff * self.commitment_cost

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
    

    def encode(self, x):
        """
        Encodes and quantizes an input tensor using VQ-VAE algorithm
        :param x: input tensor
        :return: encoder output, quantized map, commitment loss, codebook indices, codebook embedding
        """
        # Encoding
        enc = self.backbone(x)

        q_emb = self.quantize_conv(enc).permute(0, 2, 3, 1)
        quant, diff, ind, embed = self.quantize(q_emb.contiguous(), distributed=False)

        # Converting back the quantized map to B x C x H x W
        quant = quant.permute(0, 3, 1, 2)

        return enc, quant, diff, ind, embed

    def forward_img_head(self, x1, x2):

        if isinstance(x1, tuple):
            x1 = x1[-1]
        if isinstance(x2, tuple):
            x2 = x2[-1]

        z1, p1 = self.head(x1)
        z2, p2 = self.head(x2)
        
        loss = self.cts_loss(p1, z2.detach()) * 0.5 + self.cts_loss(p2, z1.detach()) * 0.5

        return loss

@MODELS.register_module()
class VQCL_v3(VQCL_v2):
    def __init__(
        self,
        sim_siam_head_quant=None,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.head_quant = build_components(sim_siam_head_quant)
        self.head_quant.init_weights()

    def forward_train(self, imgs, bboxs=None, jitter_imgs=None):
    
        im_q = imgs[:, 0, 0]
        im_k = imgs[:, 0, 1]

        q = self.backbone(im_q)
        k = self.backbone(im_k)

        bsz, c, _, _ = q.shape

        q_emb = self.quantize_conv(q).permute(0, 2, 3, 1)
        quant_q, diff1, ind, embed = self.quantize(q_emb.contiguous(), distributed=False)
        quant_q = quant_q.permute(0, 3, 1, 2)

        k_emb = self.quantize_conv(k).permute(0, 2, 3, 1)
        quant_k, diff2, ind, embed = self.quantize(k_emb.contiguous(), distributed=False)
        quant_k = quant_k.permute(0, 3, 1, 2)
        
        diff = diff1 + diff2

        losses = {}

        losses['cts_loss'] = self.forward_img_head(q, k, self.head)
        losses['cts_quant_loss'] = self.forward_img_head(quant_q, quant_k, self.head_quant)
        losses['diff'] = diff * self.commitment_cost

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

    def forward_img_head(self, x1, x2, head):

        if isinstance(x1, tuple):
            x1 = x1[-1]
        if isinstance(x2, tuple):
            x2 = x2[-1]

        z1, p1 = head(x1)
        z2, p2 = head(x2)
        
        loss = self.cts_loss(p1, z2.detach()) * 0.5 + self.cts_loss(p2, z1.detach()) * 0.5

        return loss
    

@MODELS.register_module()
class VQCL_v4(VQCL_v2):
    def __init__(
        self,
        in_channel=3,
        channel=256,
        n_res_block=2,
        n_res_channel=128,
        mse_loss=None,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.dec = Decoder(self.embed_dim, in_channel, channel, n_res_block, n_res_channel, 4)
        self.mse_loss = build_loss(mse_loss)

    def forward_train(self, imgs, bboxs=None, jitter_imgs=None):
    
        im_q = imgs[:, 0, 0]
        im_k = imgs[:, 0, 1]

        q = self.backbone(im_q)
        k = self.backbone(im_k)

        bsz, c, _, _ = q.shape

        q_emb = self.quantize_conv(q).permute(0, 2, 3, 1)
        quant_q, diff1, ind, embed = self.quantize(q_emb.contiguous(), distributed=False)
        quant_q = quant_q.permute(0, 3, 1, 2)

        k_emb = self.quantize_conv(k).permute(0, 2, 3, 1)
        quant_k, diff2, ind, embed = self.quantize(k_emb.contiguous(), distributed=False)
        quant_k = quant_k.permute(0, 3, 1, 2)

        dec_q = self.dec(quant_q)
        dec_k = self.dec(quant_k)
        
        diff = diff1 + diff2

        losses = {}

        losses['cts_loss'] = self.forward_img_head(q, k, self.head)
        losses['rec_loss'] = self.mse_loss(dec_q, im_q) + self.mse_loss(dec_k, im_k)
        losses['diff'] = diff * self.commitment_cost

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

    def forward_img_head(self, x1, x2, head):

        if isinstance(x1, tuple):
            x1 = x1[-1]
        if isinstance(x2, tuple):
            x2 = x2[-1]

        z1, p1 = head(x1)
        z2, p2 = head(x2)
        
        loss = self.cts_loss(p1, z2.detach()) * 0.5 + self.cts_loss(p2, z1.detach()) * 0.5

        return loss
    