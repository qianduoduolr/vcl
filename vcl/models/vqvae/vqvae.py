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
