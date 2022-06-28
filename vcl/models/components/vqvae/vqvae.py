from base64 import encode
from email.utils import encode_rfc2231
import mmcv
from mmcv.runner import auto_fp16, load_checkpoint
from dall_e  import map_pixels, unmap_pixels, load_model
from torch import distributed

from ...base import BaseModel

from ...builder import build_backbone, build_loss, build_components
from ...registry import COMPONENTS
from vcl.utils.helpers import *
from vcl.models.common import *
from .modules import *


import torch.nn as nn
import torch
import torch.nn.functional as F
from torchvision.ops import roi_align

@COMPONENTS.register_module()
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
        train_cfg=None,
        test_cfg=None,
        pretrained=None
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
        self.enc = Encoder(in_channel, channel, n_res_block, n_res_channel, downsample)
        self.dec = Decoder(embed_dim, in_channel, channel, n_res_block, n_res_channel, downsample)

        self.quantize_conv = nn.Conv2d(channel, embed_dim, 1)  # Dimension reduction to embedding size
        self.quantize = Quantize(embed_dim, n_embed, commitment_cost, decay)  # Vector quantization
        self.commitment_cost = commitment_cost
        self.n_embed = n_embed
        self.embed_dim = embed_dim
        
        self.loss = build_loss(loss) if loss is not None else None
        self.init_weights(pretrained)

    def init_weights(self, pretrained): 
        
        if pretrained is not None:
            print('load pretrained')
            _ = load_checkpoint(self, pretrained, map_location='cpu')

        embed = torch.randn(self.embed_dim, self.n_embed)
        self.quantize.register_buffer("embed", embed)
        self.quantize.register_buffer("cluster_size", torch.zeros(self.n_embed))
        self.quantize.register_buffer("embed_avg", embed.clone())

    def encode(self, x):
        """
        Encodes and quantizes an input tensor using VQ-VAE algorithm
        :param x: input tensor
        :return: encoder output, quantized map, commitment loss, codebook indices, codebook embedding
        """
        # Encoding
        enc_ = self.enc(x)
        
        # Convolution before quantization and converting to B x H x W x C
        enc = self.quantize_conv(enc_).permute(0, 2, 3, 1)

        # Vector quantization
        quant, diff, ind, embed = self.quantize(enc)

        # Converting back the quantized map to B x C x H x W
        quant = quant.permute(0, 3, 1, 2)

        return enc_, quant, diff, ind, embed

    def decode(self, quant):
        """
        Decodes quantized map
        :param quant: quantized map
        :return: decoded tensor in input space
        """
        dec = self.dec(quant)  # Decodes to input space

        return dec

    def forward_train(self, imgs):

        img = imgs[:, 0, 0]

        _, quant, diff, _, _ = self.encode(img)
        dec = self.decode(quant)

        losses = {}

        losses['rec_loss'] = self.loss(dec, img)
        losses['commit_loss'] = self.commitment_cost * diff

        return losses


@COMPONENTS.register_module()
class VQ(BaseModel):
    """
        Vector Quantized (Only quantization means K-means for the given features)
    """
    def __init__(
        self,
        channel=256,
        embed_dim=128,
        n_embed=4096,
        commitment_cost=0.25,
        decay=0.99,
        loss=None,
        train_cfg=None,
        test_cfg=None,
        pretrained=None
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

        if channel == embed_dim:
            self.quantize_conv = None
        else:
            self.quantize_conv = nn.Conv2d(channel, embed_dim, 1)  # Dimension reduction to embedding size
        self.quantize = Quantize(embed_dim, n_embed, commitment_cost, decay)  # Vector quantization
        self.commitment_cost = commitment_cost
        self.n_embed = n_embed
        self.embed_dim = embed_dim
        
        self.loss = build_loss(loss) if loss is not None else None
        self.init_weights(pretrained)

    def init_weights(self, pretrained): 
        
        if pretrained is not None:
            print('load pretrained')
            _ = load_checkpoint(self, pretrained, map_location='cpu')

        embed = torch.randn(self.embed_dim, self.n_embed)
        self.quantize.register_buffer("embed", embed)
        self.quantize.register_buffer("cluster_size", torch.zeros(self.n_embed))
        self.quantize.register_buffer("embed_avg", embed.clone())

    def encode(self, x):
        """
        Encodes and quantizes an input tensor using VQ-VAE algorithm
        :param x: input tensor
        :return: encoder output, quantized map, commitment loss, codebook indices, codebook embedding
        """
        # Encoding
        # Convolution before quantization and converting to B x H x W x C
        if self.quantize_conv is not None:
            enc = self.quantize_conv(x).permute(0, 2, 3, 1)
        else:
            enc = x.permute(0, 2, 3, 1)

        # Vector quantization
        quant, diff, ind, embed = self.quantize(enc)

        # Converting back the quantized map to B x C x H x W
        quant = quant.permute(0, 3, 1, 2)

        return quant, ind, diff

    def decode(self, quant):
        """
        Decodes quantized map
        :param quant: quantized map
        :return: decoded tensor in input space
        """
        dec = self.dec(quant)  # Decodes to input space

        return dec

    def forward(self, x):
        if self.quantize_conv is not None:
            enc = self.quantize_conv(x).permute(0, 2, 3, 1)
        else:
            enc = x.permute(0, 2, 3, 1)

        # Vector quantization
        quant, diff, ind, embed = self.quantize(enc)

        return diff