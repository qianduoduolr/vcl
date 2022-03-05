from base64 import encode
from email.utils import encode_rfc2231
import mmcv
from mmcv.runner import auto_fp16, load_checkpoint
from dall_e  import map_pixels, unmap_pixels, load_model
from torch import distributed

from ..base import BaseModel
from ..components import *
from ..builder import build_backbone, build_loss, build_components
from ..registry import MODELS
from vcl.utils.helpers import *
from vcl.models.common import *
from .modules import *


import torch.nn as nn
import torch
import torch.nn.functional as F
from torchvision.ops import roi_align

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

        if newed:
            self.enc = Encoder_my(in_channel, channel, n_res_block, n_res_channel, downsample)
            self.dec = Decoder_my(embed_dim, in_channel, channel, n_res_block, n_res_channel, downsample)
        else:
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


@MODELS.register_module()
class VQCL_v2(BaseModel):
    
    """Visual Quantilization base on Simsiam 
    """
    
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
        self.n_embed = n_embed

        self.quantize = Quantize(embed_dim, n_embed, commitment_cost, decay)  # Vector quantization
    
        self.backbone = build_backbone(backbone)
        self.quantize_conv = nn.Conv2d(self.backbone.feat_dim, embed_dim, 1)  # Dimension reduction to embedding size

        self.cts_loss = build_loss(loss)

        if sim_siam_head is not None:
            self.head = build_components(sim_siam_head)
            self.head.init_weights()
        else:
            self.head = None
        
        self.init_weights(pretrained)
            
    def init_weights(self, pretrained):
        
        self.backbone.init_weights()
        if pretrained is not None:
            _ = load_checkpoint(self, pretrained, map_location='cpu')
    

    def forward_train(self, imgs):
        
        im_q = imgs[:, 0, 0]
        im_k = imgs[:, 0, 1]

        q = self.backbone(im_q)
        k = self.backbone(im_k)

        bsz, c, _, _ = q.shape

        q_emb = self.quantize_conv(q).permute(0, 2, 3, 1)

        
        quant, diff, ind, embed = self.quantize(q_emb.contiguous())
    
        losses = {}
        losses['cts_loss'] = self.forward_img_head(q, k)
        losses['diff'] = diff * self.commitment_cost

        return losses, diff.item()

    def train_step(self, data_batch, optimizer, progress_ratio):

        # parser loss
        losses, diff = self(**data_batch, test_mode=False)
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
class VQCL_v5(VQCL_v2):
    """V2
    """
    def forward_train(self, imgs):
        
        im_q = imgs[:, 0, 0]
        im_k = imgs[:, 0, 1]

        q = self.backbone(im_q)
        k = self.backbone(im_k)

        bsz, c, _, _ = q.shape

        if c != self.embed_dim:
            q_emb = self.quantize_conv(q)
            k_emb = self.quantize_conv(k)
        else:
            q_emb = q
            k_emb = k
        
        quant, diff, ind, embed = self.quantize(q_emb.permute(0, 2, 3, 1).contiguous())
    
        losses = {}
        losses['cts_loss'] = self.forward_img_head(q_emb, k_emb)
        losses['diff'] = diff * self.commitment_cost
        

        return losses, diff.item()
    
    
@MODELS.register_module()
class VQCL_v6(VQCL_v2):
    
    """same with VFS (ICCV2021)
    """

    def forward_train(self, imgs):
                
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
    
        if isinstance(x1, tuple):
            q_emb = x1[0]
            k_emb = x2[0]
        else:
            c = x1.shape[1]
            
            if c != self.embed_dim:
                q_emb = self.quantize_conv(x1)
                k_emb = self.quantize_conv(x2)
            else:
                q_emb = x1
                k_emb = x2
                
        quant, diff, ind, embed = self.quantize(q_emb.permute(0, 2, 3, 1).contiguous())
        
        losses = dict()

        loss_img_head = self.forward_img_head(q_emb, k_emb, clip_len) if not isinstance(x1, tuple) else self.forward_img_head(x1[-1], x2[-1], clip_len)
        losses.update(add_prefix(loss_img_head, prefix='img_head'))
        
        losses['diff'] = diff * self.commitment_cost
        
        return losses, diff.item()
    

    def forward_img_head(self, x1, x2, clip_len):
        
        if isinstance(x1, tuple):
            x1 = x1[-1]
        if isinstance(x2, tuple):
            x2 = x2[-1]
        losses = dict()
        z1, p1 = self.head(x1)
        z2, p2 = self.head(x2)
        loss_weight = 1. / clip_len 
        losses.update(
            add_prefix(
                self.head_loss(p1, z1, p2, z2, weight=loss_weight),
                prefix='0'))

        z2_v, p2_v = images2video(z2, clip_len), images2video(p2, clip_len)
        for i in range(1, clip_len):
            losses.update(
                add_prefix(
                    self.head_loss(
                        p1,
                        z1,
                        video2images(p2_v.roll(i, dims=2)),
                        video2images(z2_v.roll(i, dims=2)),
                        weight=loss_weight),
                    prefix=f'{i}'))
        return losses
    
    def head_loss(self, p1, z1, p2, z2, mask12=None, mask21=None, weight=1.):
        assert mask12 is None
        assert mask21 is None

        losses = dict()

        loss_feat = self.cts_loss(p1, z2.detach()) * 0.5 + self.cts_loss(
            p2, z1.detach()) * 0.5
        losses['loss_feat'] = loss_feat * weight
        return losses
    


@MODELS.register_module()
class VQCL_v7(VQCL_v2):


    def forward_train(self, imgs):         
        
        im_q = imgs[:, 0, 0]
        im_k = imgs[:, 0, 1]

        q_emb, q = self.backbone(im_q)
        k_emb, k = self.backbone(im_k)
        
        bsz, c, _, _ = q.shape
        
        quant, diff, ind, embed = self.quantize(q_emb.permute(0, 2, 3, 1).contiguous())
    
        losses = {}
        losses['cts_loss'] = self.forward_img_head(q, k)
        losses['diff'] = diff * self.commitment_cost
        

        return losses, diff.item()
    



@MODELS.register_module()
class VQCL_v8(VQCL_v2):
    """V2
    """
    def forward_train(self, imgs):
        
        im_q = imgs[:, 0, 0]
        im_k = imgs[:, 0, 1]

        q = self.backbone(im_q)
        k = self.backbone(im_k)

        bsz, c, _, _ = q.shape

        if c != self.embed_dim:
            q_emb = self.quantize_conv(q)
            k_emb = self.quantize_conv(k)
        else:
            q_emb = q
            k_emb = k
        
        quant, diff, ind, embed = self.quantize(q_emb.permute(0, 2, 3, 1).contiguous())
    
        losses = {}
        losses['cts_loss'] = self.forward_img_head(q_emb, k_emb)
        losses['diff_loss'] = diff * self.commitment_cost
        

        return losses, diff.item()
    
    



@MODELS.register_module()
class VQCL_v9(VQCL_v6):


    def forward_train(self, imgs):         
        
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
    
        if isinstance(x1, tuple):
            q_emb = x1[0]
            k_emb = x2[0]
        else:
            c = x1.shape[1]
            
            if c != self.embed_dim:
                q_emb = self.quantize_conv(x1)
                k_emb = self.quantize_conv(x2)
            else:
                q_emb = x1
                k_emb = x2
                
        quant, diff, ind, embed = self.quantize(q_emb.permute(0, 2, 3, 1).contiguous())
        
        losses = dict()

        loss_img_head = self.forward_img_head(q_emb, k_emb, clip_len) if not isinstance(x1, tuple) else self.forward_img_head(x1[-1], x2[-1], clip_len)
        losses.update(add_prefix(loss_img_head, prefix='img_head'))
        
        losses['diff'] = diff * self.commitment_cost
        
        return losses, diff.item()
    
    

@MODELS.register_module()
class VQCL_v10(VQCL_v5):
    """Only for cluster
    """
    def forward_train(self, imgs):
        
        im_q = imgs[:, 0, 0]
        im_k = imgs[:, 0, 1]

        with torch.no_grad():
            self.backbone.eval()
            x1 = self.backbone(im_q)
            x2 = self.backbone(im_k)
            
            if isinstance(x1, tuple):
                q_emb = x1[0]
            else:
                q_emb = x1
        
        quant, diff, ind, embed = self.quantize(q_emb.permute(0, 2, 3, 1).contiguous())
    
        losses = {}

        return losses, diff.item()
    
    def train_step(self, data_batch, optimizer, progress_ratio):
        # parser loss
        losses, diff = self(**data_batch, test_mode=False)
        # loss, log_vars = self.parse_losses(losses)

        # optimizer
        # optimizer.zero_grad()
        
        # loss.backward()

        # optimizer.step()
        
        log_vars = {}

        log_vars['diff_item'] = diff
        outputs = dict(
            log_vars=log_vars,
            num_samples=len(data_batch['imgs'])
        )

        return outputs