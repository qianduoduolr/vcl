import mmcv
from torch import distributed

from vcl.models.losses.losses import Soft_Ce_Loss

from ..components import *
from vcl.utils.helpers import *
from vcl.utils import *


import torch.nn as nn
import torch
import torch.nn.functional as F



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

    def forward(self, z, cluster=False, soft_align=False):
        """
        :param z: Encoder output
        :return: Quantized tensor
        """

        flatten = z.reshape(-1, self.dim)  # Converting the z input to a [N x D] tensor, where D is embedding dimension
        
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
        if self.training or cluster:
            n_total = embed_onehot.sum(0)
            embed_sum = flatten.transpose(0, 1) @ embed_onehot
            
            if distributed.is_initialized():
                distributed.all_reduce(n_total)
                distributed.all_reduce(embed_sum)
            
            self.cluster_size.data.mul_(self.decay).add_(
                n_total, alpha=1 - self.decay
            )
            
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
        
        if soft_align:
            out = F.normalize(z, dim=-1)
            vq_emb = F.normalize(self.embed, dim=0)
            embed_ind_soft = torch.mm(flatten, vq_emb)
            embed_ind_soft = embed_ind_soft.view(*z.shape[:-1], self.n_embed)  # B x H x W tensor with the indices of their nearest code
            return quantize, diff, embed_ind, self.embed, embed_ind_soft
        else:
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