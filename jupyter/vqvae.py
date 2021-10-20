import torch
from torch import nn
from torch.nn import functional as F


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

    def forward(self, z):
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
        diff = self.commitment_cost * (quantize.detach() - z).pow(2).mean()

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
        self.quantize_conv = nn.Conv2d(channel, embed_dim, 1)  # Dimension reduction to embedding size
        self.quantize = Quantize(embed_dim, n_embed, commitment_cost, decay)  # Vector quantization
        self.dec = Decoder(embed_dim, in_channel, channel, n_res_block, n_res_channel, downsample)

    def forward(self, x):
        _, quant, diff, _, _ = self.encode(x)
        dec = self.decode(quant)

        return dec, diff

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


if __name__ == '__main__':
    model = VQVAE(downsample=4, n_embed=2048)