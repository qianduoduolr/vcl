import torch
import torch.nn as nn

from mmcv.runner import auto_fp16, load_checkpoint
from ...utils import get_root_logger
from ..common import change_stride
from ..registry import BACKBONES
from .swin_transformer import SwinTransformer
from ..builder import build_backbone, build_loss, build_components

@BACKBONES.register_module()
class Vq_Swin(nn.Module):
    def __init__(self,
                 transformer_blocks,
                 vqvae,
                 pretrained_vq):
        super(Vq_Swin, self).__init__()

        self.transformer_blocks = build_backbone(transformer_blocks)
        if vqvae.type != 'DALLE_Encoder':
            self.vqvae = build_components(vqvae).cuda()
            _ = load_checkpoint(self.vqvae, pretrained_vq, map_location='cpu')
            self.vq_enc = self.vqvae.encode
        else:
            raise NotImplementedError

        self.vq_type = vqvae.type
        self.init_weights()

    def init_weights(self):
        for name, param in self.vqvae.named_parameters():
            param.requires_grad = False

    def forward(self, x):

        # main forward
        with torch.no_grad():
            if self.vq_type == 'VQVAE':
                self.vqvae.eval()
                quant_x = self.vq_enc(x)[1]

            else:
                self.vq_enc.eval()
                raise NotImplementedError
        
        out = self.transformer_blocks(quant_x, skip_pe=True)

        return out