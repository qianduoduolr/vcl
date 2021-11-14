import torch
import torch.nn as nn

from mmcv.runner import auto_fp16, load_checkpoint
from ...utils import get_root_logger
from ..common import change_stride
from .resnet import *
from ..registry import BACKBONES
from .swin_transformer import SwinTransformer
from ..builder import build_backbone, build_loss, build_components



@BACKBONES.register_module()
class ResNet_NoStem(nn.Module):
    """ResNet backbone.

    Args:
        depth (int): Depth of resnet, from {18, 34, 50, 101, 152}.
        pretrained (str | None): Name of pretrained model. Default: None.
        in_channels (int): Channel num of input features. Default: 3.
        num_stages (int): Resnet stages. Default: 4.
        strides (Sequence[int]): Strides of the first block of each stage.
        dilations (Sequence[int]): Dilation of each stage.
        style (str): ``pytorch`` or ``caffe``. If set to "pytorch", the
            stride-two layer is the 3x3 conv layer, otherwise the stride-two
            layer is the first 1x1 conv layer. Default: ``pytorch``.
        frozen_stages (int): Stages to be frozen (all param fixed). -1 means
            not freezing any parameters. Default: -1.
        conv_cfg (dict): Config for norm layers. Default: dict(type='Conv').
        norm_cfg (dict):
            Config for norm layers. required keys are `type` and
            `requires_grad`. Default: dict(type='BN2d', requires_grad=True).
        act_cfg (dict): Config for activate layers.
            Default: dict(type='ReLU', inplace=True).
        norm_eval (bool): Whether to set BN layers to eval mode, namely, freeze
            running stats (mean and var). Default: False.
        partial_bn (bool): Whether to use partial bn. Default: False.
        with_cp (bool): Use checkpoint or not. Using checkpoint will save some
            memory while slowing down the training speed. Default: False.
    """
    arch_settings = {
        10: (BasicBlock, (2, 2))
    }

    def __init__(self,
                 depth,
                 inplanes,
                 pretrained=None,
                 torchvision_pretrain=True,
                 num_stages=2,
                 strides=(1, 1),
                 dilations=(1, 1),
                 out_indices=(0, ),
                 style='pytorch',
                 frozen_stages=-1,
                 conv_cfg=dict(type='Conv'),
                 norm_cfg=dict(type='BN2d', requires_grad=True),
                 act_cfg=dict(type='ReLU', inplace=True),
                 norm_eval=False,
                 partial_bn=False,
                 with_cp=False,
                 zero_init_residual=True):
        super().__init__()

        self.depth = depth
        self.pretrained = pretrained
        self.torchvision_pretrain = torchvision_pretrain
        self.num_stages = num_stages
        assert num_stages >= 1 and num_stages <= 4
        self.strides = strides
        self.dilations = dilations
        assert len(strides) == len(dilations) == num_stages
        self.out_indices = out_indices
        self.original_out_indices = out_indices
        assert max(out_indices) < num_stages
        self.style = style
        self.frozen_stages = frozen_stages
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        self.norm_eval = norm_eval
        self.partial_bn = partial_bn
        self.with_cp = with_cp
        self.zero_init_residual = zero_init_residual

        self.block, stage_blocks = self.arch_settings[depth]
        self.stage_blocks = stage_blocks[:num_stages]
        self.inplanes = inplanes


        self.res_layers = []
        for i, num_blocks in enumerate(self.stage_blocks):
            stride = strides[i]
            dilation = dilations[i]
            planes = inplanes * 2**(i+1)
            res_layer = make_res_layer(
                self.block,
                self.inplanes,
                planes,
                num_blocks,
                stride=stride,
                dilation=dilation,
                style=self.style,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg,
                with_cp=with_cp)

            self.inplanes = planes * self.block.expansion
            layer_name = f'layer{i + 1}'
            self.add_module(layer_name, res_layer)
            self.res_layers.append(layer_name)

        self._freeze_stages()

        self.feat_dim = self.block.expansion * inplanes * 2**(
            len(self.stage_blocks))


    def _load_conv_params(self, conv, state_dict_tv, module_name_tv,
                          loaded_param_names):
        """Load the conv parameters of resnet from torchvision.

        Args:
            conv (nn.Module): The destination conv module.
            state_dict_tv (OrderedDict): The state dict of pretrained
                torchvision model.
            module_name_tv (str): The name of corresponding conv module in the
                torchvision model.
            loaded_param_names (list[str]): List of parameters that have been
                loaded.
        """

        weight_tv_name = module_name_tv + '.weight'
        conv.weight.data.copy_(state_dict_tv[weight_tv_name])
        loaded_param_names.append(weight_tv_name)

        if getattr(conv, 'bias') is not None:
            bias_tv_name = module_name_tv + '.bias'
            conv.bias.data.copy_(state_dict_tv[bias_tv_name])
            loaded_param_names.append(bias_tv_name)

    def _load_bn_params(self, bn, state_dict_tv, module_name_tv,
                        loaded_param_names):
        """Load the bn parameters of resnet from torchvision.

        Args:
            bn (nn.Module): The destination bn module.
            state_dict_tv (OrderedDict): The state dict of pretrained
                torchvision model.
            module_name_tv (str): The name of corresponding bn module in the
                torchvision model.
            loaded_param_names (list[str]): List of parameters that have been
                loaded.
        """

        for param_name, param in bn.named_parameters():
            param_tv_name = f'{module_name_tv}.{param_name}'
            param_tv = state_dict_tv[param_tv_name]
            param.data.copy_(param_tv)
            loaded_param_names.append(param_tv_name)

        for param_name, param in bn.named_buffers():
            param_tv_name = f'{module_name_tv}.{param_name}'
            # some buffers like num_batches_tracked may not exist
            if param_tv_name in state_dict_tv:
                param_tv = state_dict_tv[param_tv_name]
                param.data.copy_(param_tv)
                loaded_param_names.append(param_tv_name)

    def _load_torchvision_checkpoint(self,
                                     pretrained,
                                     strict=False,
                                     logger=None):
        """Initiate the parameters from torchvision pretrained checkpoint."""
        state_dict_torchvision = _load_checkpoint(self.pretrained)
        if 'state_dict' in state_dict_torchvision:
            state_dict_torchvision = state_dict_torchvision['state_dict']

        loaded_param_names = []
        for name, module in self.named_modules():
            if isinstance(module, ConvModule):
                # we use a ConvModule to wrap conv+bn+relu layers, thus the
                # name mapping is needed
                if 'downsample' in name:
                    # layer{X}.{Y}.downsample.conv->layer{X}.{Y}.downsample.0
                    original_conv_name = name + '.0'
                    # layer{X}.{Y}.downsample.bn->layer{X}.{Y}.downsample.1
                    original_bn_name = name + '.1'
                else:
                    # layer{X}.{Y}.conv{n}.conv->layer{X}.{Y}.conv{n}
                    original_conv_name = name
                    # layer{X}.{Y}.conv{n}.bn->layer{X}.{Y}.bn{n}
                    original_bn_name = name.replace('conv', 'bn')
                self._load_conv_params(module.conv, state_dict_torchvision,
                                       original_conv_name, loaded_param_names)
                self._load_bn_params(module.bn, state_dict_torchvision,
                                     original_bn_name, loaded_param_names)

        # check if any parameters in the 2d checkpoint are not loaded
        remaining_names = set(
            state_dict_torchvision.keys()) - set(loaded_param_names)
        if remaining_names:
            logger.info(
                f'These parameters in pretrained checkpoint are not loaded'
                f': {remaining_names}')

    def init_weights(self):
        """Initiate the parameters either from existing checkpoint or from
        scratch."""
        if isinstance(self.pretrained, str):
            logger = get_root_logger()
            if self.torchvision_pretrain:
                # torchvision's
                logger.info(f'Loading {self.pretrained} as torchvision')
                self._load_torchvision_checkpoint(
                    self.pretrained, strict=False, logger=logger)
            else:
                # ours
                logger.info(f'Loading {self.pretrained} not as torchvision')
                load_checkpoint(
                    self, self.pretrained, strict=False, logger=logger)
        elif self.pretrained is None:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    kaiming_init(m)
                elif isinstance(m, nn.BatchNorm2d):
                    constant_init(m, 1)
            if self.zero_init_residual:
                for m in self.modules():
                    if isinstance(m, Bottleneck):
                        constant_init(m.conv3.norm, 0)
                    elif isinstance(m, BasicBlock):
                        constant_init(m.conv2.norm, 0)
        else:
            raise TypeError('pretrained must be a str or None')

    def forward(self, x):
        """Defines the computation performed at every call.

        Args:
            x (torch.Tensor): The input data.

        Returns:
            torch.Tensor: The feature of the input samples extracted
            by the backbone.
        """

        outs = []
        for i, layer_name in enumerate(self.res_layers):
            res_layer = getattr(self, layer_name)
            x = res_layer(x)
            if i in self.out_indices:
                outs.append(x)
        if len(outs) == 1:
            return outs[0]
        return tuple(outs)

    def forward_block(self, x, index):
        x = self.conv1(x)
        x = self.maxpool(x)
        block_idx = 0
        for i, layer_name in enumerate(self.res_layers):
            res_layer = getattr(self, layer_name)
            for block in res_layer:
                x = block(x)
                if index == block_idx:
                    return x
                block_idx += 1

    @property
    def output_stride(self):
        return np.prod(self.strides[:self.num_stages]) * 4

    def _freeze_stages(self):
        """Prevent all the parameters from being optimized before
        ``self.frozen_stages``."""
        if self.frozen_stages >= 0:
            # self.conv1.bn.eval()
            # for m in self.conv1.modules():
            #     for param in m.parameters():
            #         param.requires_grad = False
            self.conv1.eval()
            for param in self.conv1.parameters():
                param.requires_grad = False

        for i in range(1, self.frozen_stages + 1):
            m = getattr(self, f'layer{i}')
            m.eval()
            for param in m.parameters():
                param.requires_grad = False

    def _partial_bn(self):
        logger = get_root_logger()
        logger.info('Freezing BatchNorm2D except the first one.')
        count_bn = 0
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                count_bn += 1
                if count_bn >= 2:
                    m.eval()
                    # shutdown update in frozen mode
                    m.weight.requires_grad = False
                    m.bias.requires_grad = False

    def switch_strides(self, strides=None):
        for i, layer_name in enumerate(self.res_layers):
            for m in getattr(self, layer_name).modules():
                if (isinstance(m, (BasicBlock, Bottleneck))
                        and m.downsample is not None):
                    if strides is None:
                        stride = self.strides[i]
                    else:
                        stride = strides[i]
                    m.downsample.apply(partial(change_stride, stride=stride))
                    if self.depth in [18, 34] or not self.style == 'pytorch':
                        m.conv1.apply(partial(change_stride, stride=stride))
                    else:
                        m.conv2.apply(partial(change_stride, stride=stride))

    def switch_out_indices(self, out_indices=None):
        if out_indices is None:
            self.out_indices = self.original_out_indices
        else:
            self.out_indices = out_indices

    def train(self, mode=True):
        """Set the optimization status when training."""
        super().train(mode)
        self._freeze_stages()
        if mode and self.norm_eval:
            for m in self.modules():
                if isinstance(m, _BatchNorm):
                    m.eval()
        if mode and self.partial_bn:
            self._partial_bn()


@BACKBONES.register_module()
class Vq_Res(nn.Module):
    def __init__(self,
                 res_blocks,
                 vqvae,
                 pretrained_vq):
        super(Vq_Res, self).__init__()

        self.res_blocks = ResNet_NoStem(**res_blocks)
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
        
        out = self.res_blocks(quant_x)

        return out