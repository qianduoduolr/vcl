from ..registry import COMPONENTS
from ..builder import build_components
import torch
import torch.nn as nn

@COMPONENTS.register_module()
class MlpHead(nn.Module):
    pass


@COMPONENTS.register_module()
class DclHead(nn.Module):
    def __init__(self,
                 in_channels,
                 hid_channels,
                 out_channels,
                 num_grid=None):
        super(DenseCLNeck, self).__init__()

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, hid_channels), nn.ReLU(inplace=True),
            nn.Linear(hid_channels, out_channels))

        self.with_pool = num_grid != None
        if self.with_pool:
            self.pool = nn.AdaptiveAvgPool2d((num_grid, num_grid))
        self.mlp2 = nn.Sequential(
            nn.Conv2d(in_channels, hid_channels, 1), nn.ReLU(inplace=True),
            nn.Conv2d(hid_channels, out_channels, 1))
        self.avgpool2 = nn.AdaptiveAvgPool2d((1, 1))

    def init_weights(self, init_linear='normal'):
        _init_weights(self, init_linear)

    def forward(self, x):
        assert len(x) == 1
        x = x[0]

        avgpooled_x = self.avgpool(x)
        avgpooled_x = self.mlp(avgpooled_x.view(avgpooled_x.size(0), -1))

        if self.with_pool:
            x = self.pool(x) # sxs
        x = self.mlp2(x) # sxs: bxdxsxs
        avgpooled_x2 = self.avgpool2(x) # 1x1: bxdx1x1
        x = x.view(x.size(0), x.size(1), -1) # bxdxs^2
        avgpooled_x2 = avgpooled_x2.view(avgpooled_x2.size(0), -1) # bxd
        return [avgpooled_x, x, avgpooled_x2]