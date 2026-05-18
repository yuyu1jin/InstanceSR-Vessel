import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualDenseBlock(nn.Module):
    """
    residual dense block (5 convs)
    input / output: [n, num_feat, h, w]
    internal channels grow progressively: num_feat + k * growth
    """
    def __init__(self, num_feat=64, growth=16, res_scale=0.2):
        super().__init__()
        self.res_scale = res_scale

        self.conv1 = nn.Conv2d(num_feat + 0 * growth, growth, 3, 1, 1, bias=True)
        self.conv2 = nn.Conv2d(num_feat + 1 * growth, growth, 3, 1, 1, bias=True)
        self.conv3 = nn.Conv2d(num_feat + 2 * growth, growth, 3, 1, 1, bias=True)
        self.conv4 = nn.Conv2d(num_feat + 3 * growth, growth, 3, 1, 1, bias=True)
        self.conv5 = nn.Conv2d(num_feat + 4 * growth, num_feat, 3, 1, 1, bias=True)

        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        self._init_weights()

    def _init_weights(self):
        # kaiming initialization + scaling, keep residual relatively small at initialization
        for m in [self.conv1, self.conv2, self.conv3, self.conv4, self.conv5]:
            nn.init.kaiming_normal_(m.weight, a=0.2, mode="fan_in", nonlinearity="leaky_relu")
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        # key: scale down the weights of the last layer, reduce the energy of the residual branch
        self.conv5.weight.data *= 0.1

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat([x, x1], dim=1)))
        x3 = self.lrelu(self.conv3(torch.cat([x, x1, x2], dim=1)))
        x4 = self.lrelu(self.conv4(torch.cat([x, x1, x2, x3], dim=1)))
        x5 = self.conv5(torch.cat([x, x1, x2, x3, x4], dim=1))
        # rrdb res
        return x + self.res_scale * x5


class RRDB(nn.Module):
    """
    Residual in Residual Dense Block
    """
    def __init__(self, num_feat=64, growth=16, res_scale=0.2):
        super().__init__()
        self.rdb1 = ResidualDenseBlock(num_feat, growth, res_scale)
        self.rdb2 = ResidualDenseBlock(num_feat, growth, res_scale)
        self.rdb3 = ResidualDenseBlock(num_feat, growth, res_scale)
        self.res_scale = res_scale

    def forward(self, x):
        out = self.rdb1(x)
        out = self.rdb2(out)
        out = self.rdb3(out)
        return x + self.res_scale * out


class UpsampleBlock(nn.Module):

    def __init__(self, num_feat=64, out_feat=64, scale_factor=2):
        super().__init__()
        self.scale_factor = scale_factor

        self.conv1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1, bias=True)
        self.conv2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1, bias=True)

        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self._init_weights()

    def _init_weights(self):
        for m in [self.conv1, self.conv2]:
            nn.init.kaiming_normal_(m.weight, a=0.2, mode="fan_in", nonlinearity="leaky_relu")
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=self.scale_factor, mode="nearest")
        x = self.lrelu(self.conv1(x))
        x = self.conv2(x)
        return x


class Generator(nn.Module):
    """
    esrgan-style generator:
        - input / output: same range as data ([0,1]), automatically aligned through training
        - upsampling factor controlled by scale_factor (default 4x)
    """
    def __init__(
        self,
        in_channels=3,
        out_channels=3,
        num_feat=64,
        num_grow_channels=16,
        num_basic_block=4,
        scale_factor=2,
    ):
        super().__init__()

        self.conv_first = nn.Conv2d(in_channels, num_feat, 3, 1, 1, bias=True)

        # RRDB trunk
        rrdb_blocks = []
        for _ in range(num_basic_block):
            rrdb_blocks.append(RRDB(num_feat=num_feat, growth=num_grow_channels, res_scale=0.2))
        self.rrdb_trunk = nn.Sequential(*rrdb_blocks)

        # trunk conv
        self.trunk_conv = nn.Conv2d(num_feat, num_feat, 3, 1, 1, bias=True)

        # upsampling module: total factor 4
        self.upsample1 = UpsampleBlock(num_feat, num_feat, scale_factor=2)
        self.upsample2 = UpsampleBlock(num_feat, num_feat, scale_factor=2)
        self.upsample = nn.Sequential(self.upsample1, self.upsample2)

        # output head
        self.hr_conv = nn.Conv2d(num_feat, num_feat, 3, 1, 1, bias=True)
        self.conv_last = nn.Conv2d(num_feat, out_channels, 3, 1, 1, bias=True)

        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        self._init_weights()

    def _init_weights(self):
        # first/trunk/output conv 
        for m in [self.conv_first, self.trunk_conv, self.hr_conv, self.conv_last]:
            nn.init.kaiming_normal_(m.weight, a=0.2, mode="fan_in", nonlinearity="leaky_relu")
            if m.bias is not None:
                nn.init.zeros_(m.bias)

        # can appropriately scale the weights of trunk_conv and hr_conv to further reduce initial output variance
        self.trunk_conv.weight.data *= 0.8
        self.hr_conv.weight.data *= 0.8
        self.conv_last.weight.data *= 0.8

    def forward(self, x):
        # x expected to be in [0,1]
        feat = self.conv_first(x)

        trunk = self.rrdb_trunk(feat)
        trunk = self.trunk_conv(trunk)

        # long skip
        feat = feat + trunk

        # upsample to HR
        out = self.upsample(feat)

        out = self.lrelu(self.hr_conv(out))
        out = self.conv_last(out)

        out = torch.sigmoid(out)  # constrain output to [0,1], accelerate training convergence
        return out
