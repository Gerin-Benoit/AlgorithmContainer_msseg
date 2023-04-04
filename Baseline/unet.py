import torch
import torch.nn as nn
from torch.nn import Parameter
import torch.distributions as D
import numpy as np
from spectral_norm_conv_inplace import *
from spectral_norm_fc import *
from monai.networks import nets as nets
from monai.utils import ensure_tuple_rep
from swin_attention import SwinL2AttentionBasicLayer


# from sklearn.mixture import GaussianMixture

class ActNormLP2D(nn.Module):
    def __init__(self, num_channels):
        super(ActNormLP2D, self).__init__()
        self.num_channels = num_channels
        self._log_scale = Parameter(torch.zeros(num_channels))  # Tensor not zeros
        self._shift = Parameter(torch.zeros(num_channels))
        self._init = False
        self.eps = 1e-6

    def log_scale(self):
        return self._log_scale[None, :, None, None]

    def shift(self):
        return self._shift[None, :, None, None]

    def forward(self, x):
        if not self._init:
            with torch.no_grad():
                # initialize params to input stats
                assert self.num_channels == x.size(1)

                mean = torch.transpose(x, 0, 1).contiguous().view(self.num_channels, -1).mean(dim=1, keepdim=False)
                zero_mean = x - mean[None, :, None, None]
                var = torch.transpose(zero_mean ** 2, 0, 1).contiguous().view(self.num_channels, -1).mean(dim=1)
                std = (var + self.eps) ** .5
                log_scale = torch.log(1. / std)
                self._log_scale.data = torch.clamp(log_scale, None, 0.0)
                self._shift.data = - mean * torch.exp(log_scale)
                self._init = True

        log_scale = self.log_scale()
        return x * torch.exp(log_scale) + self.shift()


class ActNormLP3D(nn.Module):
    def __init__(self, num_channels, affine=True):
        super(ActNormLP3D, self).__init__()
        self.num_channels = num_channels
        self._log_scale = Parameter(torch.Tensor(num_channels))
        self._shift = Parameter(torch.Tensor(num_channels))
        self._init = False
        self.eps = 1e-6

    def log_scale(self):
        return self._log_scale[None, :, None, None, None]

    def shift(self):
        return self._shift[None, :, None, None, None]

    def forward(self, x):
        if not self._init:
            with torch.no_grad():
                # initialize params to input stats
                assert self.num_channels == x.size(1)
                mean = torch.transpose(x, 0, 1).contiguous().view(self.num_channels, -1).mean(dim=1, keepdim=False)
                zero_mean = x - mean[None, :, None, None, None]
                var = torch.transpose(zero_mean ** 2, 0, 1).contiguous().view(self.num_channels, -1).mean(dim=1)
                std = (var + self.eps) ** .5
                log_scale = torch.log(1. / std)
                self._log_scale.data = torch.clamp(log_scale, None, 0.0)
                self._shift.data = - mean * torch.exp(log_scale)
                self._init = True

        log_scale = self.log_scale()
        return x * torch.exp(log_scale) + self.shift()


class basic_block_2D(nn.Module):
    def __init__(self, norm_layer, in_shape, activation, bottleneck_reduction=2, repetition=2, bias=True, c=1):
        super().__init__()
        C, H, W = in_shape
        self.Residual = nn.Sequential(
            wrapper_spectral_norm(nn.Conv2d(C, C, kernel_size=3, padding=1, stride=1, bias=bias),
                                  (C, H, W), 3, c=c),
            norm_layer(C),
            activation(inplace=True),
            wrapper_spectral_norm(nn.Conv2d(C, C, kernel_size=3, padding=1, stride=1, bias=bias),
                                  (C, H, W), 3, c=c),
            norm_layer(C),
        )
        self.out_activation = activation(inplace=True)

    def forward(self, x):
        identity = x
        residual = self.Residual(x)
        return self.out_activation(identity + residual)


class bottleneck_2D(nn.Module):
    def __init__(self, norm_layer, channels, in_shape, activation, bottleneck_reduction=2, bias=True, c=1):
        super().__init__()
        C, H, W = in_shape
        self.Residual = nn.Sequential(
            wrapper_spectral_norm(
                nn.Conv2d(channels, channels // bottleneck_reduction, kernel_size=1, padding=0, stride=1, bias=bias),
                (C, H, W), 1, c=c),
            norm_layer(channels // bottleneck_reduction),
            activation(inplace=True),
            wrapper_spectral_norm(
                nn.Conv2d(channels // bottleneck_reduction, channels // bottleneck_reduction, kernel_size=3, padding=1,
                          stride=1, bias=bias),
                (C // bottleneck_reduction, H, W), 3, c=c),
            norm_layer(channels // bottleneck_reduction),
            activation(inplace=True),
            wrapper_spectral_norm(
                nn.Conv2d(channels // bottleneck_reduction, channels, kernel_size=1, padding=0,
                          stride=1, bias=bias),
                (C // bottleneck_reduction, H, W), 1, c=c),
            norm_layer(channels)

        )
        self.out_activation = activation(inplace=True)

    def forward(self, x):
        identity = x
        residual = self.Residual(x)
        return self.out_activation(identity + residual)


class bottleneck_block_2D(nn.Module):

    def __init__(self, norm_layer, in_shape, activation, bottleneck_reduction=2, repetition=2, bias=True, c=1):
        super().__init__()
        C, H, W = in_shape
        self.n_channels = C
        self.repetition = repetition
        module_list = [
            bottleneck_2D(norm_layer=norm_layer, channels=C, in_shape=in_shape, activation=activation,
                          bottleneck_reduction=bottleneck_reduction, bias=bias, c=c) for
            i in range(repetition)]
        self.multi_residual = nn.Sequential(*module_list)

    def forward(self, x):
        return self.multi_residual(x)


class basic_block_3D(nn.Module):
    def __init__(self, norm_layer, in_shape, activation, bottleneck_reduction=2, repetition=2, bias=True, c=1,
                 dropout=0):
        super().__init__()
        C, H, W, Z = in_shape
        if dropout != 0:
            self.Residual = nn.Sequential(
                wrapper_spectral_norm(nn.Conv3d(C, C, kernel_size=3, padding=1, stride=1, bias=bias),
                                      (C, H, W, Z), 3, c=c),
                norm_layer(C, affine=True) if not norm_layer == nn.LayerNorm else nn.LayerNorm((C, H, W, Z)),
                nn.Dropout3d(p=dropout, inplace=True),
                activation(inplace=True),
                wrapper_spectral_norm(nn.Conv3d(C, C, kernel_size=3, padding=1, stride=1, bias=bias),
                                      (C, H, W, Z), 3, c=c),
                norm_layer(C, affine=True) if not norm_layer == nn.LayerNorm else nn.LayerNorm((C, H, W, Z)),
                nn.Dropout3d(p=dropout, inplace=True),
            )
        else:
            self.Residual = nn.Sequential(
                wrapper_spectral_norm(nn.Conv3d(C, C, kernel_size=3, padding=1, stride=1, bias=bias),
                                      (C, H, W, Z), 3, c=c),
                norm_layer(C, affine=True) if not norm_layer == nn.LayerNorm else nn.LayerNorm((C, H, W, Z)),
                activation(inplace=True),
                wrapper_spectral_norm(nn.Conv3d(C, C, kernel_size=3, padding=1, stride=1, bias=bias),
                                      (C, H, W, Z), 3, c=c),
                norm_layer(C, affine=True) if not norm_layer == nn.LayerNorm else nn.LayerNorm((C, H, W, Z)),
            )
        self.out_activation = activation(inplace=True)

    def forward(self, x):
        identity = x
        residual = self.Residual(x)
        return self.out_activation(identity + residual)


class bottleneck_3D(nn.Module):
    def __init__(self, norm_layer, channels, in_shape, activation, bottleneck_reduction=2, bias=True, c=1, dropout=0):
        super().__init__()
        C, H, W, Z = in_shape
        if dropout != 0:
            self.Residual = nn.Sequential(
                wrapper_spectral_norm(
                    nn.Conv3d(channels, channels // bottleneck_reduction, kernel_size=1, padding=0, stride=1,
                              bias=bias),
                    (C, H, W, Z), 1, c=c),
                norm_layer(channels // bottleneck_reduction,
                           affine=True) if not norm_layer == nn.LayerNorm else nn.LayerNorm(
                    (channels // bottleneck_reduction, H, W, Z)),
                nn.Dropout3d(p=dropout, inplace=True),
                activation(inplace=True),
                wrapper_spectral_norm(
                    nn.Conv3d(channels // bottleneck_reduction, channels // bottleneck_reduction, kernel_size=3,
                              padding=1,
                              stride=1, bias=bias),
                    (C // bottleneck_reduction, H, W, Z), 3, c=c),
                norm_layer(channels // bottleneck_reduction,
                           affine=True) if not norm_layer == nn.LayerNorm else nn.LayerNorm(
                    (channels // bottleneck_reduction, H, W, Z)),
                nn.Dropout3d(p=dropout, inplace=True),
                activation(inplace=True),
                wrapper_spectral_norm(
                    nn.Conv3d(channels // bottleneck_reduction, channels, kernel_size=1, padding=0,
                              stride=1, bias=bias),
                    (C // bottleneck_reduction, H, W, Z), 1, c=c),
                norm_layer(channels, affine=True) if not norm_layer == nn.LayerNorm else nn.LayerNorm(
                    (channels, H, W, Z)),
                nn.Dropout3d(p=dropout, inplace=True),

            )
        else:
            self.Residual = nn.Sequential(
                wrapper_spectral_norm(
                    nn.Conv3d(channels, channels // bottleneck_reduction, kernel_size=1, padding=0, stride=1,
                              bias=bias),
                    (C, H, W, Z), 1, c=c),
                norm_layer(channels // bottleneck_reduction, affine=True),
                activation(inplace=True),
                wrapper_spectral_norm(
                    nn.Conv3d(channels // bottleneck_reduction, channels // bottleneck_reduction, kernel_size=3,
                              padding=1,
                              stride=1, bias=bias),
                    (C // bottleneck_reduction, H, W, Z), 3, c=c),
                norm_layer(channels // bottleneck_reduction, affine=True),
                activation(inplace=True),
                wrapper_spectral_norm(
                    nn.Conv3d(channels // bottleneck_reduction, channels, kernel_size=1, padding=0,
                              stride=1, bias=bias),
                    (C // bottleneck_reduction, H, W, Z), 1, c=c),
                norm_layer(channels, affine=True),

            )
        self.out_activation = activation(inplace=True)

    def forward(self, x):
        identity = x
        residual = self.Residual(x)
        return self.out_activation(identity + residual)


class bottleneck_block_3D(nn.Module):

    def __init__(self, norm_layer, in_shape, activation, bottleneck_reduction=2, repetition=2, bias=True, c=1,
                 dropout=0):
        super().__init__()
        C, H, W, Z = in_shape
        self.n_channels = C
        self.repetition = repetition
        module_list = [
            bottleneck_3D(norm_layer=norm_layer, channels=C, in_shape=in_shape, activation=activation,
                          bottleneck_reduction=bottleneck_reduction, bias=bias, c=c, dropout=dropout) for
            i in range(repetition)]
        self.multi_residual = nn.Sequential(*module_list)

    def forward(self, x):
        return self.multi_residual(x)


class PytorchUNet2D(nn.Module):
    def __init__(self, in_shape,
                 num_classes: int = 2,
                 c: float = 1,
                 n_channels=1,  # grayscale
                 norm_layer=ActNormLP2D,
                 activation=nn.ELU,
                 block_type=basic_block_2D,
                 bottleneck_reduction=2,
                 repetition=2,
                 depth=4,
                 device='cpu'
                 ):
        super(PytorchUNet2D, self).__init__()
        """Unet"""
        C, H, W = in_shape
        self.num_classes = num_classes
        self.n_channels = n_channels
        self.device = device
        self.channels_list = [16, 32, 64, 128, 256]  # [32, 64, 128, 256, 512] #[20, 40, 80, 160, 320]

        in_channels = n_channels
        out_channels = self.channels_list[0]

        self.inc = nn.Sequential(
            wrapper_spectral_norm(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=True), (C, H, W),
                                  3, c=c),
            norm_layer(out_channels, affine=True),
            activation(inplace=True),
            wrapper_spectral_norm(nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=True),
                                  (out_channels, H, W), 3, c=c),
            norm_layer(out_channels, affine=True),
            activation(inplace=True)
        )
        self.down_list = nn.ModuleList([])
        for i in range(depth):
            self.down_list.append(Down_Smooth_2D(self.channels_list[i], self.channels_list[i + 1], norm_layer,
                                                 in_shape=(self.channels_list[i], H // (2 ** i), W // (2 ** i)),
                                                 activation=activation, block_type=block_type,
                                                 bottleneck_reduction=bottleneck_reduction, repetition=repetition, c=c))

        # Decoder for the segmentation task
        self.bottom = Up_Smooth_Bottom_2D(self.channels_list[depth], self.channels_list[depth - 1], norm_layer,
                                          in_shape=(self.channels_list[depth], H // (2 ** depth), W // (2 ** depth)),
                                          activation=activation, block_type=block_type,
                                          bottleneck_reduction=bottleneck_reduction, repetition=repetition, c=c)
        self.up_list = nn.ModuleList([])
        for i in range(depth - 1, 0, -1):
            self.up_list.append(Up_Smooth_2D(self.channels_list[i] * 2, self.channels_list[i - 1], norm_layer,
                                             in_shape=(
                                                 self.channels_list[i] * 2, H // (2 ** (i - 1)), W // 2 ** (i - 1)),
                                             activation=activation, block_type=block_type,
                                             bottleneck_reduction=bottleneck_reduction, repetition=repetition, c=c))

        self.outc = OutConv_Smooth_2D(self.channels_list[0], num_classes, in_shape=(self.channels_list[0], H, W),
                                      c=None)  # prediction for each voxel
        """Spectral Normalization"""
        self.smoothness = torch.tensor(c).to(self.device) if c is not None else None

    def forward(self, x):
        x = self.inc(x)
        fms = []
        for down in self.down_list:
            x = down(x)
            fms.append(x)

        x = self.bottom(fms[-1])

        for i, up in enumerate(self.up_list):
            x = up(x, fms[-(i + 2)])

        x = self.outc(x)

        return x

    def forward_fm(self, x, level=5):
        x = self.inc(x)
        fms = []
        for down in self.down_list:
            x = down(x)
            fms.append(x)

        x = self.bottom(fms[-1])
        fm_density = [x]

        for i, up in enumerate(self.up_list):
            x = up(x, fms[-(i + 2)])
            fm_density.append(x)

        return fm_density

    def forward_fm_and_seg(self, x):
        x = self.inc(x)
        fms = []
        for down in self.down_list:
            x = down(x)
            fms.append(x)

        fm_density = [x]
        x = self.bottom(fms[-1])
        fm_density.append(x)

        for i, up in enumerate(self.up_list):
            x = up(x, fms[-(i + 2)])
            fm_density.append(x)

        x = self.outc(x)

        return fm_density, x

    def clamp_norm_layers(self):
        if self.smoothness is not None:
            for name, p in self.named_parameters():
                if "_log_scale" in name:
                    p.data.clamp_(None, torch.log(self.smoothness))


class PytorchUNet3D(nn.Module):
    def __init__(self, in_shape,
                 num_classes: int = 2,
                 c: float = 1,
                 n_channels=1,  # grayscale
                 norm_layer=ActNormLP3D,
                 activation=nn.ELU,
                 block_type=basic_block_3D,
                 bottleneck_reduction=2,
                 repetition=2,
                 depth=4,
                 cout=1,
                 device='cpu',
                 ssl=False,
                 dropout=0.0,
                 dropout_style=None,
                 swin=False
                 ):
        super(PytorchUNet3D, self).__init__()
        """Unet"""
        C, H, W, Z = in_shape
        self.num_classes = num_classes
        self.n_channels = n_channels
        self.device = device
        self.channels_list = [12, 24, 48, 96, 192] if swin else [20, 40, 80, 160, 320]
        self.num_heads = [3, 3, 4, 4, 6]
        self.window_size = ensure_tuple_rep(7, 3)
        self.dropout = dropout
        self.dropout_style = dropout_style

        if dropout != 0:
            self.inc = nn.Sequential(
                wrapper_spectral_norm(nn.Conv3d(n_channels, self.channels_list[0], kernel_size=3, padding=1, bias=True),
                                      (C, H, W, Z), 3, c=cout),
                norm_layer(self.channels_list[0], affine=True) if not norm_layer == nn.LayerNorm else nn.LayerNorm(
                    (self.channels_list[0], H, W, Z)),
                nn.Dropout3d(p=self.dropout, inplace=True),
                activation(inplace=True),
                wrapper_spectral_norm(
                    nn.Conv3d(self.channels_list[0], self.channels_list[0], kernel_size=3, padding=1, bias=True),
                    (self.channels_list[0], H, W, Z), 3, c=cout),
                norm_layer(self.channels_list[0], affine=True) if not norm_layer == nn.LayerNorm else nn.LayerNorm(
                    (self.channels_list[0], H, W, Z)),
                nn.Dropout3d(p=self.dropout, inplace=True),
                activation(inplace=True)
            )
        else:
            self.inc = nn.Sequential(
                wrapper_spectral_norm(nn.Conv3d(n_channels, self.channels_list[0], kernel_size=3, padding=1, bias=True),
                                      (C, H, W, Z), 3, c=cout),
                norm_layer(self.channels_list[0], affine=True) if not norm_layer == nn.LayerNorm else nn.LayerNorm(
                    (self.channels_list[0], H, W, Z)),
                activation(inplace=True),
                wrapper_spectral_norm(
                    nn.Conv3d(self.channels_list[0], self.channels_list[0], kernel_size=3, padding=1, bias=True),
                    (self.channels_list[0], H, W, Z), 3, c=cout),
                norm_layer(self.channels_list[0], affine=True) if not norm_layer == nn.LayerNorm else nn.LayerNorm(
                    (self.channels_list[0], H, W, Z)),
                activation(inplace=True)
            )

        self.down_list = nn.ModuleList([])
        for i in range(depth):
            self.down_list.append(Down_Smooth_3D(self.channels_list[i], self.channels_list[i + 1], norm_layer,
                                                 in_shape=(
                                                     self.channels_list[i], H // (2 ** i), W // (2 ** i),
                                                     Z // (2 ** i)),
                                                 activation=activation, block_type=block_type,
                                                 bottleneck_reduction=bottleneck_reduction, repetition=repetition, c=c,
                                                 cout=cout, dropout=self.dropout))

        if ssl:
            # Decoder for the ssl task
            self.bottom = Up_Smooth_Bottom_3D(self.channels_list[depth], self.channels_list[depth - 1], norm_layer,
                                              in_shape=(
                                                  self.channels_list[depth], H // (2 ** depth), W // (2 ** depth),
                                                  Z // (2 ** depth)),
                                              activation=activation, block_type=block_type,
                                              bottleneck_reduction=bottleneck_reduction, repetition=repetition, c=c,
                                              cout=cout, dropout=self.dropout)

            self.up_list = nn.ModuleList([])
            for i in range(depth - 1, 0, -1):
                self.up_list.append(Up_Layer_3D(self.channels_list[i] * 2, self.channels_list[i - 1], norm_layer,
                                                in_shape=(
                                                    self.channels_list[i] * 2, H // (2 ** (i - 1)), W // 2 ** (i - 1),
                                                    Z // 2 ** (i - 1)),
                                                activation=activation, block_type=block_type,
                                                bottleneck_reduction=bottleneck_reduction, repetition=repetition,
                                                c=None, dropout=self.dropout))

            self.outc = OutConv_Smooth_3D(self.channels_list[0], n_channels, in_shape=(self.channels_list[0], H, W, Z),
                                          c=None)  # prediction for each voxel


        else:
            # Decoder for the segmentation task
            self.bottom = Up_Smooth_Bottom_3D(self.channels_list[depth], self.channels_list[depth - 1], norm_layer,
                                              in_shape=(
                                                  self.channels_list[depth], H // (2 ** depth), W // (2 ** depth),
                                                  Z // (2 ** depth)),
                                              activation=activation, block_type=block_type,
                                              bottleneck_reduction=bottleneck_reduction, repetition=repetition, c=c,
                                              cout=cout, dropout=self.dropout, swin=swin,
                                              num_heads=self.num_heads[depth], window_size=self.window_size)
            self.up_list = nn.ModuleList([])
            for i in range(depth - 1, 0, -1):
                self.up_list.append(Up_Smooth_3D(self.channels_list[i] * 2, self.channels_list[i - 1], norm_layer,
                                                 in_shape=(
                                                     self.channels_list[i] * 2, H // (2 ** i), W // 2 ** i,
                                                     Z // 2 ** i),
                                                 activation=activation, block_type=block_type,
                                                 bottleneck_reduction=bottleneck_reduction, repetition=repetition, c=c,
                                                 cout=cout, dropout=self.dropout, swin=swin,
                                                 num_heads=self.num_heads[i], window_size=self.window_size))

            self.outc = OutConv_Smooth_3D(self.channels_list[0], num_classes, in_shape=(self.channels_list[0], H, W, Z),
                                          c=None)  # prediction for each voxel
        """Spectral Normalization"""
        self.smoothness = torch.tensor(c).to(self.device) if c is not None else None

    def forward(self, x):
        x = self.inc(x)
        fms = []

        for down in self.down_list:
            x = down(x)
            fms.append(x)

        x = self.bottom(fms[-1])

        for i, up in enumerate(self.up_list):
            x = up(x, fms[-(i + 2)])

        x = self.outc(x)

        return x

    def forward_ssl(self, x):
        x = self.inc(x)

        for down in self.down_list:
            x = down(x)

        x = self.bottom(x)

        for i, up in enumerate(self.up_list):
            x = up(x)

        x = self.outc(x)

        return x

    def forward_fm(self, x, level=5):
        x = self.inc(x)
        fms = []
        for down in self.down_list:
            x = down(x)
            fms.append(x)

        fm_density = [x]
        x = self.bottom(fms[-1])
        fm_density.append(x)

        for i, up in enumerate(self.up_list):
            x = up(x, fms[-(i + 2)])
            fm_density.append(x)

        return fm_density

    def forward_fm_and_seg(self, x):
        x = self.inc(x)
        fms = []
        for down in self.down_list:
            x = down(x)
            fms.append(x)

        fm_density = [x]
        x = self.bottom(fms[-1])
        fm_density.append(x)

        for i, up in enumerate(self.up_list):
            x = up(x, fms[-(i + 2)])
            fm_density.append(x)

        x = self.outc(x)

        return fm_density, x

    def clamp_norm_layers(self):
        if self.smoothness is not None:
            c = self.smoothness if self.smoothness > 0 else -self.smoothness
            for name, p in self.named_parameters():
                if "_log_scale" in name:
                    p.data.clamp_(None, torch.log(c))


class ReshapeLayer2D(nn.Module):
    def __init__(self, block_size=2):
        super(ReshapeLayer2D, self).__init__()
        self.block_size = block_size
        self.block_size_sq = block_size * block_size

    def forward(self, input):
        output = input.permute(0, 2, 3, 1)  # Channel dim last
        (batch_size, d_height, d_width, d_depth) = output.size()
        s_depth = int(d_depth / self.block_size_sq)
        s_width = int(d_width * self.block_size)
        s_height = int(d_height * self.block_size)
        # t_1 = output.contiguous().view(batch_size, d_height, d_width, self.block_size_sq, s_depth)
        t_1 = output.contiguous().view(batch_size, d_height, d_width, s_depth, self.block_size, self.block_size)
        t_1 = t_1.permute(0, 1, 4, 2, 5, 3).contiguous()
        output = t_1.view(batch_size, s_height, s_width, s_depth).contiguous()
        output = output.permute(0, 3, 1, 2)
        return output.contiguous()


class ReshapeLayer3D(nn.Module):
    def __init__(self, block_size=2):
        super(ReshapeLayer3D, self).__init__()
        self.block_size = block_size
        self.block_size_cub = block_size ** 3

    def forward(self, input):
        output = input.permute(0, 2, 3, 4, 1)  # Channel dim last
        (batch_size, d_height, d_width, d_depth, d_channel) = output.size()
        s_channel = int(d_channel / self.block_size_cub)
        s_width = int(d_width * self.block_size)
        s_height = int(d_height * self.block_size)
        s_depth = int(d_depth * self.block_size)
        t_1 = output.contiguous().view(batch_size, d_height, d_width, d_depth, s_channel, self.block_size,
                                       self.block_size, self.block_size)
        t_1 = t_1.permute(0, 1, 5, 2, 6, 3, 7, 4).contiguous()
        output = t_1.view(batch_size, s_height, s_width, s_depth, s_channel).contiguous()
        output = output.permute(0, 4, 1, 2, 3)
        return output.contiguous()


class OutConv_Smooth_2D(nn.Module):
    def __init__(self, in_channels, out_channels, in_shape, c=1):
        super().__init__()
        self.conv = wrapper_spectral_norm(nn.Conv2d(in_channels, out_channels, kernel_size=1), in_shape, 1, c=c)

    def forward(self, x):
        return self.conv(x)


class OutConv_Smooth_3D(nn.Module):
    def __init__(self, in_channels, out_channels, in_shape, c=1):
        super().__init__()

        self.conv = wrapper_spectral_norm(nn.Conv3d(in_channels, out_channels, kernel_size=1), in_shape, 1, c=c)

    def forward(self, x):
        return self.conv(x)


class Down_Smooth_2D(nn.Module):

    def __init__(self, in_channels, out_channels, norm_layer, activation, in_shape, block_type=bottleneck_block_2D,
                 bottleneck_reduction=2, repetition=2, bias=True, c=1):
        super().__init__()
        # 1x1 invertible conv
        # MaxPooling
        # Residual blocks
        C, H, W = in_shape
        self.in_conv = wrapper_spectral_norm(nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0, bias=bias),
                                             (2 * C, H, W), 1, c=c)
        self.pooling = nn.MaxPool2d(kernel_size=2, padding=0, stride=2)  # padding et stride à verifier

        self.residual_blocks = block_type(norm_layer=norm_layer, in_shape=(out_channels, H // 2, W // 2),
                                          activation=activation,
                                          bottleneck_reduction=bottleneck_reduction, repetition=repetition, bias=bias,
                                          c=c)

    def forward(self, x):
        x = self.in_conv(x)
        x = self.pooling(x)
        return self.residual_blocks(x)


class Down_Smooth_3D(nn.Module):

    def __init__(self, in_channels, out_channels, norm_layer, in_shape, activation, block_type=bottleneck_block_3D,
                 bottleneck_reduction=2, repetition=2, bias=True, c=1, cout=1, dropout=0):
        super().__init__()
        # 1x1 invertible conv
        # MaxPooling
        # Residual connection :
        #   Conv + Norm + Act
        #   Conv + Norm + Act
        C, H, W, Z = in_shape
        self.in_conv = wrapper_spectral_norm(nn.Conv3d(in_channels, out_channels, kernel_size=1, padding=0, bias=bias),
                                             (C, H, W, Z), 1, c=cout)
        self.pooling = nn.MaxPool3d(kernel_size=2, padding=0, stride=2)  # padding et stride à verifier

        self.residual_blocks = block_type(norm_layer=norm_layer, in_shape=(out_channels, H // 2, W // 2, Z // 2),
                                          activation=activation,
                                          bottleneck_reduction=bottleneck_reduction, repetition=repetition, bias=bias,
                                          c=c, dropout=dropout)

    def forward(self, x):
        x = self.in_conv(x)
        x = self.pooling(x)
        return self.residual_blocks(x)


class Up_Smooth_2D(nn.Module):

    def __init__(self, in_channels, out_channels, norm_layer, in_shape, activation, block_type=bottleneck_block_2D,
                 bottleneck_reduction=2, repetition=2, bias=True, c=1):
        super().__init__()
        # concatenation
        # Reshaping
        # 1x1 invertible conv
        # Residual connection :
        #   Conv + Norm + Act
        #   Conv + Norm + Act
        C, H, W = in_shape
        self.reshape = ReshapeLayer2D()
        self.in_conv = wrapper_spectral_norm(nn.Conv2d(out_channels, out_channels, kernel_size=1, padding=0, bias=bias),
                                             (C // 4, H * 2, W * 2), 1, c=c)
        self.residual_blocks = block_type(norm_layer=norm_layer, in_shape=(C // 4, H * 2, W * 2), activation=activation,
                                          bottleneck_reduction=bottleneck_reduction, repetition=repetition, bias=bias,
                                          c=c)

    def forward(self, x1, x2):
        x = torch.cat([x2, x1], dim=1)
        x = self.reshape(x)
        x = self.in_conv(x)
        return self.residual_blocks(x)


class Up_Smooth_3D(nn.Module):

    def __init__(self, in_channels, out_channels, norm_layer, in_shape, activation, block_type=bottleneck_block_3D,
                 bottleneck_reduction=2, repetition=2, bias=True, c=1, cout=1, dropout=0, swin=False, num_heads=0,
                 window_size=None):
        super().__init__()
        # concatenation
        # Reshaping
        # 1x1 invertible conv
        # Residual connection :
        #   Conv + Norm + Act
        #   Conv + Norm + Act
        self.swin = swin
        C, H, W, Z = in_shape
        self.reshape = ReshapeLayer3D()
        self.in_conv = wrapper_spectral_norm(
            nn.Conv3d(out_channels // 2, out_channels, kernel_size=1, padding=0, bias=bias),
            (C // 8, H * 2, W * 2, Z * 2), 1, c=cout)
        self.residual_blocks = block_type(norm_layer=norm_layer, in_shape=(C // 4, H * 2, W * 2, Z * 2),
                                          activation=activation,
                                          bottleneck_reduction=bottleneck_reduction, repetition=repetition, bias=bias,
                                          c=c, dropout=dropout)
        if swin:
            self.swin_layer = SwinL2AttentionBasicLayer(
                dim=C // 2,
                depth=2,
                num_heads=num_heads,
                window_size=window_size,
                drop_path=[0, 0],
            )

    def forward(self, x1, x2):
        if self.swin:
            x2 = self.swin_layer(x2)
        x = torch.cat([x2, x1], dim=1)
        x = self.reshape(x)
        x = self.in_conv(x)
        return self.residual_blocks(x)


class Up_Layer_3D(nn.Module):
    def __init__(self, in_channels, out_channels, norm_layer, in_shape, activation, block_type=bottleneck_block_3D,
                 bottleneck_reduction=2, repetition=2, bias=True, c=None, dropout=0):
        super().__init__()
        # Reshaping
        # 1x1 invertible conv
        # Residual connection :
        #   Conv + Norm + Act
        #   Conv + Norm + Act

        C, H, W, Z = in_shape
        self.in_conv = nn.ConvTranspose3d(in_channels // 2, in_channels // 4, kernel_size=2, stride=2, bias=bias)

        self.residual_blocks = block_type(norm_layer=norm_layer, in_shape=(out_channels, H * 2, W * 2, Z * 2),
                                          activation=activation,
                                          bottleneck_reduction=bottleneck_reduction, repetition=repetition, bias=bias,
                                          c=c, dropout=dropout)

    def forward(self, x):
        x = self.in_conv(x)

        return self.residual_blocks(x)


class Up_Smooth_Bottom_2D(nn.Module):

    def __init__(self, in_channels, out_channels, norm_layer, in_shape, activation, block_type=bottleneck_block_2D,
                 bottleneck_reduction=2, repetition=2, bias=True, c=1):
        super().__init__()
        # Reshaping
        # 1x1 invertible conv
        # Residual connection :
        #   Conv + Norm + Act
        #   Conv + Norm + Act
        C, H, W = in_shape
        self.reshape = ReshapeLayer2D()
        self.in_conv = wrapper_spectral_norm(
            nn.Conv2d(out_channels // 2, out_channels, kernel_size=1, padding=0, bias=bias), (C, H, W), 1, c=c)
        self.residual_blocks = block_type(norm_layer=norm_layer, in_shape=(out_channels, H, W), activation=activation,
                                          bottleneck_reduction=bottleneck_reduction, repetition=repetition, bias=bias,
                                          c=c)

    def forward(self, x):
        x = self.reshape(x)
        x = self.in_conv(x)
        return self.residual_blocks(x)


class Up_Smooth_Bottom_3D(nn.Module):

    def __init__(self, in_channels, out_channels, norm_layer, in_shape, activation, block_type=bottleneck_block_3D,
                 bottleneck_reduction=2, repetition=2, bias=True, c=1, cout=1, dropout=0, swin=False, num_heads=None,
                 window_size=None):
        super().__init__()
        # Reshaping
        # 1x1 invertible conv
        # Residual connection :
        #   Conv + Norm + Act
        #   Conv + Norm + Act
        self.swin = swin
        C, H, W, Z = in_shape
        self.reshape = ReshapeLayer3D()
        self.in_conv = wrapper_spectral_norm(
            nn.Conv3d(out_channels // 4, out_channels, kernel_size=1, padding=0, bias=bias),
            (C // 8, H * 2, W * 2, Z * 2), 1, c=cout)
        self.residual_blocks = block_type(norm_layer=norm_layer, in_shape=(out_channels, H * 2, W * 2, Z * 2),
                                          activation=activation,
                                          bottleneck_reduction=bottleneck_reduction, repetition=repetition, bias=bias,
                                          c=c, dropout=dropout)

        if swin:
            self.swin_layer = SwinL2AttentionBasicLayer(
                dim=C,
                depth=2,
                num_heads=num_heads,
                window_size=window_size,
                drop_path=[0, 0],
            )

    def forward(self, x):
        if self.swin:
            x = self.swin_layer(x)
        x = self.reshape(x)
        x = self.in_conv(x)
        return self.residual_blocks(x)


class SpectralSwinUNETR(nets.SwinUNETR):
    def __init__(self, img_size, in_channels, out_channels, c=1, **kwargs):
        super().__init__(img_size, in_channels, out_channels, **kwargs)
        self.c = c
        for name, module in self.named_modules():
            if isinstance(module, nn.Conv3d):
                ks = module.kernel_size
                setattr(self, name, wrapper_spectral_norm(module, kernel_size=ks[0], c=self.c))
            elif isinstance(module, nn.ModuleList) and all(isinstance(sub_module, nn.Linear) for sub_module in module):
                # Replace MLP with classical spectral norm
                new_module = nn.Sequential()
                for sub_name, sub_module in module.named_children():
                    sub_module_spectral_norm = nn.utils.spectral_norm(sub_module)
                    new_module.add_module(sub_name, sub_module_spectral_norm)
                setattr(model, name, new_module)

    def clamp_norm_layers(self):
        pass


def wrapper_spectral_norm(layer, shapes, kernel_size, c=1):
    if c is None:
        return layer
    if c > 0:
        return spectral_norm_fc(layer, c,
                                n_power_iterations=1)

    if c < 0:
        if kernel_size == 1:
            # use spectral norm fc, because bound are tight for 1x convolutions
            return spectral_norm_fc(layer, -c,
                                    n_power_iterations=1)
        else:
            # use spectral norm based on conv, because bound not tight
            return spectral_norm_conv(layer, -c, shapes,
                                      n_power_iterations=1)


def init_weights(m):
    if type(m) == nn.Conv3d:
        nn.init.kaiming_uniform_(m.weight, mode="fan_out", nonlinearity="relu")


if __name__ == '__main__':
    """
    in_shape = (1, 64, 32, 32)
    x = torch.rand((1, 1, 64, 32, 32))
    model = PytorchUNet3D(in_shape=in_shape, c=None,norm_layer = nn.BatchNorm3d)
    model.eval()
    y = model(x)
    print(y.shape)
    exit()

    # print(model.inc[1])

    # INITIALIZATION
    model.apply(init_weights)


    value_init = model.inc[1].log_scale()
    for i in range(2):
        x = torch.rand((1, 1, 32, 32), requires_grad=True)
        target = torch.rand((1, 2, 32, 32)) * 20
        lossfn = torch.nn.MSELoss()

        out = model(x)
        optimizer = torch.optim.SGD(model.parameters(), lr=1e-2, weight_decay=0)
        out = model(x)
        loss = lossfn(out, target)
        loss.backward()
        optimizer.step()
    model.clamp_norm_layers()
    value_out = model.inc[1].log_scale()
    print(value_out)
    """
    """
    in_shape = (1, 64, 32)
    x = torch.rand((1, 1, 64, 32))
    model = PytorchUNet2D(in_shape=in_shape, c=None, norm_layer = nn.BatchNorm2d)
    model.eval()
    y = model(x)
    print(y.shape)
    """

    in_shape = (1, 64, 64, 64)
    x = torch.rand((1, 1, 64, 64, 64))
    model = PytorchUNet3D(in_shape=in_shape, c=None, norm_layer=ActNormLP3D, ssl=True)
    model.eval()
    y = model.forward_ssl(x)
    print(y.shape)
