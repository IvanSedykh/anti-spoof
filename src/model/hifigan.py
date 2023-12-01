from collections import namedtuple
from dataclasses import dataclass
from typing import NamedTuple
from omegaconf import DictConfig
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch import Tensor
from torch.nn import Conv1d, ConvTranspose1d, AvgPool1d, Conv2d
from torch.nn.utils.parametrizations import weight_norm, spectral_norm
from torch.nn.utils.parametrize import remove_parametrizations


LRELU_SLOPE = 0.1


def init_weights(m, mean=0.0, std=0.01):
    classname = m.__class__.__name__
    if "Conv" in classname:
        m.weight.data.normal_(mean, std)


def get_padding(kernel_size, dilation=1):
    return int((kernel_size * dilation - dilation) / 2)


class ResBlock(nn.Module):
    def __init__(self, num_channels: int, kernel_size: int = 3, dilations=(1, 3, 5)):
        super().__init__()

        convs1 = []
        for dilration_rate in dilations:
            convs1.append(
                Conv1d(
                    num_channels,
                    num_channels,
                    kernel_size,
                    padding="same",
                    dilation=dilration_rate,
                )
            )
        self.convs1 = nn.ModuleList([weight_norm(c) for c in convs1])

        convs2 = []
        for _ in dilations:
            convs2.append(
                Conv1d(num_channels, num_channels, kernel_size, padding="same")
            )
        self.convs2 = nn.ModuleList([weight_norm(c) for c in convs2])
        self.apply(init_weights)

    def forward(self, x):
        for c1, c2 in zip(self.convs1, self.convs2):
            xt = F.leaky_relu(x, LRELU_SLOPE)
            xt = c1(xt)
            xt = F.leaky_relu(xt, LRELU_SLOPE)
            xt = c2(xt)
            x = xt + x
        return x

    def remove_weight_norm(self):
        for l in self.convs1:
            remove_parametrizations(l)
        for l in self.convs2:
            remove_parametrizations(l)


@dataclass
class GeneratorConfig:
    resblock_kernel_sizes: list[int]
    # upsample_rates: list[int]
    upsample_initial_channel: int
    upsample_kernel_sizes: list[int]
    resblock_dilation_sizes: list[list[int]]
    n_mels: int = 80


class MRF(nn.Module):
    """Stack of ResBlocks with different dilations, kernels"""

    def __init__(
        self,
        num_channels: int,
        kernel_sizes: list[int],
        dilation_sizes: list[list[int]],
    ):
        super().__init__()
        assert len(kernel_sizes) == len(dilation_sizes)
        self.num_blocks = len(kernel_sizes)
        self.blocks = nn.ModuleList()
        for k, d in zip(kernel_sizes, dilation_sizes):
            self.blocks.append(ResBlock(num_channels, k, d))

    def forward(self, x: Tensor):
        output = None
        for block in self.blocks:
            if output is None:
                output = block(x)
            else:
                output += block(x)
        output /= self.num_blocks
        return output


class Generator(nn.Module):
    def __init__(
        self,
        config: GeneratorConfig,
    ):
        super().__init__()

        self.init_conv = weight_norm(
            Conv1d(config.n_mels, config.upsample_initial_channel, 7, 1, padding=3)
        )

        upsamplings = []
        for i, kernel_size in enumerate(config.upsample_kernel_sizes):
            stride = kernel_size // 2
            in_channels = config.upsample_initial_channel // (2**i)
            # each upsampling block reduce hidden size in half
            upsamplings.append(
                ConvTranspose1d(
                    in_channels,
                    in_channels // 2,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=(kernel_size - stride) // 2,
                )
            )
        self.upsamplings = nn.ModuleList([weight_norm(up) for up in upsamplings])

        self.mrfs = nn.ModuleList()
        for i in range(len(self.upsamplings)):
            num_channels = config.upsample_initial_channel // (2 ** (i + 1))
            self.mrfs.append(
                MRF(
                    num_channels,
                    config.resblock_kernel_sizes,
                    config.resblock_dilation_sizes,
                )
            )

        self.out_conv = weight_norm(Conv1d(num_channels, 1, 7, 1, padding=3))
        self.upsamplings.apply(init_weights)
        self.out_conv.apply(init_weights)

    def forward(self, x):
        # input [BS, n_mels, T]
        x = self.init_conv(x)
        for up, mrf in zip(self.upsamplings, self.mrfs):
            x = F.leaky_relu(x, LRELU_SLOPE)
            x = up(x)
            x = mrf(x)
        x = F.leaky_relu(x)
        x = self.out_conv(x)
        x = torch.tanh(x)

        # output [BS, 1, T']
        return x

    def remove_weight_norm(self):
        print("Removing weight norm...")
        for l in self.upsamplings:
            remove_parametrizations(l)
        for l in self.resblocks:
            l.remove_weight_norm()
        remove_parametrizations(self.init_conv)
        remove_parametrizations(self.out_conv)


class SubDiscriminatorPeriod(nn.Module):
    def __init__(self, period: int, kernel_size: int = 5, stride: int = 3):
        super().__init__()
        self.period = period

        convs = []
        in_channels = 1  # wav has 1 emb dim
        for l in range(4):
            out_channels = 2 ** (5 + l)
            padding = get_padding(5, 1)
            convs.append(
                Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=(kernel_size, 1),
                    stride=(stride, 1),
                    padding=(padding, 1),
                )
            )
            in_channels = out_channels
        convs.append(
            Conv2d(out_channels, 1024, kernel_size=(kernel_size, 1), padding="same")
        )
        self.convs = nn.ModuleList([weight_norm(c) for c in convs])

        self.out_conv = weight_norm(Conv2d(1024, 1, (3, 1), padding="same"))

    def pad_reshape(self, x):
        # [B, 1, T] -> [B, 1, T//p, p]
        # todo: check, may be error

        padding = self.period - x.shape[2] % self.period
        x = F.pad(x, (0, padding), "reflect")
        x = x.view(x.shape[0], 1, x.shape[2] // self.period, self.period)

        return x

    def forward(self, x):
        features = []

        x = self.pad_reshape(x)

        for conv in self.convs:
            x = conv(x)
            x = F.leaky_relu(x, LRELU_SLOPE)
            features.append(x)
        x = self.out_conv(x)
        features.append(x)
        x = torch.flatten(x, 1, -1)

        return x, features


# DiscriminatorOutput = namedtuple('DiscriminatorOutput', ['real_predictions', 'fake_predictions'])
class DiscriminatorOutput(NamedTuple):
    real_predictions: list[Tensor]
    fake_predictions: list[Tensor]
    real_features: list[list[Tensor]]
    fake_features: list[list[Tensor]]


class MultiPeriodDiscriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.discriminators = nn.ModuleList(
            [
                SubDiscriminatorPeriod(2),
                SubDiscriminatorPeriod(3),
                SubDiscriminatorPeriod(5),
                SubDiscriminatorPeriod(7),
                SubDiscriminatorPeriod(11),
            ]
        )

    def process_wav(self, wav):
        preds = []
        features = []
        for subdisc in self.discriminators:
            sub_pred, sub_features = subdisc(wav)
            preds.append(sub_pred)
            features.append(sub_features)
        return preds, features

    def forward(self, real_wav, fake_wav) -> DiscriminatorOutput:
        real_preds, real_features = self.process_wav(real_wav)
        fake_preds, fake_features = self.process_wav(fake_wav)

        return DiscriminatorOutput(real_preds, fake_preds, real_features, fake_features)


class SubDiscriminatorScale(nn.Module):
    def __init__(self, use_spectral_norm=False):
        super().__init__()
        norm_f = weight_norm if use_spectral_norm == False else spectral_norm
        self.convs = nn.ModuleList(
            [
                norm_f(Conv1d(1, 128, 15, 1, padding=7)),
                norm_f(Conv1d(128, 128, 41, 2, groups=4, padding=20)),
                norm_f(Conv1d(128, 256, 41, 2, groups=16, padding=20)),
                norm_f(Conv1d(256, 512, 41, 4, groups=16, padding=20)),
                norm_f(Conv1d(512, 1024, 41, 4, groups=16, padding=20)),
                norm_f(Conv1d(1024, 1024, 41, 1, groups=16, padding=20)),
                norm_f(Conv1d(1024, 1024, 5, 1, padding=2)),
            ]
        )
        self.out_conv = norm_f(Conv1d(1024, 1, 3, 1, padding=1))

    def forward(self, x):
        features = []
        for conv in self.convs:
            x = conv(x)
            x = F.leaky_relu(x, LRELU_SLOPE)
            features.append(x)
        x = self.out_conv(x)
        features.append(x)
        x = torch.flatten(x, 1, -1)

        return x, features


class MultiScaleDiscriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.discriminators = nn.ModuleList(
            [
                SubDiscriminatorScale(use_spectral_norm=True),
                SubDiscriminatorScale(),
                SubDiscriminatorScale(),
            ]
        )
        self.meanpools = nn.ModuleList(
            [nn.Identity(), AvgPool1d(4, 2, padding=2), AvgPool1d(4, 2, padding=2)]
        )

    def process_wav(self, wav):
        preds = []
        features = []
        for pool, subdisc in zip(self.meanpools, self.discriminators):
            wav = pool(wav)
            sub_pred, sub_features = subdisc(wav)
            preds.append(sub_pred)
            features.append(sub_features)
        return preds, features

    def forward(self, real_wav, fake_wav) -> DiscriminatorOutput:
        real_preds, real_features = self.process_wav(real_wav)
        fake_preds, fake_features = self.process_wav(fake_wav)

        return DiscriminatorOutput(real_preds, fake_preds, real_features, fake_features)
