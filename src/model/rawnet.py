import torch
from torch import nn
import torch.nn.functional as F

from src.model.modules import SincConv_fast

LRELU_SLOPE = 0.3


class InputLayer(nn.Module):
    def __init__(self, out_channels: int = 128, kernel_size: int = 1024):
        super().__init__()
        self.sincconv = SincConv_fast(
            out_channels=out_channels,
            kernel_size=kernel_size,
            sample_rate=16000,
            min_low_hz=0,
            min_band_hz=0,
        )
        self.pool = nn.MaxPool1d(3)
        self.bn = nn.BatchNorm1d(out_channels)
        self.lrelu = nn.LeakyReLU(LRELU_SLOPE)

    def forward(self, x):
        x = self.sincconv(x)
        x = self.pool(x)
        x = self.bn(x)
        x = self.lrelu(x)
        return x


class FMS(nn.Module):
    def __init__(self, num_channels: int) -> None:
        super().__init__()
        self.num_channels = num_channels

        self.fc = nn.Linear(num_channels, num_channels)

    def forward(self, x):
        s = F.adaptive_avg_pool1d(x, 1).view(x.size(0), -1)
        s = F.sigmoid(self.fc(s)).view(x.size(0), x.size(1), -1)
        x = x * s
        x = x + s
        return x

        # # x: (bs, num_channels, num_frames)
        # # global avg pool over frames
        # s = x.mean(dim=2)
        # # s: (bs, num_channels)
        # s = F.sigmoid(self.fc(s))
        # # s: (bs, num_channels)
        # x = x * s.unsqueeze(2)
        # x = x + s.unsqueeze(2)
        # # x: (bs, num_channels, num_frames)
        # return x


class ResBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()

        self.conv1 = nn.Conv1d(in_channels, in_channels, kernel_size=3, padding="same")
        self.bn1 = nn.BatchNorm1d(in_channels)
        self.lrelu1 = nn.LeakyReLU(LRELU_SLOPE)

        self.conv2 = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding="same")
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.lrelu2 = nn.LeakyReLU(LRELU_SLOPE)

        self.pool = nn.MaxPool1d(3)

        self.fms = FMS(out_channels)

        if in_channels != out_channels:
            self.downsample = nn.Conv1d(
                in_channels, out_channels, kernel_size=1, padding="same"
            )
            # init weights
            # nn.init.dirac_(self.downsample.weight)
            # nn.init.zeros_(self.downsample.bias)
        else:
            self.downsample = nn.Identity()

    def forward(self, x):
        # x: (bs, in_channels, num_frames)
        residual = x

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.lrelu1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.lrelu2(x)

        residual = self.downsample(residual)

        x = x + residual
        x = self.pool(x)
        x = self.fms(x)
        # x: (bs, out_channels, num_frames//3)
        return x


class TransformerAggregation(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.channels = channels

        self.transformer = nn.TransformerEncoderLayer(
            d_model=channels,
            nhead=4,
            dim_feedforward=channels,
            dropout=0.1,
            activation="gelu",
            norm_first=True,
            batch_first=True,
        )

    def forward(self, x):
        # print(f"transformer input shape: {x.shape}")
        # x: (bs, num_frames, channels)
        hiddens = self.transformer(x)
        # hiddens: (bs, num_frames, channels)
        # take mean over frames
        x = hiddens.mean(dim=1)
        # x: (bs, channels)
        return x


class GRUAggregation(nn.Module):
    def __init__(self, channels: int, out_channels: int):
        super().__init__()
        self.gru = nn.GRU(
            channels,
            out_channels,
            batch_first=True,
            num_layers=3,
            bidirectional=False,
            dropout=0.1,
        )

    def forward(self, x):
        # x: (bs, num_frames, channels)
        x, _ = self.gru(x)
        x = x[:, -1, :]
        # x: (bs, channels)
        return x


class RawNet(nn.Module):
    def __init__(
        self,
        channels_list: list[int],
        agg_hidden_size: int,
        input_abs: bool = True,
        num_classes: int = 2,
        **kwargs
    ) -> None:
        super().__init__()
        self.channels_list = channels_list
        self.num_classes = num_classes
        self.input_abs = input_abs

        self.input_layer = InputLayer(out_channels=channels_list[0])

        self.resblocks = nn.ModuleList()
        for in_channels, out_channels in zip(channels_list[:-1], channels_list[1:]):
            self.resblocks.append(ResBlock(in_channels, out_channels))

        # self.aggregation = TransformerAggregation(channels_list[-1])
        self.aggregation = GRUAggregation(channels_list[-1], agg_hidden_size)

        self.fc = nn.Linear(agg_hidden_size, num_classes)

    def forward(self, x):
        # x: (bs, 1, 64000)
        x = self.input_layer(x)
        if self.input_abs:
            x = torch.abs(x)
        # x: (bs, 128, 21290)
        for resblock in self.resblocks:
            x = resblock(x)
        emb = self.aggregation(x.transpose(1, 2))
        # emb: (bs, agg_hidden_size)
        logits = self.fc(emb)
        # logits: (bs, num_classes)
        return logits
