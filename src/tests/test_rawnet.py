import pytest

import torch

from src.model.modules import SincConv_fast
from src.model.rawnet import InputLayer, FMS, ResBlock, RawNet
from src.utils import count_params

def test_sincconv():
    sincconv = SincConv_fast(out_channels=128, kernel_size=129, sample_rate=16000)

    bs = 32
    x = torch.randn(bs, 1, 64_000)
    y = sincconv(x)
    assert y.shape == (bs, 128, y.shape[2])
    assert y.shape[2] // 3 == 21_290


def test_input_layer():
    input_layer = InputLayer()
    bs = 32
    x = torch.randn(bs, 1, 64_000)
    y = input_layer(x)
    assert y.shape == (bs, 128, 21_290)

def test_fms():
    fms = FMS(num_channels=128)
    bs = 32

    SEQ_LEN = 2048
    x = torch.randn(bs, 128, SEQ_LEN)
    y = fms(x)
    assert y.shape == (bs, 128, SEQ_LEN)

def test_resblock():
    resblock = ResBlock(in_channels=128, out_channels=128)
    bs = 32

    SEQ_LEN = 2048
    x = torch.randn(bs, 128, SEQ_LEN)
    y = resblock(x)
    assert y.shape == (bs, 128, SEQ_LEN // 3)

def test_resblock_downsample():
    resblock = ResBlock(in_channels=128, out_channels=256)
    bs = 32

    SEQ_LEN = 2048
    x = torch.randn(bs, 128, SEQ_LEN)
    y = resblock(x)
    assert y.shape == (bs, 256, SEQ_LEN // 3)

def test_rawnet():
    channels_list = [128] * 2 + [512] * 4
    model = RawNet(channels_list=channels_list)
    print(f"# parameters: {count_params(model)}")

    bs = 32
    x = torch.randn(bs, 1, 64_000)
    y = model(x)
    assert y.shape == (bs, 2)
