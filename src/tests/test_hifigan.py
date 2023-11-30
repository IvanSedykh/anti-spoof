import pytest

import torch


from src.model.hifigan import (
    Generator,
    GeneratorConfig,
    MultiPeriodDiscriminator,
    MultiScaleDiscriminator,
)
from src.transforms.mel import MelSpectrogram


def test_generator():
    generator_config = GeneratorConfig(
        upsample_rates=[8, 8, 2, 2],
        upsample_kernel_sizes=[16, 16, 4, 4],
        upsample_initial_channel=128,
        resblock_kernel_sizes=[3, 7, 11],
        resblock_dilation_sizes=[[1, 3, 5], [1, 3, 5], [1, 3, 5]],
    )

    generator = Generator(generator_config)

    mel_transform = MelSpectrogram()

    bs = 5
    wav_len = 5*256
    wav = torch.randn(bs, wav_len)
    print(f"{wav.shape=}")
    mel = mel_transform(wav)
    print(f"{mel.shape=}")

    out = generator(mel)
    print(f"{out.shape=}")
    assert out.shape == (bs, 1, wav_len)


def test_multi_period_discriminator():
    discriminator = MultiPeriodDiscriminator()

    bs = 7
    wav_len = 5000
    generated_wav = torch.randn(bs, wav_len)
    print(f"{generated_wav.shape=}")
    true_wav = torch.randn(bs, wav_len)
    print(f"{true_wav.shape=}")


    y_d_rs, y_d_gs, fmap_rs, fmap_gs = discriminator(true_wav.unsqueeze(1), generated_wav.unsqueeze(1))
    assert len(y_d_rs) == 5
    assert len(y_d_gs) == 5
    assert len(fmap_rs) == 5
    assert len(fmap_gs) == 5
    print(f"{len(y_d_rs)=}")
    for y_d_r in y_d_rs:
        print(f"{y_d_r.shape=}")
        break

    print(f"{len(y_d_gs)=}")
    for y_d_g in y_d_gs:
        print(f"{y_d_g.shape=}")
        break

    print(f"{len(fmap_rs)=}")
    print(f"{len(fmap_gs)=}")


def test_multi_scale_discriminator():
    discriminator = MultiScaleDiscriminator()

    bs = 7
    wav_len = 5000
    generated_wav = torch.randn(bs, wav_len)
    print(f"{generated_wav.shape=}")
    true_wav = torch.randn(bs, wav_len)
    print(f"{true_wav.shape=}")

    
    y_d_rs, y_d_gs, fmap_rs, fmap_gs = discriminator(true_wav.unsqueeze(1), generated_wav.unsqueeze(1))
    NUM_SCALES = 3
    assert len(y_d_rs) == NUM_SCALES
    assert len(y_d_gs) == NUM_SCALES
    assert len(fmap_rs) == NUM_SCALES
    assert len(fmap_gs) == NUM_SCALES
    print(f"{len(y_d_rs)=}")
    for y_d_r in y_d_rs:
        print(f"{y_d_r.shape=}")
        break

    print(f"{len(y_d_gs)=}")
    for y_d_g in y_d_gs:
        print(f"{y_d_g.shape=}")
        break

    print(f"{len(fmap_rs)=}")
    print(f"{len(fmap_gs)=}")


