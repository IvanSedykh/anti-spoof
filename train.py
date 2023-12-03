import argparse
import collections
from typing import Any
import warnings
import logging

import numpy as np
import torch
from torch import nn
import hydra
from hydra.utils import instantiate
from omegaconf import OmegaConf, DictConfig
from dotenv import load_dotenv
from torch.utils.data import DataLoader
from accelerate import Accelerator
import transformers
import wandb


from src.model.hifigan import (
    Generator,
    MultiScaleDiscriminator,
    MultiPeriodDiscriminator,
    GeneratorConfig,
)
from src.loss.ganloss import GeneratorLoss, DiscriminatorLoss, FeatureLoss
from src.transforms.mel import MelSpectrogram
from src.utils import prepare_device, count_params, inf_loop
from src.utils.object_loading import get_datasets
from src.collate_fn.collate import collate_fn


warnings.filterwarnings("ignore", category=UserWarning)

# setup logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# fix random seeds for reproducibility
SEED = 0xDEADBEEF
torch.manual_seed(SEED)
# let's go fast boi
torch.backends.cudnn.benchmark = True
np.random.seed(SEED)

load_dotenv()


@hydra.main(config_path="config", config_name="config")
def main(config: DictConfig):
    print(OmegaConf.to_yaml(config))
    print(config)

    # setup data_loader instances
    datasets = get_datasets(config)

    train_loader = DataLoader(
        datasets["train"],
        batch_size=config.data.train.batch_size,
        shuffle=True,
        num_workers=config.data.train.num_workers,
        # collate_fn=collate_fn,
        pin_memory=True,
        drop_last=True,
    )

    mel_transform = MelSpectrogram()

    # build model architecture, then print to console
    generator_config = GeneratorConfig(**config.generator_config)
    generator = Generator(generator_config)
    generator.train()
    logger.info(generator)
    print(f"# parameters generator: {count_params(generator)}")

    msd = MultiScaleDiscriminator()
    msd.train()
    logger.info(msd)
    print(f"# parameters MSD: {count_params(msd)}")

    mpd = MultiPeriodDiscriminator()
    mpd.train()
    logger.info(mpd)
    print(f"# parameters MPD: {count_params(mpd)}")

    # setup losses
    generator_loss = GeneratorLoss()
    discriminator_loss = DiscriminatorLoss()
    feature_loss = FeatureLoss()
    mel_loss = nn.L1Loss()

    # setup optimizer
    optimizer_g = torch.optim.AdamW(
        generator.parameters(),
        lr=config.optimizer.lr,
        betas=config.optimizer.betas,
        weight_decay=config.optimizer.weight_decay,
        eps=config.optimizer.eps,
    )
    scheduler_g = transformers.get_linear_schedule_with_warmup(
        optimizer_g,
        num_training_steps=config.trainer_args.max_steps,
        num_warmup_steps=config.trainer_args.warmup_steps,
    )

    optimizer_d = torch.optim.AdamW(
        list(msd.parameters()) + list(mpd.parameters()),
        lr=config.optimizer.lr,
        betas=config.optimizer.betas,
        weight_decay=config.optimizer.weight_decay,
        eps=config.optimizer.eps,
    )
    scheduler_d = transformers.get_linear_schedule_with_warmup(
        optimizer_d,
        num_training_steps=config.trainer_args.max_steps,
        num_warmup_steps=config.trainer_args.warmup_steps,
    )

    # setup accelerator
    accelerator = Accelerator(log_with="wandb", step_scheduler_with_optimizer=False)
    accelerator.init_trackers("nv_dla", config=OmegaConf.to_container(config))
    (
        generator,
        msd,
        mpd,
        optimizer_g,
        scheduler_g,
        optimizer_d,
        scheduler_d,
        train_loader,
    ) = accelerator.prepare(
        generator,
        msd,
        mpd,
        optimizer_g,
        scheduler_g,
        optimizer_d,
        scheduler_d,
        train_loader,
    )
    mel_transform = mel_transform.to(accelerator.device)

    for step, batch in enumerate(inf_loop(train_loader)):
        if step >= config.trainer_args.max_steps:
            break

        real_wavs = batch["wav"]
        real_mels = mel_transform(real_wavs)

        fake_wav = generator(real_mels)
        fake_mels = mel_transform(fake_wav.squeeze())

        # ======== Discriminator ========
        optimizer_d.zero_grad()
        # MSD
        # d-discriminator, s - scale
        (
            d_s_real_predictions,
            d_s_fake_predictions,
            d_s_real_features,
            d_s_fake_features,
        ) = msd(real_wavs.unsqueeze(1), fake_wav.detach())
        d_loss_msd = discriminator_loss(d_s_real_predictions, d_s_fake_predictions)

        # MPD
        (
            d_p_real_predictions,
            d_p_fake_predictions,
            d_p_real_features,
            d_p_fake_features,
        ) = mpd(real_wavs.unsqueeze(1), fake_wav.detach())
        d_loss_mpd = discriminator_loss(d_p_real_predictions, d_p_fake_predictions)

        d_loss = d_loss_mpd + d_loss_msd
        # d_loss.backward()
        accelerator.backward(d_loss)
        grad_norm_msd = accelerator.clip_grad_norm_(
            msd.parameters(), config.trainer_args.max_grad_norm
        )
        grad_norm_mpd = accelerator.clip_grad_norm_(
            mpd.parameters(), config.trainer_args.max_grad_norm
        )
        optimizer_d.step()
        scheduler_d.step()

        # ======== Generator ========
        optimizer_g.zero_grad()
        (
            d_s_real_predictions,
            d_s_fake_predictions,
            d_s_real_features,
            d_s_fake_features,
        ) = msd(real_wavs.unsqueeze(1), fake_wav)
        (
            d_p_real_predictions,
            d_p_fake_predictions,
            d_p_real_features,
            d_p_fake_features,
        ) = mpd(real_wavs.unsqueeze(1), fake_wav)

        # ADV loss
        g_loss_msd = generator_loss(d_s_fake_predictions)
        g_loss_mpd = generator_loss(d_p_fake_predictions)
        g_adv_loss = g_loss_msd + g_loss_mpd
        # feature loss
        g_loss_feature_msd = feature_loss(d_s_real_features, d_s_fake_features)
        g_loss_feature_mpd = feature_loss(d_p_real_features, d_p_fake_features)
        g_feature_loss = g_loss_feature_mpd + g_loss_feature_msd
        # mel loss
        g_loss_mel = mel_loss(real_mels, fake_mels)
        # todo: parametrize weights with config
        g_loss = (
            g_adv_loss
            + 2 * g_feature_loss
            + 45 * g_loss_mel
        )
        # g_loss.backward()
        accelerator.backward(g_loss)
        grad_norm_g = accelerator.clip_grad_norm_(
            generator.parameters(), config.trainer_args.max_grad_norm
        )
        optimizer_g.step()
        scheduler_g.step()

        if step % config.trainer_args.logging_steps == 0:
            learning_rate_g = optimizer_g.param_groups[0]["lr"]
            learning_rate_d = optimizer_d.param_groups[0]["lr"]

            real_spectrogram_sample = real_mels[0].detach().cpu().unsqueeze(2).numpy()
            fake_spectrogram_sample = fake_mels[0].detach().cpu().unsqueeze(2).numpy()
            real_wav_sample = real_wavs[0].cpu().numpy()
            fake_wav_sample = fake_wav[0].detach().cpu().squeeze().numpy()

            # todo: log 3 test audio

            accelerator.log(
                {
                    "d_loss": d_loss,
                    "d_loss_msd": d_loss_msd,
                    "d_loss_mpd": d_loss_mpd,
                    "g_loss": g_loss,
                    "g_loss_mel": g_loss_mel,
                    "g_loss_msd": g_loss_msd,
                    "g_loss_mpd": g_loss_mpd,
                    "g_loss_feature_msd": g_loss_feature_msd,
                    "g_loss_feature_mpd": g_loss_feature_mpd,
                    "grad_norm_mpd": grad_norm_mpd,
                    "grad_norm_msd": grad_norm_msd,
                    "grad_norm_g": grad_norm_g,
                    "lr_g": learning_rate_g,
                    "lr_d": learning_rate_d,
                    "true_mel": wandb.Image(real_spectrogram_sample),
                    "gen_mel": wandb.Image(fake_spectrogram_sample),
                    "true_wav": wandb.Audio(real_wav_sample, sample_rate=22050),
                    "gen_wav": wandb.Audio(fake_wav_sample, sample_rate=22050),
                },
                step=step,
            )

            logger.info(
                f"step: {step}, d_loss: {d_loss.item():.4f}, g_loss: {g_loss.item():.4f}, g_loss_mel: {g_loss_mel.item():.4f}"
            )

        if step % config.trainer_args.save_steps == 0 and step > 0:
            chkpt_dir = f"{config.trainer_args.output_dir}/checkpoints/step-{step}"
            accelerator.save_model(
                generator, save_directory=chkpt_dir, safe_serialization=True
            )

    accelerator.end_training()


if __name__ == "__main__":
    main()
