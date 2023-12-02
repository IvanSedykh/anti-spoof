import os
from pathlib import Path

from hydra import initialize, initialize_config_module, initialize_config_dir, compose
import hydra
from hydra.utils import get_original_cwd, to_absolute_path
import numpy as np
import torch
import wandb
import pandas as pd
from dotenv import load_dotenv
from omegaconf import OmegaConf, DictConfig
from safetensors.torch import load_file
from scipy.io.wavfile import write
import torchaudio

from src.model.hifigan import (
    Generator,
    GeneratorConfig,
)
from src.transforms.mel import MelSpectrogram


load_dotenv()


def load_data(audio_dir: str) -> dict:
    path = Path(audio_dir)
    wavs = []
    fnames = []
    for wav_name in path.glob("*.wav"):
        wav, sr = torchaudio.load(wav_name)
        wavs.append(wav)
        fnames.append(wav_name.name)
    return {"wav": wavs, "fname": fnames}


def load_checkpoint(generator: Generator, c_path: str):
    state_dict = load_file(c_path, device="cuda")
    generator.load_state_dict(state_dict)
    generator.eval()


def get_checkpoint_fnames(c_dir: Path):
    return list(c_dir.rglob("*.safetensors"))


@hydra.main(config_path="config", config_name="config")
@torch.no_grad()
def main(config: DictConfig):
    test_audio_dir = to_absolute_path(config.test_audio_dir)

    data = load_data(test_audio_dir)

    generator_config = GeneratorConfig(**config.generator_config)
    generator = Generator(generator_config).cuda()
    mel_transform = MelSpectrogram().cuda()

    # scans all checkpoints in the subtree
    checkpoint_dir = Path(to_absolute_path(config.checkpoint_dir))
    model_c_fnames = get_checkpoint_fnames(checkpoint_dir)

    with wandb.init(config=OmegaConf.to_container(config)) as run:
        records = []

        os.makedirs("results", exist_ok=True)

        for c_fname in model_c_fnames:
            load_checkpoint(generator, c_fname)
            step = str(c_fname.parent.name).split("-")[-1]

            for i, real_wav in enumerate(data["wav"]):
                mel = mel_transform(real_wav.reshape(1, -1).cuda())

                generated_wav = generator(mel)

                audio_numpy = generated_wav.reshape(-1).cpu().numpy()

                rate = 22050
                audio_out_path = f"results/generated_{data['fname'][i]}"
                write(audio_out_path, rate, audio_numpy)
                records.append(
                    {
                        "fname": data["fname"][i],
                        "gen_audio": wandb.Audio(audio_out_path),
                        "step": step,
                    }
                )

        run.log({"samples": wandb.Table(dataframe=pd.DataFrame.from_records(records))})


if __name__ == "__main__":
    main()
