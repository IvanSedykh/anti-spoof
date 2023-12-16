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

from src.model.rawnet import RawNet
from src.datasets.asvspoof_dataset import ASV_Dataset


load_dotenv()


def load_data(audio_dir: str) -> dict:
    path = Path(audio_dir)
    wavs = []
    fnames = []
    sample_rates = []
    # glob wav and flac
    for wav_name in path.glob("*.wav"):
        wav, sr = torchaudio.load(wav_name)
        # wav = ASV_Dataset.prepare_wav(wav.reshape(1, -1), sr)
        wavs.append(wav)
        fnames.append(wav_name.name)
        sample_rates.append(sr)

    for flac_name in path.glob("*.flac"):
        wav, sr = torchaudio.load(flac_name)
        # wav = ASV_Dataset.prepare_wav(wav.reshape(1, -1), sr)
        wavs.append(wav)
        fnames.append(flac_name.name)
        sample_rates.append(sr)
    return {"wav": wavs, "fname": fnames, "sr": sample_rates}


def load_checkpoint(model: RawNet, c_path: str):
    # state_dict = load_file(c_path, device="cuda")
    state_dict = torch.load(c_path, map_location='cuda:0')
    model.load_state_dict(state_dict)
    model.eval()


def get_checkpoint_fnames(c_dir: Path):
    # return list(c_dir.rglob("*.safetensors"))
    return list(c_dir.rglob("*.pth"))


@hydra.main(config_path="config", config_name="config")
@torch.no_grad()
def main(config: DictConfig):
    test_audio_dir = to_absolute_path(config.test_audio_dir)

    data = load_data(test_audio_dir)

    model = RawNet(**config.model).cuda()

    # scans all checkpoints in the subtree
    checkpoint_dir = Path(to_absolute_path(config.checkpoint_dir))
    model_c_fnames = get_checkpoint_fnames(checkpoint_dir)

    with wandb.init(config=OmegaConf.to_container(config)) as run:
        records = []

        os.makedirs("results", exist_ok=True)

        for c_fname in model_c_fnames:
            print(f"{c_fname=}")
            load_checkpoint(model, c_fname)

            for i, (wav, sr) in enumerate(zip(data["wav"], data["sr"])):
                wav = ASV_Dataset.prepare_wav(wav, sr).reshape(1, 1, -1).cuda()

                logits = model(wav)

                # softmax
                probs = torch.softmax(logits, dim=1)
                # get bonafide probability
                prob = probs[0, 1].item()

                records.append(
                    {
                        "fname": data["fname"][i],
                        "audio": wandb.Audio(wav.cpu().reshape(-1,).numpy(), sample_rate=ASV_Dataset.SR),
                        "bonafide_prob": prob,
                    }
                )

        run.log({"samples": wandb.Table(dataframe=pd.DataFrame.from_records(records))})


if __name__ == "__main__":
    main()
