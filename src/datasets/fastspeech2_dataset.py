import os
import time
import numpy as np
from pathlib import Path
from omegaconf import DictConfig
from hydra.utils import to_absolute_path
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from src.utils.text import text_to_sequence
from .fastspeech_dataset import process_text


def get_data_to_buffer(train_config):
    buffer = list()
    text = process_text(to_absolute_path(train_config.data_path))
    if hasattr(train_config, "limit"):
        text = text[: train_config.limit]

    start = time.perf_counter()
    for i in tqdm(range(len(text))):
        # load mel
        mel_gt_name = os.path.join(
            train_config.mel_ground_truth, "ljspeech-mel-%05d.npy" % (i + 1)
        )
        mel_gt_name = to_absolute_path(mel_gt_name)
        mel_gt_target = np.load(mel_gt_name)

        # load pitch
        pitch_gt_name = (
            Path(train_config.pitch_path) / f"ljspeech-pitch-{(i + 1):05d}.npy"
        )
        pitch_gt_name = to_absolute_path(pitch_gt_name)
        pitch_gt = np.load(pitch_gt_name)

        # load energy
        energy_gt_name = (
            Path(train_config.energy_path) / f"ljspeech-energy-{(i + 1):05d}.npy"
        )
        energy_gt_name = to_absolute_path(energy_gt_name)
        energy_gt = np.load(energy_gt_name)

        # load duration target
        duration = np.load(
            to_absolute_path(os.path.join(train_config.alignment_path, str(i) + ".npy"))
        )

        # load text
        character = text[i][0 : len(text[i]) - 1]
        character = np.array(text_to_sequence(character, train_config.text_cleaners))

        character = torch.from_numpy(character)
        duration = torch.from_numpy(duration)
        mel_gt_target = torch.from_numpy(mel_gt_target)
        pitch_gt = torch.from_numpy(pitch_gt)
        energy_gt = torch.from_numpy(energy_gt)

        buffer.append(
            {
                "text": character,
                "duration": duration,
                "mel_target": mel_gt_target,
                "pitch_target": pitch_gt,
                "energy_target": energy_gt,
            }
        )

    end = time.perf_counter()
    print("cost {:.2f}s to load all data into buffer.".format(end - start))

    return buffer


class FastSpeech2Dataset(Dataset):
    def __init__(self, buffer):
        self.buffer = buffer

    @classmethod
    def from_config(cls, train_config: DictConfig):
        buffer = get_data_to_buffer(train_config)
        return cls(buffer)

    def __len__(self):
        return len(self.buffer)

    def __getitem__(self, idx):
        return self.buffer[idx]
