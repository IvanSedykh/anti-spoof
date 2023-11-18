import os
import time
import numpy as np
from omegaconf import DictConfig
from hydra.utils import to_absolute_path
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from src.utils.text import text_to_sequence


def process_text(train_text_path):
    with open(train_text_path, "r", encoding="utf-8") as f:
        txt = []
        for line in f.readlines():
            txt.append(line)

        return txt


def get_data_to_buffer(train_config):
    buffer = list()
    text = process_text(to_absolute_path(train_config.data_path))

    start = time.perf_counter()
    for i in tqdm(range(len(text))):
        mel_gt_name = os.path.join(
            train_config.mel_ground_truth, "ljspeech-mel-%05d.npy" % (i + 1)
        )
        mel_gt_name = to_absolute_path(mel_gt_name)
        mel_gt_target = np.load(mel_gt_name)
        duration = np.load(
            to_absolute_path(os.path.join(train_config.alignment_path, str(i) + ".npy"))
        )
        character = text[i][0 : len(text[i]) - 1]
        character = np.array(text_to_sequence(character, train_config.text_cleaners))

        character = torch.from_numpy(character)
        duration = torch.from_numpy(duration)
        mel_gt_target = torch.from_numpy(mel_gt_target)

        buffer.append(
            {"text": character, "duration": duration, "mel_target": mel_gt_target}
        )

    end = time.perf_counter()
    print("cost {:.2f}s to load all data into buffer.".format(end - start))

    return buffer


class BufferDataset(Dataset):
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
