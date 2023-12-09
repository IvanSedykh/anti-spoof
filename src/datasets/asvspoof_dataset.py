from pathlib import Path
from typing import Any, Tuple, Union
from torch import Tensor
import torch
from torch.utils.data import Dataset
from torchaudio import load
import numpy as np


class ASV_Dataset(Dataset):
    MAX_FRAMES = 64000

    def __init__(self, data_path: str, split: str) -> None:
        """_summary_

        Args:
            data_path (str): path to LA directory
            split (str): train dev or eval
        """
        assert split in ['train', 'dev', 'eval']
        self.split = split

        audio_dir = Path(data_path) / f"ASVspoof2019_LA_{split}" / "flac"

        # read cm protocol
        protocol_dir = Path(data_path) / f"ASVspoof2019_LA_cm_protocols"
        # find fname containing split
        protocol_file = list(protocol_dir.glob(f"*{split}*"))[0]
        # read protocol file
        protocol_data = np.loadtxt(protocol_file, dtype=str)
        ids = protocol_data[:, 1]
        labels = protocol_data[:, 4]
        label_dict = dict(zip(ids, labels))

        self.items = []
        for audio_id, label in label_dict.items():
            # set 0 to spoof and 1 to bona fide
            assert label in ['spoof', 'bonafide']
            label = 0 if label == 'spoof' else 1
            audio_file = audio_dir / f"{audio_id}.flac"
            self.items.append({
                'audio_file': audio_file,
                'label': label,
                'audio_id': audio_id,
            })


    def __getitem__(self, index: Any) -> Any:
        item = self.items[index]
        wav, sr = load(item['audio_file'])
        label = item['label']
        # take first channel, first 64000 frames
        wav = wav[0, :ASV_Dataset.MAX_FRAMES]

        return {
            'wav': wav,
            'label': label,
            'audio_id': item['audio_id'],
        }

    def __len__(self):
        return len(self.items)