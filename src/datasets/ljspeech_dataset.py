from pathlib import Path
from typing import Any, Tuple, Union
from torch import Tensor
import torch
from torch.utils.data import Dataset

from torchaudio.datasets import LJSPEECH

from src.transforms.mel import MelSpectrogram


_RELEASE_CONFIGS = {
    "release1": {
        "folder_in_archive": "wavs",
        "url": "https://data.keithito.com/data/speech/LJSpeech-1.1.tar.bz2",
        "checksum": "be1a30453f28eb8dd26af4101ae40cbf2c50413b1bb21936cbcdc6fae3de8aa5",
    }
}


class LJSpeechDataset(LJSPEECH):
    SR = 22050

    def __init__(
        self,
        root: str | Path,
        url: str = _RELEASE_CONFIGS["release1"]["url"],
        folder_in_archive: str = _RELEASE_CONFIGS["release1"]["folder_in_archive"],
        download: bool = False,
        duration: float = 1.0,
    ) -> None:
        # audio patch duraion in seconds rounded to 256
        self.num_frames = int(duration * self.SR // 256 * 256)
        print(f"Dataset num_frames set to {self.num_frames}")
        super().__init__(root, url, folder_in_archive, download)

    def __getitem__(self, n: int) -> Tuple[Tensor, int, str, str]:
        wav, sr, text, text_norm = super().__getitem__(n)
        # cut random part of the audio
        if self.num_frames < wav.size(1):
            start = torch.randint(0, wav.size(1) - self.num_frames, (1,))
            wav = wav[:, start : start + self.num_frames]
        item = {
            "wav": wav.squeeze(0),
        }
        return item
