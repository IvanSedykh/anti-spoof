import logging
from pathlib import Path
import random
from typing import List

import numpy as np
import torch
import torchaudio
from torch import Tensor
from torch.utils.data import Dataset

from src.base.base_text_encoder import BaseTextEncoder
from src.utils.parse_config import ConfigParser

logger = logging.getLogger(__name__)


class BaseDataset(Dataset):
    def __init__(
            self,
            index,
            config_parser: ConfigParser,
            wave_augs=None,
            limit=None,
            max_audio_length=None,
    ):
        self.config_parser = config_parser
        self.wave_augs = wave_augs

        self._assert_index_is_valid(index)
        index = self._filter_records_from_dataset(index, max_audio_length, limit)
        # it's a good idea to sort index by audio length
        # It would be easier to write length-based batch samplers later
        index = self._sort_index(index)
        self._index: List[dict] = index

    def __getitem__(self, ind):
        data_dict = self._index[ind]
        ref_path = data_dict["ref_path"]
        mix_path = data_dict["mix_path"]
        target_path = data_dict["target_path"]
        speaker_id = data_dict.get("speaker_id", -100)
        speaker_id = torch.tensor(speaker_id, dtype=torch.long)

        ref_wave = self.load_audio(ref_path)
        mix_wave = self.load_audio(mix_path)
        target_wave = self.load_audio(target_path)

    
        return {
            "ref_wav": ref_wave,
            "ref_wav_len": ref_wave.size(1),
            "mix_wav": mix_wave,
            "target_wav": target_wave,
            "speaker_id": speaker_id,
            # durations
            "mix_duration": mix_wave.size(1) / self.config_parser["preprocessing"]["sr"],
            "ref_duration": ref_wave.size(1) / self.config_parser["preprocessing"]["sr"],
            "target_duration": target_wave.size(1) / self.config_parser["preprocessing"]["sr"],
            # paths
            "ref_path": ref_path,
            "mix_path": mix_path,
            "target_path": target_path,
        }

    @staticmethod
    def _sort_index(index):
        return sorted(index, key=lambda x: x["audio_len"])

    def __len__(self):
        return len(self._index)

    def load_audio(self, path):
        audio_tensor, sr = torchaudio.load(path)
        audio_tensor = audio_tensor[0:1, :]  # remove all channels but the first
        target_sr = self.config_parser["preprocessing"]["sr"]
        if sr != target_sr:
            audio_tensor = torchaudio.functional.resample(audio_tensor, sr, target_sr)
        return audio_tensor

    def process_wave(self, audio_tensor_wave: Tensor):
        with torch.no_grad():
            if self.wave_augs is not None:
                audio_tensor_wave = self.wave_augs(audio_tensor_wave)
            return audio_tensor_wave

    @staticmethod
    def _filter_records_from_dataset(
            index: list, max_audio_length, limit
    ) -> list:
        initial_size = len(index)
        if max_audio_length is not None:
            exceeds_audio_length = np.array([el["audio_len"] for el in index]) > max_audio_length
            _total = exceeds_audio_length.sum()
            logger.info(
                f"{_total} ({_total / initial_size:.1%}) records are longer then "
                f"{max_audio_length} seconds. Excluding them."
            )
        else:
            exceeds_audio_length = False

        initial_size = len(index)

        records_to_filter = exceeds_audio_length

        if records_to_filter is not False and records_to_filter.any():
            _total = records_to_filter.sum()
            index = [el for el, exclude in zip(index, records_to_filter) if not exclude]
            logger.info(
                f"Filtered {_total}({_total / initial_size:.1%}) records  from dataset"
            )

        if limit is not None:
            random.seed(42)  # best seed for deep learning
            random.shuffle(index)
            index = index[:limit]
        logger.info(f"Final total dataset len {len(index)}")
        return index

    @staticmethod
    def _assert_index_is_valid(index):
        for entry in index:
            assert "ref_path" in entry, f"Entry {entry} does not have ref_path"
            assert "mix_path" in entry, f"Entry {entry} does not have mix_path"
            assert "target_path" in entry, f"Entry {entry} does not have target_path"
            assert "audio_len" in entry, f"Entry {entry} does not have audio_len"
            # check that all files exist
            for k, v in entry.items():
                if k.endswith("_path"):
                    assert Path(v).exists(), f"File {v} does not exist"