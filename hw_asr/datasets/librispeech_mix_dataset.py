import json
import logging
import os
import shutil
from pathlib import Path

import torchaudio
from speechbrain.utils.data_utils import download_file
from tqdm import tqdm

from hw_asr.base.base_dataset import BaseDataset
from hw_asr.utils import ROOT_PATH

logger = logging.getLogger(__name__)


class LibrispeechMixDataset(BaseDataset):
    def __init__(self, data_dir: str, *args, **kwargs):
        self.data_dir = Path(data_dir)
        index = self._get_or_load_index(data_dir)
        super().__init__(index, *args, **kwargs)

    def _get_or_load_index(self, data_dir: str):
        index_path = Path(data_dir) / "index.json"
        if index_path.exists():
            with index_path.open() as f:
                index = json.load(f)
        else:
            index = self._create_index(data_dir)
            with index_path.open("w") as f:
                json.dump(index, f, indent=2)
        return index

    def _create_index(self, data_dir: str):
        index = []
        dir = Path(data_dir)
        speaker_id_to_class = {}
        for file in dir.glob("*-mixed.wav"):
            speaker_id = file.name.split("_")[0]
            if speaker_id not in speaker_id_to_class:
                speaker_id_to_class[speaker_id] = len(speaker_id_to_class)
            speaker_id = speaker_id_to_class[speaker_id]
            t_info = torchaudio.info(str(file))
            
            length = t_info.num_frames / t_info.sample_rate
            index.append(
                {
                    "mix_path": str(file),
                    "ref_path": str(file).replace("-mixed.wav", "-ref.wav"),
                    "target_path": str(file).replace("-mixed.wav", "-target.wav"),
                    "speaker_id": speaker_id,
                    "audio_len": length,
                }
            )
            # check if all files exist
            for k, v in index[-1].items():
                if k.endswith("_path"):
                    assert Path(v).exists(), f"File {v} does not exist"

        return index
