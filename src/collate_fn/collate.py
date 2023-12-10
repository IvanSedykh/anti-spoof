import logging
from typing import List
import numpy as np

import torch
from torch.nn import functional as F

logger = logging.getLogger(__name__)


def collate_fn(batch: List[dict]) -> dict:
    # batch is a list of dicts 'wav', 'label', 'audio_id'

    # get wav
    wavs = [item["wav"] for item in batch]
    wavs = torch.stack(wavs)
    # get labels
    labels = torch.LongTensor([item["label"] for item in batch])
    # get audio_ids
    audio_ids = [item["audio_id"] for item in batch]

    return {
        "wav": wavs,
        "label": labels,
        "audio_id": audio_ids,
    }
