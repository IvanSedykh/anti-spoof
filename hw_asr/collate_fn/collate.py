import logging
from typing import List

import torch

logger = logging.getLogger(__name__)


def collate_fn(dataset_items: List[dict]):
    """
    Collate and pad fields in dataset items
    """

    result_batch = {}
    # TODO: your code here

    # stack torch tensors, collect strings to list
    # and pad them to the same length

    result_batch["mix_wav"] = torch.stack(
        [torch.squeeze(item["mix_wav"]) for item in dataset_items]
    )

    # use padding for ref wav
    result_batch["ref_wav"] = torch.nn.utils.rnn.pad_sequence(
        [torch.squeeze(item["ref_wav"]) for item in dataset_items], batch_first=True
    )
    result_batch["ref_wav_len"] = torch.tensor(
        [item["ref_wav_len"] for item in dataset_items]
    )

    result_batch["target_wav"] = torch.stack(
        [torch.squeeze(item["target_wav"]) for item in dataset_items]
    )

    result_batch["speaker_id"] = torch.tensor(
        [item["speaker_id"] for item in dataset_items]
    )

    return result_batch
