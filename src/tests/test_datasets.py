import pytest
import torch
from torch.utils.data import DataLoader

from src.datasets.fastspeech_dataset import BufferDataset, get_data_to_buffer
from src.collate_fn.collate import make_collate_fn_tensor
from src.utils import ROOT_PATH

from src.tests.test_fastspeech import train_config

def test_fastspeechdataset(train_config):
    buffer = get_data_to_buffer(train_config)
    dataset = BufferDataset(buffer)

    item = dataset[0]
    # print(item)

    collate_fn = make_collate_fn_tensor(train_config.batch_expand_size)

    loader = DataLoader(
        dataset,
        batch_size=train_config.batch_size * train_config.batch_expand_size,
        drop_last=True,
        collate_fn=collate_fn
    )

    for batch in loader:
        break
    # print(batch)
