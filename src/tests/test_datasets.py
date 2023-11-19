import pytest
import torch
from torch.utils.data import DataLoader

from src.datasets.fastspeech_dataset import BufferDataset, get_data_to_buffer
from src.collate_fn.collate import collate_fn
from src.utils import ROOT_PATH

from src.tests.test_fastspeech import train_config

def print_dict_shape(d):
    for k, v in d.items():
        if isinstance(v, torch.Tensor):
            print(k, v.shape)
        else:
            print(k, v)

def test_fastspeechdataset(train_config):
    buffer = get_data_to_buffer(train_config, 10)
    dataset = BufferDataset(buffer)

    item = dataset[0]
    # print(item)
    print(f"Item:")
    print_dict_shape(item)


    loader = DataLoader(
        dataset,
        batch_size=train_config.batch_size,
        drop_last=True,
        collate_fn=collate_fn
    )

    for batch in loader:
        break
    print("Batch:")
    print_dict_shape(batch)
