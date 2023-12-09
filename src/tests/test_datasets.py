import pytest
import torch
from torch.utils.data import DataLoader

from src.datasets.asvspoof_dataset import ASV_Dataset
from src.collate_fn.collate import collate_fn
from src.utils import ROOT_PATH


def print_dict_shape(d):
    for k, v in d.items():
        if isinstance(v, torch.Tensor):
            print(k, v.shape)
        else:
            print(k, v)

def test_dataset():
    dataset = ASV_Dataset(ROOT_PATH / "data" / "LA", "train")
    assert len(dataset) == 25380

    item = dataset[0]
    assert 'wav' in item
    assert 'label' in item

    print_dict_shape(item)

def test_dataloader():
    dataset = ASV_Dataset(ROOT_PATH / "data" / "LA", "dev")
    dataloader = DataLoader(dataset, batch_size=32, collate_fn=collate_fn)
    batch = next(iter(dataloader))
    print_dict_shape(batch)
