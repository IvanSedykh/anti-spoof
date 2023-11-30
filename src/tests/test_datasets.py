import pytest
import torch
from torch.utils.data import DataLoader

from src.datasets.ljspeech_dataset import LJSpeechDataset
from src.collate_fn.collate import collate_fn
from src.utils import ROOT_PATH


def print_dict_shape(d):
    for k, v in d.items():
        if isinstance(v, torch.Tensor):
            print(k, v.shape)
        else:
            print(k, v)

def test_dataset():
    dataset = LJSpeechDataset(root='data', duration=0.1)
    print(f"{len(dataset)=}")
    item = dataset[0]
    print(item)
    print_dict_shape(item)
    assert item['wav'].ndim == 1
    assert item['wav'].shape[0] % 256 == 0
