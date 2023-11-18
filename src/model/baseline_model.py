import sys
import torch
from torch import nn
from torch.nn import Sequential

from .base_model import BaseModel


class BaselineModel(BaseModel):
    def __init__(self, **batch):
        super().__init__(**batch)
        # heh
        self.scale = nn.Parameter(torch.tensor([1.0], requires_grad=True))
        self.bias = nn.Parameter(torch.tensor([0.0], requires_grad=True))

    def forward(self, **batch):
        mix_wav = batch['mix_wav']
        pred = self.scale * mix_wav + self.bias
        output = {"predict_wav": pred}
        return output