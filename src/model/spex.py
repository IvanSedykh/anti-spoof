from src.base.base_model import BaseModel
from .nnet.spex_plus import SpEx_Plus


# todo: reimplement
class Spex(SpEx_Plus, BaseModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, **batch):
        x = batch['mix_wav']
        aux = batch['ref_wav']
        aux_len = batch['ref_wav_len']
        output = super().forward(x, aux, aux_len)
        short, middle, long, speaker_logits = output
        output = {
            "predict_wav": short,
            "middle_wav": middle,
            "long_wav": long,
            "speaker_logits": speaker_logits
        }
        return output
