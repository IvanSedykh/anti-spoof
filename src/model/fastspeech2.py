import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .fastspeech import Encoder, Decoder, LengthRegulator, get_mask_from_lengths
from .fastspeech import DurationPredictor as Predictor


def print_hist(tensor: torch.Tensor):
    tensor = tensor.cpu().float()
    hist, bins = torch.histogram(tensor, bins=10)
    print(f"{hist=}")
    print(f"{bins=}")
    print(f"{tensor.mean()=}")
    print(f"{tensor.std()=}")


class EmbeddingRegulator(nn.Module):
    """For pitch/energy regulaton"""

    def __init__(self, model_config, regulator_config):
        super().__init__()
        # regulator config -- specific config for pitch/energy

        # use the same params for all predictors
        self.predictor = Predictor(model_config)

        # TODO: may be apply exponent somewhere
        bins = torch.linspace(
            np.log(regulator_config.min),
            np.log(regulator_config.max),
            regulator_config.n_bins - 1,
        )
        bins = torch.exp(bins)
        self.register_buffer("bins", bins)

        self.embedding = nn.Embedding(
            regulator_config.n_bins, embedding_dim=model_config.encoder_dim
        )

    def forward(self, frames, target=None, control=1.0):
        # predicts log + 1 of actual value
        prediction = self.predictor(frames)

        # inference
        if target is None:
            # inverse log1p
            prediction = torch.expm1(prediction)
            prediction = prediction * control
            emb = self.embedding(torch.bucketize(prediction, self.bins))
        # training
        else:
            classes = torch.bucketize(target, self.bins)

            # print_hist(classes)
            emb = self.embedding(classes)
        return prediction, emb


class FastSpeech2(nn.Module):
    """FastSpeech 2222"""

    def __init__(self, model_config):
        super().__init__()

        self.encoder = Encoder(model_config)
        self.length_regulator = LengthRegulator(model_config)
        self.pitch_regulator = EmbeddingRegulator(
            model_config, model_config.pitch_regulator_config
        )
        self.energy_regulator = EmbeddingRegulator(
            model_config, model_config.energy_regulator_config
        )

        self.decoder = Decoder(model_config)

        self.mel_linear = nn.Linear(model_config.decoder_dim, model_config.num_mels)

    def mask_tensor(self, mel_output, position, mel_max_length):
        lengths = torch.max(position, -1)[0]
        mask = ~get_mask_from_lengths(lengths, max_len=mel_max_length)
        mask = mask.unsqueeze(-1).expand(-1, -1, mel_output.size(-1))
        return mel_output.masked_fill(mask, 0.0)

    # todo: pass args as dict
    def forward(
        self,
        src_seq,
        src_pos,
        mel_pos=None,
        mel_max_length=None,
        length_target=None,
        pitch_target=None,
        energy_target=None,
        speed_control=1.0,
        pitch_control=1.0,
        energy_control=1.0,
        **kwargs,
    ):
        encoder_output, _ = self.encoder(src_seq, src_pos)

        if self.training:
            length_regulator_output, duration_predictor_output = self.length_regulator(
                encoder_output,
                target=length_target,
                alpha=speed_control,
                mel_max_length=mel_max_length,
            )

            pitch_pred, pitch_emb = self.pitch_regulator(
                length_regulator_output, pitch_target, pitch_control
            )
            energy_pred, energy_emb = self.energy_regulator(
                length_regulator_output, energy_target, energy_control
            )

            adapter_outputs = length_regulator_output + pitch_emb + energy_emb

            decoder_output = self.decoder(adapter_outputs, mel_pos)

            mel_output = self.mel_linear(decoder_output)
            mel_output = self.mask_tensor(mel_output, mel_pos, mel_max_length)

            # return mel_output, duration_predictor_output
            return {
                "mel_output": mel_output,
                "duration_predictor_output": duration_predictor_output,
                "pitch_predictor_output": pitch_pred,
                "energy_predictor_output": energy_pred,
            }
        else:
            length_regulator_output, decoder_pos = self.length_regulator(
                encoder_output, alpha=speed_control
            )
            _, pitch_emb = self.pitch_regulator(
                length_regulator_output, control=pitch_control
            )
            _, energy_emb = self.energy_regulator(
                length_regulator_output, control=energy_control
            )
            adapter_outputs = length_regulator_output + pitch_emb + energy_emb

            decoder_output = self.decoder(adapter_outputs, decoder_pos)

            mel_output = self.mel_linear(decoder_output)

            # return mel_output
            return {"mel_output": mel_output}
