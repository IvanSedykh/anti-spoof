import torch
import torch.nn as nn
from torch.nn import functional as F



class FastSpeechLossInner(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse_loss = nn.MSELoss()
        self.l1_loss = nn.L1Loss()

    def forward(self, mel, duration_predicted, mel_target, duration_predictor_target):
        mel_loss = self.mse_loss(mel, mel_target)

        duration_predictor_loss = self.l1_loss(duration_predicted,
                                               duration_predictor_target.float())

        return mel_loss, duration_predictor_loss
    

class FastSpeechLoss(FastSpeechLossInner):
    def forward(self, inputs: dict, outputs: dict):
        mel = outputs["mel_output"]
        mel_target = inputs["mel_target"]
        duration_predicted = outputs["duration_predictor_output"]
        duration_predictor_target = inputs["length_target"]

        mel_loss, duration_predictor_loss = super().forward(mel, duration_predicted, mel_target, duration_predictor_target)
        return {
            "mel_loss": mel_loss,
            "duration_predictor_loss": duration_predictor_loss
        }


class FastSpeech2Loss(nn.Module):
    def __init__(self,):
        super().__init__()

    def forward(self, inputs: dict, outputs: dict):
        mel = outputs["mel_output"]
        mel_target = inputs["mel_target"]

        duration_predicted = outputs["duration_predictor_output"]
        duration_predictor_target = inputs["length_target"]

        pitch_predicted = outputs['pitch_predictor_output']
        pitch_target = inputs['pitch_target']
        pitch_target = torch.log1p(pitch_target)

        energy_predicted = outputs['energy_predictor_output']
        energy_target = inputs['energy_target']
        energy_target = torch.log1p(energy_target)

        mel_loss = F.mse_loss(mel, mel_target)
        duration_predictor_loss = F.l1_loss(duration_predicted, duration_predictor_target.float())
        pitch_predictor_loss = F.mse_loss(pitch_predicted, pitch_target.float())
        energy_predictor_loss = F.mse_loss(energy_predicted, energy_target.float())

        return {
            "mel_loss": mel_loss,
            "duration_predictor_loss": duration_predictor_loss,
            "pitch_predictor_loss": pitch_predictor_loss,
            "energy_predictor_loss": energy_predictor_loss
        }

    