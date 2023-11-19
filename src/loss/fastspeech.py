import torch
import torch.nn as nn



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
        loss = mel_loss + duration_predictor_loss
        return loss
