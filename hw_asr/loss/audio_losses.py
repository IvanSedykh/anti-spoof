from torch import nn

from torchmetrics.functional.audio import scale_invariant_signal_distortion_ratio


def si_sdr(preds, target):
    return scale_invariant_signal_distortion_ratio(preds, target).mean()


class SI_SDR_Loss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, preds, target, **batch):
        val = -si_sdr(preds, target)
        return val


class SpexLoss(nn.Module):
    def __init__(self, alpha: float, beta: float, gamma: float):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

    def forward(self, inputs: dict, outputs: dict):
        pred_wav = outputs["predict_wav"]  # short
        middle_wav = outputs["middle_wav"]
        long_wav = outputs["long_wav"]
        speaker_logits = outputs["speaker_logits"]

        target_wav = inputs["target_wav"]

        return -1 * (
            (1 - self.alpha - self.beta) * si_sdr(pred_wav, target_wav)
            + self.alpha * si_sdr(middle_wav, target_wav)
            + self.beta * si_sdr(long_wav, target_wav)
        ) + self.gamma * nn.functional.cross_entropy(
            speaker_logits, inputs["speaker_id"]
        )
