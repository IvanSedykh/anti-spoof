from torch import nn

from torchmetrics.audio import ScaleInvariantSignalDistortionRatio
from torchmetrics.functional.audio import scale_invariant_signal_distortion_ratio, perceptual_evaluation_speech_quality

from src.base.base_metric import BaseMetric

class SI_SDR_Metric(BaseMetric):
    def __init__(self, name=None, *args, **kwargs):
        super().__init__(name, *args, **kwargs)
    
    def __call__(self, preds, target, **batch):
        val = scale_invariant_signal_distortion_ratio(preds, target).mean()
        return val
    
class PESQ_Metric(BaseMetric):
    def __init__(self, name=None, *args, **kwargs):
        super().__init__(name, *args, **kwargs)

    def __call__(self, preds, target, **batch):
        val = perceptual_evaluation_speech_quality(
            preds, target, fs=16_000, mode="wb", n_processes=4
            ).mean()
        return val

