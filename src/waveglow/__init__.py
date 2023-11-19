import os
import torch

# import waveglow.inference
# import waveglow.mel2samp
from . import inference
from . import mel2samp

def get_WaveGlow(path: str):
    # waveglow_path = os.path.join("waveglow", "pretrained_model")
    # waveglow_path = os.path.join(waveglow_path, "waveglow_256channels.pt")
    wave_glow = torch.load(path, map_location='cpu')['model']
    wave_glow = wave_glow.remove_weightnorm(wave_glow)
    # wave_glow.cuda().eval()
    wave_glow.eval()
    for m in wave_glow.modules():
        if 'Conv' in str(type(m)):
            setattr(m, 'padding_mode', 'zeros')

    return wave_glow