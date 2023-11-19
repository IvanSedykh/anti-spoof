import os
from hydra import initialize, initialize_config_module, initialize_config_dir, compose
import hydra
from hydra.utils import get_original_cwd, to_absolute_path
import numpy as np
import torch
from omegaconf import OmegaConf, DictConfig
from safetensors.torch import load_file
from scipy.io.wavfile import write

from src.utils.text import text_to_sequence


def synthesis(model, text, alpha=1.0):
    text = np.array(text)
    text = np.stack([text])
    src_pos = np.array([i+1 for i in range(text.shape[1])])
    src_pos = np.stack([src_pos])
    sequence = torch.from_numpy(text).long().cuda()
    src_pos = torch.from_numpy(src_pos).long().cuda()
    
    with torch.no_grad():
        output = model(sequence, src_pos, alpha=alpha)
        mel = output['mel_output']
    return mel[0].cpu().transpose(0, 1), mel.contiguous().transpose(1, 2)


def get_data():
    tests = [
        "A defibrillator is a device that gives a high energy electric shock to the heart of someone who is in cardiac arrest",
        "Massachusetts Institute of Technology may be best known for its math, science and engineering education",
        "Wasserstein distance or Kantorovich Rubinstein metric is a distance function defined between probability distributions on a given metric space",
        "Printing, in the only sense with which we are at present concerned, differs from most if not from all the arts and crafts represented in the Exhibition",
        "in being comparatively modern.",
        "For although the Chinese took impressions from wood blocks engraved in relief for centuries before the woodcutters of the Netherlands, by a similar process",
        "produced the block books, which were the immediate predecessors of the true printed book,"
    ]
    data_list = list(text_to_sequence(test, ["english_cleaners"]) for test in tests)

    return data_list


@hydra.main(config_path="config", config_name="config")
@torch.no_grad()
def main(config: DictConfig):
    model = hydra.utils.instantiate(config.model)
    # todo: add to config
    checkpoint_path = to_absolute_path("outputs/2023-11-19/21-44-45/output/checkpoint-6000/model.safetensors")
    checkpoint_path = to_absolute_path(config.checkpoint_path)
    state_dict = load_file(checkpoint_path, device="cpu")
    model.load_state_dict(state_dict)
    model = model.cuda()
    model.eval()

    waveglow = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_waveglow', model_math='fp32')
    waveglow = waveglow.remove_weightnorm(waveglow)
    waveglow = waveglow.to('cuda')
    waveglow.eval()

    data_list = get_data()
    for speed in [0.8, 1.0, 1.2]:

        os.makedirs("results", exist_ok=True)
        for i , text in enumerate(data_list):
            mel, mel_cuda = synthesis(model, text, alpha=speed)
            audio = waveglow.infer(mel_cuda)
            audio_numpy = audio[0].data.cpu().numpy()

            rate = 22050
            write(f"results/audio_{i}_speed={speed}.wav", rate, audio_numpy)


if __name__ == '__main__':
    main()