import os
import shutil
from pathlib import Path

import numpy as np
import pyworld as pw
import torchaudio
import librosa
from scipy.interpolate import interp1d
from tqdm import tqdm
from fire import Fire


# modified from
# https://github.com/ming024/FastSpeech2/blob/master/preprocessor/preprocessor.py


def main(data_path: str = "data", out_path: str | None = None):
    data_dir = Path(data_path)
    mel_dir = data_dir / "mels"
    wav_dir = data_dir / "LJSpeech-1.1" / "wavs"
    if out_path is None:
        out_dir = data_dir / "pitch"
    else:
        out_dir = Path(out_path)
    out_dir.mkdir(exist_ok=True, parents=True)

    min_pitch = float("inf")
    max_pitch = -float("inf")

    for wav_fpath, mel_fpath in tqdm(
        zip(sorted(wav_dir.iterdir()), sorted(mel_dir.iterdir()))
    ):
        mel = np.load(mel_fpath)

        # wav, sr = librosa.load(wav_fpath)
        wav, sr = torchaudio.load(wav_fpath)
        wav = wav.squeeze(0).numpy()

        wav = wav.astype(np.float64)
        assert wav.ndim == 1  # for mono audio

        hop_length = wav.shape[0] / mel.shape[0]
        frame_period = hop_length / sr * 1000

        pitch, t = pw.dio(wav, sr, frame_period=frame_period)
        pitch = pw.stonemask(wav, pitch, t, sr)

        # no need for alignments
        pitch = pitch[: mel.shape[0]]

        nonzero_ids = np.nonzero(pitch)[0]
        # print(f"{nonzero_ids.shape=}")

        interp_fn = interp1d(
            nonzero_ids,
            pitch[nonzero_ids],
            fill_value=(pitch[nonzero_ids][0], pitch[nonzero_ids][-1]),
            bounds_error=False,
        )

        pitch = interp_fn(np.arange(pitch.shape[0]))

        np.save(out_dir / mel_fpath.name.replace("mel", "pitch"), pitch)

        min_pitch = min(min_pitch, np.min(pitch))
        max_pitch = max(max_pitch, np.max(pitch))

    print(min_pitch, max_pitch)


if __name__ == "__main__":
    Fire(main)
