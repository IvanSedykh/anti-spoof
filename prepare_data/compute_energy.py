import os
import shutil
from pathlib import Path

import numpy as np
from fire import Fire
from tqdm import tqdm


def main(data_path: str = "data", out_path: str | None = None):
    data_dir = Path(data_path)
    mel_dir = data_dir / "mels"
    if out_path is None:
        out_dir = data_dir / "energy"
    else:
        out_dir = Path(out_path)
    out_dir.mkdir(exist_ok=True, parents=True)

    min_energy = float("inf")
    max_energy = -float("inf")

    for fpath in tqdm(mel_dir.iterdir()):
        mel = np.load(fpath)
        # energy is mel frame norm
        energy = np.linalg.norm(mel, axis=-1)
        np.save(out_dir / fpath.name.replace("mel", "energy"), energy)

        min_energy = min(min_energy, np.min(energy))
        max_energy = max(max_energy, np.max(energy))

    print(min_energy, max_energy)


if __name__ == "__main__":
    Fire(main)
