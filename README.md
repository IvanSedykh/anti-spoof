# Text 2 Speech (FastSpeech2)


Check WANDB report with generated samples [here](https://wandb.ai/idsedykh/tts_dla/reports/FastSpeech2--Vmlldzo2MDM2ODg4?accessToken=ls1iem1dgdfd6633dm9c9yo7m9sb70uvr8gj73apd56xugdaw6n0gi54vrxanptj).

MOS score is 5/5.  
BONUS: implemented Hydra config.

## Installation guide

1. install conda, create new env
2. install torch
   ``` shell
   conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
   ```
3. install other stuff

   ```shell
   pip install -r ./requirements.txt
   ```
4. pray and hope


## Reproduction guide

Dowbload data `bash download_data.sh`

run `python train.py --config-name config_fastspeech2`

For generation check [run_synthesis.slurm]() script. Checkpoint may be downloaded [here](https://www.youtube.com/watch?v=dQw4w9WgXcQ) or [here](https://disk.yandex.ru/d/tOfUBdgKYCWRxQ).
