# Text 2 Speech 2 - Vocoder (HiFi-GAN)


Check WANDB report with generated samples [here](https://api.wandb.ai/links/idsedykh/res9n5ln).

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

Download data `bash download_data.sh`

run `python train.py`

For generation please check [run_synthesis.slurm]() script. Checkpoint may be downloaded [here](https://www.youtube.com/watch?v=dQw4w9WgXcQ) or just use the one in the current repo at path `checkpoints/step-190000`.

something like this should work:
```bash
python synthesise.py \
    +test_audio_dir=test_data/audios \
    +checkpoint_dir=$CHECKPOINT_DIR
```
