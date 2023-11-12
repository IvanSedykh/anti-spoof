# Source Separation


Check WANDB report [here](https://wandb.ai/idsedykh/ss_dla/reports/source-separation--Vmlldzo1OTQ1NzMy?accessToken=lqxcuksw2cv02q8vov12cp4aqowq516dc4967ljplqjx73ecnrf37tnafe8pev65).

Scores on my validation set: 
```
'eval_SI_SDR': 11.45,
'eval_PESQ': 2.05
```

Scores on the public test set: 
```
'test_SI_SDR': 10.48,
'test_PESQ': 1.97
```

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


run `python train.py -c hw_asr/configs/spex_1.json`

To compute metrics:

Download the checkpoint, config [here](https://disk.yandex.ru/d/LJjUXm1ue_i2ug).

```bash
python test.py \
    -r default_test_model/checkpoint-82000 \
    -c default_test_model/config.json \
    -t test_dir_path
```
