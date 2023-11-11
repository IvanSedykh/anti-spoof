# Source Separation


Check WANDB report [here]().


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


run `python train.py -c hw_asr/configs/`

To compute metrics:

Download the checkpoint, config [here]().

test-other:
```bash
python test.py \
    --batch-size 32 \
    --jobs 8 \
    -c default_test_model/config-other.json \
```


test-clean:
```bash
python test.py \
    --batch-size 32 \
    --jobs 8 \
    -c default_test_model/config-clean.json \
```

I have used a pretty powerful server, so it may fail in case of weaker machine.