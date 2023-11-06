python mix.py \
    --data_dir ../asr/data/datasets/librispeech/dev-clean/ \
    --out_folder data/datasets/librispeech/dev-clean/ \
    --n_files 20 \
    --num_speakers 100 \
    --snr_levels [0] \
    --test False \
    --num_workers 4

# todo make full datasets