# train dataset
echo "Building train"
python mix.py \
    --data_dir ../asr/data/datasets/librispeech/train-clean-100/ \
    --out_folder data/datasets/librispeech/train-clean-100/ \
    --n_files 25000 \
    --num_speakers 100 \
    --snr_levels [0] \
    --test False \
    --num_workers 8


# test dataset
echo "Building test"
python mix.py \
    --data_dir ../asr/data/datasets/librispeech/dev-clean/ \
    --out_folder data/datasets/librispeech/dev-clean/ \
    --n_files 3000 \
    --num_speakers 100 \
    --snr_levels [0] \
    --test False \
    --num_workers 8
