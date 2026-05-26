#!/bin/bash
set -e

echo "Starting AMC ECG pruning..."
# bash scripts/run_search_mobilenetv2.sh
# source ~/anaconda3/bin/activate your_env

TRAIN_PATH="./data/split_csvs/In_Out_distribution/train_stratified.csv"
VAL_PATH="./data/split_csvs/In_Out_distribution/val_stratified.csv"
CKPT_PATH="./experiments/mobilenetV2/train_mobilenetv2/In_Out_distribution.pth"

LABELS=(
426783006
426177001
164934002
427393009
713426002
427084000
59118001
164889003
59931005
47665007
445118002
39732003
164890007
164909002
270492004
251146004
284470004
)

PRESERVE_RATIO=0.5
LBOUND=0.05
RBOUND=1.0
MAX_EPISODES=1000

python mobilenetv2_search.py \
    --train_path "${TRAIN_PATH}" \
    --val_path "${VAL_PATH}" \
    --ckpt_path "${CKPT_PATH}" \
    --preserve_ratio "${PRESERVE_RATIO}" \
    --max_episodes "${MAX_EPISODES}" \
    --lbound "${LBOUND}" \
    --rbound "${RBOUND}" \
    --labels "${LABELS[@]}" \
    --hidden1 300 \
    --hidden2 300 \
    --lr_c 1e-3 \
    --lr_a 1e-4 \
    --warmup 100 \
    --discount 1.0 \
    --bsize 16 \
    --rmsize 100 \
    --window_length 1 \
    --tau 0.01 \
    --init_delta 0.5 \
    --delta_decay 0.95 \
    --epsilon 50000 \
    --n_calibration_batches 5 \
    --n_points_per_layer 50 \
    --channel_round 1 \
    --reward_alpha 0.5 \
    --train_batch_size 32 \
    --val_batch_size 1 \
    --num_workers 0

echo "SEARCH FINISHED"
