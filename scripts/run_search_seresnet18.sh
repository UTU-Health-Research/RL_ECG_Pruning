#!/bin/bash


python amc_ddpg_v3.py \
    --train_csv=./data/split_csvs/In_Out_distribution/train_stratified.csv \
    --val_csv=./data/split_csvs/In_Out_distribution/val_stratified.csv \
    --model_path=./experiments/train_stratified/resnet18_In_Out_distribution.pth \
    --labels 426783006 426177001 164934002 427393009 713426002 427084000 59118001 164889003 59931005 47665007 445118002 39732003 164890007 164909002 270492004 251146004 284470004 \
    --preserve_ratio=0.1 \
    --lbound=0.05 \
    --rbound=1.0 \
    --reward= reward \
    --acc_metric=micro_auroc \
    --seed=2027 \
    --n_calibration_batches=60 \
    --channel_round=4 \
    --env_version=v4 \
    --warmup=100 \
    --train_episode=500 \
    --discount=1.0 \
    --init_delta=0.5 \
    --delta_decay=0.995 \
    --delta_min=0.15 \
    --hidden1=300 \
    --hidden2=300 \
    --lr_c=0.001 \
    --lr_a=0.0001 \
    --bsize=64 \
    --tau=0.01 \
    --data_bsize=64 \
    --rmsize=500 \
    --output=./experiments/123

echo "=============================================="
echo "Complete!"
echo "=============================================="
