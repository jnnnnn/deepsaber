#!/bin/bash

#export PYTHONPATH="/home_directory/.local/lib/python3.5/site-packages/"

py=/usr/bin/python3
#py=/media/usr/bin/python3
dataset=general_beat_saber
model=wavenet
layers=5
blocks=3
exp=block_placement_test
num_windows=10

$py train.py --data_dir=../AugData --dataset_name=$dataset --model=$model --batch_size=1 --output_length=1 --num_windows=$num_windows --nepoch=500 --nepoch_decay=500 --layers=$layers --blocks=$blocks \
    --print_freq=1 --experiment_name=$exp --save_by_iter --save_latest_freq=100 \
    --val_epoch_freq=0 \
    --feature_name=mel \
    --feature_size=100 \
    --concat_outputs \
    --input_channels=$((100+1+4)) \
    --num_classes=$((1+4)) \
    --extra_output \
    --workers=4 \
    --level_diff=Expert \
    --reduced_state \
    --binarized \
    --gpu_ids=0 \
    #--dilation_channels=512 \
    #--residual_channels=256 \
    #--skip_channels=256 \
    #--end_channels=512 \
    #--continue_train \
    #--load_iter=68000 \
    #--load \
    # --gpu_ids=0,1,2,3,4,5,6,7 \

# would be nice to find a better way to look_ahead than the time shifts tbh :P
