#!/usr/bin/env bash

# Where the ImageNet2012 TFR is stored to. Replace this with yours
DATA_DIR=../FOODX-251_Dataset/val_TFR

# Where the the checkpoint to evaluate is saved to. Replace this with yours
MODEL_DIR=../assembled-cnn/train/m0/best

python assembled-cnn/mce/eval_robustness.py \
--robustness_type=ce \
--gpu_index=0 \
--num_classes=251 \
--batch_size=256 \
--resnet_size=50 \
--image_size=256 \
--bl_alpha=1 \
--bl_beta=2 \
--resnet_version=2 \
--anti_alias_type=sconv \
--anti_alias_filter_size=3 \
--data_format=channels_first \
--use_sk_block=True \
--label_file=datasets/foodx251.txt \
--data_dir=${DATA_DIR} \
--model_dir=${MODEL_DIR}