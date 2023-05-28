#!/bin/bash

python train_single.py \
                --train_list="data/cls_split/train.txt" \
                --val_list="data/test.txt" \
                --save_name="radformer" \
                --lr=0.003 \
                --optim="sgd" \
                --batch_size=32 \
                --global_weight=0.6 \
                --fusion_weight=0.1 \
                --local_weight=0.3 \
                --epochs=100 \
                --pretrain \
                --load_local \
                --num_layers=4 \
                --save_dir="$1"
