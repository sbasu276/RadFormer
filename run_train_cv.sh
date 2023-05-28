#!/bin/bash

for i in {0..9}
do
	python train_cv.py \
		--train_list="data/cls_split/train_$i.txt" \
		--val_list="data/cls_split/val_$i.txt" \
		--save_name="radformer" \
		--lr=0.003 \
		--optim="sgd" \
		--batch_size=32 \
		--global_weight=0.6 \
		--fusion_weight=0.1 \
		--local_weight=0.3 \
		--epochs=60 \
		--pretrain \
		--load_local \
		--num_layers=4 \
		--save_dir="$1/fold-$i"
done
