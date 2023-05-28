#!/bin/bash

python test.py \
	--img_dir="data/imgs" \
	--val_list="data/test.txt" \
	--model_file="model_weights/radformer/radformer.pkl"
