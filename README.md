## RadFormer

This is the official implementation for the paper "RadFormer: Transformers with global–local attention for interpretable and accurate Gallbladder Cancer detection" 
[https://doi.org/10.1016/j.media.2022.102676](https://doi.org/10.1016/j.media.2022.102676)

### Dataset
To get the Gallbladder Cancer Ultrasound Dataset (GBCU) follow the instructions [here](https://gbc-iitd.github.io/data/gbcu).

### Model Zoo
1. Plesse download the zip file containing models from [this link](https://drive.google.com/file/d/151pPVWQBR5M3RdZW4a616y9VVHl0uZBc/view).
2. Unzip and save the directory `model_weights`

### Running the Evaluation Code
1. Unzip the model weights, and make sure to keep them in the `model_weights` directory.
2. Run the following commands.
```
bash run_test.sh
```

### Running the Training Code
1. Unzip the model weights, and make sure to keep them in the `model_weights` directory.
2. Run the following commands.
```
bash run_train.sh
```
For running 10-fold cross-validation, use:
```
bash run_train_cv.sh
```

### Finetuning on Custom Dataset
1. Format the dataset as the GBCU.
2. Use the `run_train.sh` scipt, and modify the arguments (`--img_dir`, `--train_list`, `--val_list`) according to the dataset.

### Citation
```
@article{basu2023radformer,
  title={RadFormer: Transformers with global--local attention for interpretable and accurate Gallbladder Cancer detection},
  author={Basu, Soumen and Gupta, Mayank and Rana, Pratyaksha and Gupta, Pankaj and Arora, Chetan},
  journal={Medical Image Analysis},
  volume={83},
  pages={102676},
  year={2023},
  publisher={Elsevier}
}
```
