# CKRA-MedVQA

This is the implementation of Beyond Static Knowledge: Dynamic Context-Aware Cross-Modal Contrastive Learning for Medical Visual Question Answering.

## Abstract

![Overall_framework](Overall_framework.svg)

## Requirements
Run the following command to install the required packages:
```bash
conda env create -f environment.yaml # method 1
pip install -r requirements.txt # method 2
```

## Preparation

```bash
├── checkpoints
├── data
│   ├── vqa_medvqa_2019_test.arrow
│   ├── ......
├── download
│   ├── checkpoints
│   ├── biobert_v1.1
│   ├── pretrained
│   │   ├── m3ae.ckpt
│   ├── roberta-base
├── m3ae
├── prepro
├── run_scripts
```

### Dataset

Please follow [here](https://github.com/zhjohnchan/M3AE?tab=readme-ov-file#1-dataset-preparation-1) and only use the `SLAKE and VQA-RAD datasets`.

### Pretrained

Download the [m3ae pretrained weight](https://drive.google.com/drive/folders/1b3_kiSHH8khOQaa7pPiX_ZQnUIBxeWWn) and put it in the `download/pretrained`.

### roberta-base

Download the [roberta-base](https://drive.google.com/drive/folders/1ouRx5ZAi98LuS6QyT3hHim9Uh7R1YY1H) and put it in the `download/roberta-base`.

### BioBert

Download the [BioBert](https://huggingface.co/dmis-lab/biobert-v1.1/tree/main) and put it in the `download/biobert_v1.1`.

### Checkpoints

Download the [checkpoints](https://1drv.ms/f/c/0ef3f7692d30fc19/En6cIAzp7r1Iseb-3y1vyw8BF-_NjnusZUB-Dp2nYI3ZGA?e=DbGU3Y) we trained and put it in the `download/checkpoints`.


## Train & Test

```bash
# cd this file 
bash run_scripts/ckra_train.sh
# cd this file
bash run_scripts/ckra_test.sh
```

## Citations
```angular2
@inproceedings{chen2022m3ae,
  title={Multi-Modal Masked Autoencoders for Medical Vision-and-Language Pre-Training},
  author={Chen, Zhihong and Du, Yuhao and Hu, Jinpeng and Liu, Yang and Li, Guanbin and Wan, Xiang and Chang, Tsung-Hui},
  booktitle={International Conference on Medical Image Computing and Computer-Assisted Intervention},
  year={2022},
  organization={Springer}
}
```
