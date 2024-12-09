This repository contatins the implementation for the paper:
**Evaluation of an Artificial Intelligence-Based System for Real-Time High-Quality Photodocumentation during Esophagogastroduodenoscopy**
Authors : Byeong Yun Ahn, Junwoo Lee, Jeonga Seol, Ji Yoon Kim, Raymond Kim, and Hyunsoo Chung
[Link to the Paper]()

## 01. Code Structure
`main.py` : Main script for training and evalutation
`data/` : Contains data preprocessing scripts
`core/` : Contains model training and evalutation
`model/` : Contains model architectures
`utils/` : Helper functions and utilities

## 02. Prerequisites
- Python >= 3.8
- PyTorch >= 2.1.0
- Install dependencies:
```bash
pip install -r requirements.txt 
```

## 03. Model Training 
To train the model
```bash
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node 2 --master_port 1234 main.py --model-type "swin_b" --epochs 300 --batch-size 128 --learning-rate 1e-5 --num_workers 8
```
