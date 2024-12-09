This repository contatins the implementation for the paper:
**Evaluation of an Artificial Intelligence-Based System for Real-Time High-Quality Photodocumentation during Esophagogastroduodenoscopy**
Authors : Byeong Yun Ahn, Junwoo Lee, Jeonga Seol, Ji Yoon Kim, Raymond Kim, and Hyunsoo Chung
[Link to the Paper]()

## 01. Code Structure
infer_code/  
├── APT_DataLoader.py        # Script for data loading and preprocessing  
├── apt.py                   # Main script for managing the overall inference process  
├── bestshot_selection.py    # Logic for selecting the best shot from the results  
├── model_build.py           # Script for initializing and building the model architecture  
├── model_inference.py       # Script for executing model inference  

## 02. Prerequisites
- Python >= 3.8
Install required libraries:
```bash
pip install -r requirements.txt
```

## 03. Dataset Preparation
- To run the model inference, you need a folder containing **consecutive frames as image files.** 
- The images should meet the following requirements:
    - Supported file formats : `.jpg`, `.jpeg`, `.JPEG`, `.png`
    - Images should be stored in the same folder and will be loaded in the order of their file names.

dataset/  
├── folder_01/  
│   ├── frame_001.jpg  
│   ├── frame_002.jpg  
│   ├── frame_003.jpg  
│   └── ...  

## 04. Usage 
- Use the main script `apt.py` to start the inference process
```bash
python apt.py
```

## 05. Result and validation 
- Validate the inference results based on the criteria outlined in the paper.
- Example outputs will be saved in the `output/` directory.