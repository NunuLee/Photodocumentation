from torch.utils import data
import os
from PIL import Image
import logging
import pandas as pd
from torchvision import transforms as T

data_path = '/root/dataset/APT/'

class APTDataset(data.Dataset):
    def __init__(self, image_size, split, root):
        
        self.annotations = pd.read_csv(root + f'{split}.csv', header=None)
        self.basedir = root
        self.data_path = data_path
        self.image_size = image_size
        self.split = split
        self.transform_input = T.Compose(
            [
                T.ToTensor(), 
                T.Resize(size=(image_size,image_size)), 
                T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]), 
            ]
        )
    def __len__(self):
        """__len__"""
        return len(self.annotations)

    def __getitem__(self, index):
        out = dict()

        path_img = os.path.join(self.data_path, self.annotations.iloc[index, 0])
        landmark = int(self.annotations.iloc[index, 1])
        landmark2 = int(self.annotations.iloc[index, 2])
  
        logging.getLogger('PIL').setLevel(logging.WARNING)
        image = Image.open(path_img).convert('RGB')
        image = self.transform_input(image)

        out["image"] = image
        out["label"] = landmark
        out["label2"] = landmark2

        return out