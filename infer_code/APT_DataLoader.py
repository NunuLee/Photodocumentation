from torch.utils import data
from torchvision import transforms as T
from PIL import Image
import torch 
import numpy as np
import cv2
import logging

class APT_Dataset(data.Dataset):
    def __init__(self, img, image_size):
        self.img = img
        self.image_size = image_size
        self.transform_input4test = T.Compose(
            [
                T.ToTensor(),
                T.Resize((self.image_size, self.image_size), antialias=True),
                T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]), 
            ]
        )

    def __len__(self):
        """__len__"""
        return 1

    def __getitem__(self, index):

        logging.getLogger("PIL").setLevel(logging.WARNING)

        x = Image.fromarray(self.img, "RGB")
        x = self.transform_input4test(x)

        return x
    
def load_data(frame, image_size):
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    dataset = APT_Dataset(frame, image_size)
    dataloader = data.DataLoader(dataset)
    
    return dataloader
