# importing albumentation librairies
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2

# PyTorch libraries
import torch
import numpy as np
from PIL import Image

import cv2

class MRIDataset(object):
    def __init__(self, imagesPath, labels):
        self.imagesPath = imagesPath
        self.labels = labels

    def __len__(self):
        return len(self.imagesPath)

    def __getitem__(self, idx):
        
        MRI_PIL = Image.open(self.imagesPath[idx])
        MRI_array = np.array(MRI_PIL)
        
        # MRI 1 channel to 3 channels
        MRI_3_channels = cv2.merge((MRI_array,MRI_array,MRI_array))
        
        # Apply transforms
        preprocess = self.preprocessing()
        sample = preprocess(image=MRI_3_channels)       
        img = sample["image"]   
        
        label = torch.tensor(self.labels[idx])
        return img, label

    def preprocessing(self):
        # Preprocessing vgg16 pretrained
        preprocessing = [
            A.Resize(height=256, width=256),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ]
        return A.Compose(preprocessing)
