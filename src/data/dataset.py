import torch
import torch.utils.data
import numpy as np
import os
import cv2

class DatasetCorine(torch.utils.data.Dataset):
    def __init__(self, 
                 corine_images_path, 
                 corine_labels_path, 
                 augmentation=None,
                 image_mean = [0.485, 0.456, 0.406, 0.432],
                 image_std = [0.229, 0.224, 0.225, 0.228]):

        self.ids = sorted(os.listdir(corine_images_path)) 
        self.images = [os.path.join(corine_images_path, image_id) for image_id in self.ids]
        self.labels = [os.path.join(corine_labels_path, label_id) for label_id in self.ids]
        self.augmentation = augmentation
        self.image_mean = image_mean
        self.image_std = image_std

    def __getitem__(self, index):
        
        image_path = self.images[index]
        label_path = self.labels[index]

        image = cv2.imread(image_path, -1)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        label = np.load(label_path)
        
        if self.augmentation:
            sample = self.augmentation(image=image, mask=label)
            image, label = sample['image'], sample['mask']
        
        # Image normalization
        image = image / ((2**16)-1)
        image = (image - self.image_mean) / self.image_std
        
        image = np.transpose(image, (2, 0, 1)) # (H, W, C) -> (C, H, W)
        
        image = torch.from_numpy(image).float()
        label = torch.from_numpy(label).long()
    
        data = {
            "img" : image,
            "label" : label,
            "name" : image_path
        }

        return data

    def __len__(self):
        return len(self.ids)