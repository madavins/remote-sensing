import torch
import torch.utils.data
import numpy as np
import os

class DatasetCorine(torch.utils.data.Dataset):
    def __init__(self, corine_images_path, corine_labels_path, augmentation=None):

        self.ids = sorted(os.listdir(corine_images_path)) #returns image names
        self.images = [os.path.join(corine_images_path, image_id) for image_id in self.ids]
        self.labels = [os.path.join(corine_labels_path, label_id) for label_id in self.ids]
        self.augmentation = augmentation

        self.num_examples = len(self.ids)

    def __getitem__(self, index):
        
        
        image = np.load(self.images[index])
        label = np.load(self.labels[index])
        
        image = np.transpose(image, (1, 2, 0))
        
        if self.augmentation:
            sample = self.augmentation(image=image, mask=label)
            image, label = sample['image'], sample['mask']
        
        ################################################################################################################
        # normalize the img (with the mean and std):
        ################################################################################################################
        #image = np.transpose(image, (1, 2, 0))
        image = image / ((2**16)-1)
        mean = np.mean(image, axis=(0,1))
        std = np.std(image, axis=(0,1))
        image = image - mean
        image = image/std
        
        image = image.astype(np.float32)
        image = np.transpose(image, (2, 0, 1))
        
        ################################################################################################################
        # convert numpy -> torch:
        ################################################################################################################
        image = torch.from_numpy(image) # (shape: (4, 512, 512))
        label = torch.from_numpy(label)
    
        
        data = {
            "img" : image,
            "label" : label,
            "mean" : mean,
            "std" : std,
            "name" : self.images[index]
        }
        return data

    def __len__(self):
        return self.num_examples