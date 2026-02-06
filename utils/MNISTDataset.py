import torch

import numpy as np

def to_categorical(y, num_classes):
    return np.eye(num_classes)[y]
    
class MNISTDataset(torch.utils.data.Dataset):
    def __init__(self, images, labels):
        self.images = images
        self.labels = labels
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        X = self.images[idx]
        y = to_categorical(self.labels[idx], 10)
        
        X = np.array(X, dtype=np.float32) / 255.0
        X = np.expand_dims(X, axis=0)
        
        return torch.tensor(X), torch.tensor(y, dtype=torch.float32)
