import torch
import numpy as np
import cv2  # pip install opencv-python

# Create a categorical label from y
def to_categorical(y, num_classes):
    truth = np.zeros(num_classes)
    truth[y] = 1
    return truth

class MNISTDataset(torch.utils.data.Dataset):
    def __init__(self, images, labels, target_size=128):
        self.images = images
        self.labels = labels
        self.target_size = target_size
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img = self.images[idx]

        # Convert to grayscale if image has channels
        if img.ndim == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Resize to 128x128
        img = cv2.resize(img, (self.target_size, self.target_size), interpolation=cv2.INTER_NEAREST_EXACT)

        # Normalize to 0-1
        X = img.astype(np.float32) / 255.0

        # Add channel dimension -> (1, 128, 128)
        X = np.expand_dims(X, axis=0)

        # One-hot label
        y = to_categorical(self.labels[idx], 10).astype(np.float32)

        return torch.tensor(X), torch.tensor(y)