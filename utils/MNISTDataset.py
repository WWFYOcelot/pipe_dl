import torch

import numpy as np

# Create a categorical label from y
# i.e., if y = 5 and num_classes = 10, this function returns [0, 0, 0, 0, 0, 1, 0, 0, 0, 0]
def to_categorical(y, num_classes):
    truth = np.zeros(num_classes)
    truth[y] = 1
    return truth
    
class MNISTDataset(torch.utils.data.Dataset):
    def __init__(self, images, labels):
        self.images = images
        self.labels = labels
    
    # The data loader class calls len() on our dataset to know how many samples there are
    def __len__(self):
        return len(self.images)
    
    # The data loader calls getitem() several times during a batch to load the batch for AI processing
    def __getitem__(self, idx):
        # Normalize the input image to 0-1
        X = self.images[idx].astype(np.float32) / 255.0
        
        # Convert the label to its categorical version
        y = to_categorical(self.labels[idx], 10).astype(np.float32)
        
        # PyTorch expects the input data to be of shape (batch_size, channels, height, width)
        # The (batch) part is handled by the data loader, but our images don't have any channels (depth)
        # expand_dims lets us convert a (28, 28) image to (1, 28, 28)
        X = np.expand_dims(X, axis=0)
        
        # A tensor is a ND array with additional information, such as known shape, datatype, etc.
        # Convert our image and label to torch tensors and return
        return torch.tensor(X), torch.tensor(y)
