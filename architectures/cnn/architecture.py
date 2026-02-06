import torch

class Network(torch.nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        self.conv1 = torch.nn.Conv2d(1, 16, (3, 3), stride=(1, 1), padding=1)

        self.fc1 = torch.nn.Linear(3136, 256)
        self.fc2 = torch.nn.Linear(256, 10)

        self.loss = torch.nn.functional.binary_cross_entropy
        
    def forward(self, x):
        x = self.conv1(x)
        x = torch.nn.functional.relu(x)

        x = torch.nn.functional.max_pool2d(x, (2, 2))

        x = torch.flatten(x, start_dim=1)
        
        x = self.fc1(x)
        x = torch.nn.functional.relu(x)

        x = self.fc2(x)
        x = torch.nn.functional.softmax(x, dim=1)
        
        return x
    
    