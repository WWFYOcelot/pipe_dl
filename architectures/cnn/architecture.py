import torch

class Network(torch.nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        
        # Our network architecture consist of one convolutional layer and two dense layers
        # In PyTorch, the Dense layer is called Linear
        
        self.conv1 = torch.nn.Conv2d(1, 16, (3, 3), stride=(1, 1), padding=1)

        
        self.fc1 = torch.nn.Linear(65536, 256)
        self.fc2 = torch.nn.Linear(256, 10)

        # The 3136 is not arbitrary
        #   Input image:           ( 1, 28, 28)
        #   Output of convolution: (16, 28, 28)
        #   Output of pooling:     (16, 14, 14)
        #       16 x 14 x 14 = 3136, so the input to this first dense layer is 3136 elements
        
        # Define the loss function as binary cross entropy
        self.loss = torch.nn.functional.binary_cross_entropy
        
    # The forward function is the feedforward component of the network
    # Pass an input through each layer and return the network output
    def forward(self, x):
        # Apply the convolutional layer
        x = self.conv1(x)
        x = torch.nn.functional.relu(x) # Activation function, adds nonlinearity to conv output

        # Apply the pooling operation
        x = torch.nn.functional.max_pool2d(x, (2, 2))

        # Flatten the data to pass to the dense layers
        x = torch.flatten(x, start_dim=1)
        
        # Apply the dense layers
        x = self.fc1(x)
        x = torch.nn.functional.relu(x)

        x = self.fc2(x)
        
        # Softmax ensures that the resultant vector sums to 1
        # This allows us to interpret model output as class probabilities
        # Increasing the probability of one class necessarily decreases the others
        x = torch.nn.functional.softmax(x, dim=1) # The first dimension is batch, so we apply softmax to the second dimension
        
        return x
    
    