import torch
from torch.utils.data import DataLoader
import torchvision

import numpy as np
import matplotlib.pyplot as plt
import pickle
import click
import importlib
import cv2

from utils.MNISTDataset import MNISTDataset

# importlib provides a direct mechanism to import a file/class based on a string
# I use it to dynamically load the model architecture based on command line arguments
# This lets you define multiple different models (in architectures/) and load them easily
def import_architecture(architecture_name):
    architecture = getattr(importlib.import_module(f"architectures.{architecture_name}.architecture"), "Network")
    return architecture

def feedforward_batch(model, device, data_loader, optimizer, is_train=True):
    # Some parts of the model behave differently during training
    if is_train:
        model.train()
    else:
        #   Since we don't want to train when processing the validation data, 
        #   put the model in evaluation mode
        model.eval()
   
    epoch_correct = 0
    total_samples = 0
    epoch_losses = []
    for batch_idx, (batch_X, batch_y) in enumerate(data_loader):
        print(f"\tBatch {batch_idx + 1} / {len(data_loader)}", end=" | ")
        
        batch_X, batch_y = batch_X.to(device), batch_y.to(device)
        
        # zero_grad() ensures the optimizer isn't using the last 
        # batch's information on the current batch
        if is_train:
            optimizer.zero_grad()
        
        # Pass all the samples to the model for prediction
        predictions = model(batch_X)
        
        # The model outputs 10 numbers (probability per class)
        # The actual prediction is the largest number across the array
        pred_labels = predictions.argmax(dim=1)
        true_labels = batch_y.argmax(dim=1) # Apply this to batch_y too, undoes the to_categorical call
        
        # Add the correct guesses and total samples
        epoch_correct += (pred_labels == true_labels).sum().item()
        total_samples += len(batch_X)
        
        #Compute the error between predictions and batch_y
        loss = model.loss(predictions, batch_y)
        
        # If we're training, call loss.backward()
        # This is the actual backpropagation step
        if is_train:
            loss.backward() # Compute the adjustments for each weight
            optimizer.step() # Apply them to the network
        
        epoch_losses.append(loss.item())
        print(f"loss={np.mean(epoch_losses)}", end="           \r")
   
    # Compute the accuracy on the dataset
    epoch_accuracy = (epoch_correct / total_samples) * 100
    
    print()
    print(f"\t\tAccuracy: {epoch_accuracy}")
    print()
    
    # Return the mean loss and accuracy of the model this epoch
    return np.mean(epoch_losses), epoch_accuracy

def create_binary_dataset(pos_img, neg_img, n):
    assert n % 2 == 0, "n must be even so it can be evenly split."

    half = n // 2

    # Duplicate images
    X = np.array([pos_img] * half + [neg_img] * half)

    # Labels: positive = 1, negative = 0
    y = np.array([1] * half + [0] * half)

    # Shuffle while preserving pairing
    indices = np.random.permutation(n)
    X = X[indices]
    y = y[indices]

    return X, y

# Click provides a convenient interface for command line arguments
@click.command(context_settings=dict(max_content_width=800))
@click.option('-m', "--model_name", required=True, help="Model architecture to train.")
@click.option('-s', "--samples", required=False, help="Max number of samples to use.")
@click.option('-e', "--epochs", required=False, help="Number of epochs to train.")
def main(
    model_name,
    samples,
    epochs
):
    #If you have CUDA installed and working, it's faster to train 
    #   on GPU, so you can set the device to "cuda"
    device = torch.device("cpu")

    pos_image = cv2.imread("defect.png", cv2.IMREAD_GRAYSCALE)
    neg_image = cv2.imread("nodefect.png", cv2.IMREAD_GRAYSCALE)

    X, y = create_binary_dataset(pos_image, neg_image, n=1000)
        
    '''
    #Graph two samples for reference
    fig, ax = plt.subplots(1, 2)
    ax[0].imshow(X[0], cmap='gray')
    ax[1].imshow(X[1], cmap='gray')
    ax[0].set_title("Label: " + str(y[0]))
    ax[1].set_title("Label: " + str(y[1]))
    plt.show()
    exit()
    '''
    # Whether I want to debug or produce a less performant model, this allows me to cut down on the dataset size
    if samples:
        samples = int(samples)
        X = X[:samples]
        y = y[:samples]

    #For this example, we'll use 70% of the samples for training, 15% for validation, and 15% for testing
    n_samples = len(X)
    
    X_train = X[: int(0.7 * n_samples)]
    y_train = y[: int(0.7 * n_samples)]
    
    X_val = X[int(0.7 * n_samples): int(0.85 * n_samples)]
    y_val = y[int(0.7 * n_samples): int(0.85 * n_samples)]
    
    X_test = X[int(0.85 * n_samples):]
    y_test = y[int(0.85 * n_samples):]
    
    print(f"Train: {len(X_train)} | {len(y_train)}")
    print(f"Val: {len(X_val)} | {len(y_val)}")
    print(f"Test: {len(X_test)} | {len(y_test)}")
    
    # Data Loaders let us avoid loading the entire dataset in RAM all at once
    #   Using the data loader, a batch of samples is loaded on demand and the memory is freed after its used
    #   This sacrifices speed (more file i/o) for memory
    train_loader = DataLoader(MNISTDataset(X_train, y_train), batch_size=8, shuffle=True)
    val_loader = DataLoader(MNISTDataset(X_val, y_val), batch_size=8, shuffle=False)
    
    model = import_architecture(model_name)().to(device)

    # Controls how backpropagation occurs
    optimizer = torch.optim.Adam(model.parameters())
    
    # Create a dictionary to save information from this training session
    history = {
        'train_loss': [],
        'train_accuracy': [],
        'val_loss': [],
        'val_accuracy': [],
    }
    
    if epochs:
        epochs = int(epochs)
    else:
        # Default epochs to 10 unless otherwise specified
        epochs = 10

    for epoch in range(1, epochs + 1):
        print(f"Epoch {epoch}")
        train_loss, train_accuracy = feedforward_batch(model, device, train_loader, optimizer, is_train=True)
        val_loss, val_accuracy = feedforward_batch(model, device, val_loader, optimizer=None, is_train=False)
        
        # Save epoch results to the history dict
        history['train_loss'].append(train_loss)
        history['train_accuracy'].append(train_accuracy)
        history['val_loss'].append(val_loss)
        history['val_accuracy'].append(val_accuracy)
    
    # Save the model
    # This lets us run predictions multiple times without needing to train a new model
    torch.save(model.state_dict(), f"architectures/{model_name}/model.pth")
    
    history["X_test"] = X_test
    history["y_test"] = y_test

    pickle.dump(history, open(f"architectures/{model_name}/training_history.pickle", "wb"))

    # Graph the epoch losses and save to disk
    plt.plot(range(0, epochs), history['train_loss'], color='blue', label='train_loss')
    plt.plot(range(0, epochs), history['val_loss'], color='red', label='val_loss')
    plt.legend()
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss Curves")
    plt.savefig(f"architectures/{model_name}/loss_curves.png")
    
    plt.cla()
    
    # Graph the epoch accuracies and save to disk
    plt.plot(range(0, epochs), history['train_accuracy'], color='blue', label=f"train_accuracy [max={np.max(history['train_accuracy']):.2f}]")
    plt.plot(range(0, epochs), history['val_accuracy'], color='red', label=f"val_accuracy [max={np.max(history['val_accuracy']):.2f}]")
    plt.legend()
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Epoch Accuracy")
    plt.savefig(f"architectures/{model_name}/epoch_accuracy.png")

if __name__ == "__main__":
    main()