import torch
from torch.utils.data import DataLoader
import torchvision

import numpy as np
import matplotlib.pyplot as plt
import pickle
import click
import importlib

from utils.MNISTDataset import MNISTDataset

def import_architecture(architecture_name):
    architecture = getattr(importlib.import_module(f"architectures.{architecture_name}.architecture"), "Network")
    return architecture

def feedforward_batch(model, device, data_loader, optimizer, is_train=True):
    if is_train:
        model.train()
    else:
        model.eval()
   
    epoch_correct = 0
    total_samples = 0
    epoch_losses = []
    for batch_idx, (batch_X, batch_y) in enumerate(data_loader):
        print(f"\tBatch {batch_idx + 1} / {len(data_loader)}", end=" | ")
        
        batch_X, batch_y = batch_X.to(device), batch_y.to(device)
        if is_train:
            optimizer.zero_grad()
        
        predictions = model(batch_X)
        
        pred_labels = predictions.argmax(dim=1)
        true_labels = batch_y.argmax(dim=1)
        
        epoch_correct += (pred_labels == true_labels).sum().item()
        total_samples += len(batch_X)
        
        loss = model.loss(predictions, batch_y)
        if is_train:
            loss.backward()
        
        epoch_losses.append(loss.item())

        print(f"loss={np.mean(epoch_losses)}", end="           \r")
        if is_train:
            optimizer.step()
       
    epoch_accuracy = (epoch_correct / total_samples) * 100
    
    print()
    print(f"\t\tAccuracy: {epoch_accuracy}")
    print()
    
    return np.mean(epoch_losses), epoch_accuracy

@click.command(context_settings=dict(max_content_width=800))
@click.option('-m', "--model_name", required=True, help="Model architecture to train.")
@click.option('-s', "--samples", required=False, help="Max number of samples to use.")
@click.option('-e', "--epochs", required=False, help="Number of epochs to train.")
def main(
    model_name,
    samples,
    epochs
):
    #If you have CUDA installed and working, it's faster to train on GPU.
    #Otherwise, set this to "cpu"
    device = torch.device("cuda")

    #Load the MNIST dataset torchvision
    mnist = torchvision.datasets.MNIST(root="./data/", train=True, download=True)

    X = mnist.data.numpy()
    y = mnist.targets.numpy()
    
    #Shuffle X & y, preserving the 1-1 relationship
    indices = list(range(0, len(X)))
    np.random.shuffle(indices)
    
    X = X[indices]
    y = y[indices]
    
    #Graph two samples for reference
    '''
    fig, ax = plt.subplots(1, 2)
    ax[0].imshow(x_train[0], cmap='gray')
    ax[1].imshow(x_train[1], cmap='gray')
    ax[0].set_title("Label: " + str(y_train[0]))
    ax[1].set_title("Label: " + str(y_train[1]))
    plt.show()
    exit()
    '''

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
        
    train_loader = DataLoader(MNISTDataset(X_train, y_train), batch_size=64, shuffle=True)
    val_loader = DataLoader(MNISTDataset(X_val, y_val), batch_size=64, shuffle=False)
    
    model = import_architecture(model_name)().to(device)
    optimizer = torch.optim.Adam(model.parameters())
    
    history = {
        'train_loss': [],
        'train_accuracy': [],
        'val_loss': [],
        'val_accuracy': [],
    }
    
    if epochs:
        epochs = int(epochs)
    else:
        epochs = 10

    for epoch in range(1, epochs + 1):
        print(f"Epoch {epoch}")
        train_loss, train_accuracy = feedforward_batch(model, device, train_loader, optimizer, is_train=True)
        val_loss, val_accuracy = feedforward_batch(model, device, val_loader, optimizer=None, is_train=False)
        
        history['train_loss'].append(train_loss)
        history['train_accuracy'].append(train_accuracy)
        history['val_loss'].append(val_loss)
        history['val_accuracy'].append(val_accuracy)
        
    torch.save(model.state_dict(), f"architectures/{model_name}/model.pth")
    metadata = {
        "history": history,
        "X_test": X_test,
        "y_test": y_test
    }
    pickle.dump(metadata, open(f"architectures/{model_name}/training_metadata.pickle", "wb"))

    plt.plot(range(0, epochs), history['train_loss'], color='blue', label='train_loss')
    plt.plot(range(0, epochs), history['val_loss'], color='red', label='val_loss')
    plt.legend()
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss Curves")
    plt.savefig(f"architectures/{model_name}/loss_curves.png")
    
    plt.cla()
    
    plt.plot(range(0, epochs), history['train_accuracy'], color='blue', label=f"train_accuracy [max={np.max(history['train_accuracy']):.2f}]")
    plt.plot(range(0, epochs), history['val_accuracy'], color='red', label=f"val_accuracy [max={np.max(history['val_accuracy']):.2f}]")
    plt.legend()
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Epoch Accuracy")
    plt.savefig(f"architectures/{model_name}/epoch_accuracy.png")

if __name__ == "__main__":
    main()