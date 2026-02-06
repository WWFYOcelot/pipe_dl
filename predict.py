import torch
import torchvision

import numpy as np
import matplotlib.pyplot as plt
import pickle
import click
import importlib
import os

from utils.MNISTDataset import MNISTDataset

def import_architecture(architecture_name):
    architecture = getattr(importlib.import_module(f"architectures.{architecture_name}.architecture"), "Network")
    return architecture

@click.command(context_settings=dict(max_content_width=800))
@click.option('-m', "--model_name", required=True, help="Model architecture to train.")
@click.option('-s', "--save_figures", required=False, is_flag=True, help="Whether to save pngs of the predictions.")
def main(
    model_name,
    save_figures
):
    device = torch.device("cuda")

    #Load the test dataset
    metadata = pickle.load(open(f"architectures/{model_name}/training_metadata.pickle", "rb"))
    X_test = metadata['X_test']
    y_test = metadata['y_test']
    
    test_loader = DataLoader(MNISTDataset(X_test, y_test), batch_size=1, shuffle=False)
    
    model = import_architecture(model_name)().to(device)
    model.load_state_dict(torch.load(f"architectures/{model_name}/model.pth", map_location=device))
    model.eval()
    
    predictions = []
    ground_truths = []
    os.makedirs(f"architectures/{model_name}/predictions", exist_ok=True)
    for batch_idx, (batch_X, batch_y) in enumerate(test_loader):
        batch_X, batch_y = batch_X.to(device), batch_y.to(device)
        prediction = model(batch_X)
        
        predictions.append(prediction.argmax(dim=1).item())
        ground_truths.append(batch_y.argmax(dim=1).item())
        
        if save_figures:
            img = batch_X.squeeze().detach().cpu().numpy()
            plt.imshow(img, cmap="gray")
            plt.title(f"Label: {ground_truths[-1]} | Prediction: {predictions[-1]}")
            plt.savefig(f"architectures/{model_name}/predictions/sample_{batch_idx:.2f}.png")
            plt.cla()
        
    test_accuracy = np.mean(np.array(predictions) == np.array(ground_truths))
    print(f"Testing dataset accuracy: {test_accuracy}")

if __name__ == "__main__":
    main()