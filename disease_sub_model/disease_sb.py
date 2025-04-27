import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
from torchvision import datasets, transforms, models
from torch.utils.data.sampler import SubsetRandomSampler
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary

from datetime import datetime
from logging import Logger

logger = Logger("root")

## Import dataset
transform = transforms.Compose(
    [transforms.Resize(255), transforms.CenterCrop(224), transforms.ToTensor()]
)

dataset = datasets.ImageFolder("Dataset", transform=transform)
indices = list(range(len(dataset)))
split = int(np.floor(0.85 * len(dataset)))
validation = int(np.floor(0.70 * split))

logger.info("0: validation_size: {}, split_size: {}, dataset_size: {}"
            , validation, split, len(dataset))

logger.info(f"length of train size: {validation}")
logger.info(f"length of validation size: {split - validation}")
logger.info(f"length of test size: {len(dataset) - validation}")

np.random.shuffle(indices)


## Model creation
train_indices, validation_indices, test_indices = (
    indices[:validation],
    indices[validation:split],
    indices[split:],
)

train_sampler = SubsetRandomSampler(train_indices)
validation_sampler = SubsetRandomSampler(validation_indices)
test_sampler = SubsetRandomSampler(test_indices)

targets_size = len(dataset.class_to_idx)

class CNN(nn.Module):
    def __init(self, K):
        super(CNN, self).__init__()

        channels = [3, 32, 64, 128, 256]  # input and output channels for each block
        layers = []

        for i in range(len(channels) - 1):
            in_ch = channels[i]
            out_ch = channels[i + 1]

            # First conv
            layers.append(nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1))
            layers.append(nn.ReLU())
            layers.append(nn.BatchNorm2d(out_ch))

            # Second conv
            layers.append(nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1))
            layers.append(nn.ReLU())
            layers.append(nn.BatchNorm2d(out_ch))

            # Max pooling after each block
            layers.append(nn.MaxPool2d(2))

        self.conv_layers = nn.Sequential(
            *layers
        )

        self.dense_layers = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(50176, 1024),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(1024, K),
        )
    
    def forward(self, X):
        out = self.conv_layers(X)

        # Flatten
        out = out.view(-1, 50176)

        # Fully connected
        out = self.dense_layers(out)

        return out
    
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.warn(f"using device: {device}")

model = CNN(targets_size)
mod = model.to(device=device)
logger.debug("CNN formed: {}", mod)

mod_sum = summary(model, (3, 224, 224))
logger.debug("Model Summary: \n", mod_sum)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters())


## Batch Gradient Descent
def batch_gd(model, criterion, train_loader, validation_loader, epochs):
    train_losses = np.zeros(epochs)
    validation_losses = np.zeros(epochs)

    for epoch in range(epochs):
        t0 = datetime.now()
        train_loss = []
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()
            output = model(inputs)
            
            loss = criterion(output, targets)
            train_loss.append(loss.item())
            loss.backward()
            optimizer.step()
        
        mean_train_loss = np.mean(train_loss)

        validation_loss = []
        for inputs, targets in validation_loader:
            inputs, targets = inputs.to(device), targets.to(device)

            output = model(inputs)
            loss = criterion(output, targets)
            validation_loss.append(loss.item())
        
        mean_validation_loss = np.mean(validation_loss)

        train_losses[epoch] = mean_train_loss
        validation_losses[epoch] = mean_validation_loss

        dt = datetime.now() - t0
        logger.info(
            f"Epoch: {epoch+1}/{epochs} \tTraining Loss: {train_loss:.3f} \tValidation Loss: {validation_loss:.3f} \tDuration: {dt}"
        )
    
    return train_losses, validation_losses

batch_size = 64
train_loader = torch.utils.data.DataLoader(
    dataset, batch_size, sampler=train_sampler
)
test_loader = torch.utils.data.DataLoader(
    dataset, batch_size, sampler=test_sampler
)
validation_loader = torch.utils.data.DataLoader(
    dataset, batch_size, sampler=validation_sampler
)

train_losses, validation_losses = batch_gd(
    model, criterion, train_loader, validation_loader, 5
)

# save the model
torch.save(model.state_dict(), 'plant_disease_model.pt')

# loading th model
targets_size = 39
model
    
    