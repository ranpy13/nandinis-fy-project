import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F
import torch.nn as nn
import torchvision.transforms.functional as TF
from torchvision import datasets, transforms, models
from torch.utils.data.sampler import SubsetRandomSampler
from torchsummary import summary

from PIL import Image
from datetime import datetime
from utils.logger_util import setup_logger
logger = setup_logger(logger_name= __name__)

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
logger.warning(f"using device: {device}")

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
model = CNN(targets_size)
model.load_state_dict(torch.load("plant_disease_model.pt"))
model.eval()


## Plotting the loss
plt.plot(train_losses, label="train_loss")
plt.plot(validation_losses , label = 'validation_loss')
plt.xlabel('No of Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# accuracy
def accuracy(loader):
    n_correct = 0
    n_total = 0

    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)

        outputs = model(inputs)

        _, predictions = torch.max(outputs, 1)

        n_correct += (predictions == targets).sum().item()
        n_total += targets.shape[0]

    acc = n_correct / n_total
    return acc
    
train_acc = accuracy(train_loader)
test_acc = accuracy(test_loader)
validation_acc = accuracy(validation_loader)

logger.info(f"Train Accuracy : {train_acc}\nTest Accuracy : {test_acc}\nValidation Accuracy : {validation_acc}")


## Single image prediction
transform_index_to_disease = dataset.class_to_idx
transform_index_to_disease = dict(
    [(value, key) for key, value in transform_index_to_disease.items()]
)

data = pd.read_csv("disease_info.csv", encoding="cp1252")

def single_prediction(image_path):
    image = Image.open(image_path)
    image = image.resize((224, 224))

    input_data = TF.to_tensor(image)
    input_data = input_data.view((-1, 3, 224, 224))
    output = model(input_data)
    output = output.detach().numpy()
    
    index = np.argmax(output)
    logger.info("Original : {}", image_path[12:-4])
    pred_csv = data["disease_name"][index]
    logger.info(pred_csv)
    