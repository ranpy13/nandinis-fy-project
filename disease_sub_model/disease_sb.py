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

from typing import List, Tuple, Dict, Optional, Union
from PIL import Image
from datetime import datetime
from utils.logger_util import setup_logger

logger = setup_logger(logger_name=__name__)

class DiseaseClassifierConfig:
    """Configuration class for Disease Classifier"""
    def __init__(
        self,
        image_size: int = 224,
        batch_size: int = 64,
        train_split: float = 0.85,
        validation_split: float = 0.70,
        dropout_rate: float = 0.4,
        learning_rate: float = 0.001,
        epochs: int = 5,
        model_save_path: str = 'plant_disease_model.pt',
        disease_info_path: str = 'disease_info.csv'
    ):
        self.image_size = image_size
        self.batch_size = batch_size
        self.train_split = train_split
        self.validation_split = validation_split
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.model_save_path = model_save_path
        self.disease_info_path = disease_info_path

class DiseaseDataset:
    """Class to handle dataset operations"""
    def __init__(self, config: DiseaseClassifierConfig):
        self.config = config
        self.transform = transforms.Compose([
            transforms.Resize(255),
            transforms.CenterCrop(config.image_size),
            transforms.ToTensor()
        ])
        self.dataset = None
        self.train_sampler = None
        self.validation_sampler = None
        self.test_sampler = None

    def load_dataset(self, dataset_path: str) -> None:
        """Load and split the dataset"""
        self.dataset = datasets.ImageFolder(dataset_path, transform=self.transform)
        indices = list(range(len(self.dataset)))
        split = int(np.floor(self.config.train_split * len(self.dataset)))
        validation = int(np.floor(self.config.validation_split * split))

        logger.info(f"Dataset sizes - Validation: {validation}, Split: {split}, Total: {len(self.dataset)}")
        logger.info(f"Train size: {validation}")
        logger.info(f"Validation size: {split - validation}")
        logger.info(f"Test size: {len(self.dataset) - split}")

        np.random.shuffle(indices)
        train_indices, validation_indices, test_indices = (
            indices[:validation],
            indices[validation:split],
            indices[split:]
        )

        self.train_sampler = SubsetRandomSampler(train_indices)
        self.validation_sampler = SubsetRandomSampler(validation_indices)
        self.test_sampler = SubsetRandomSampler(test_indices)

    def get_data_loaders(self) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
        """Get data loaders for train, validation and test sets"""
        train_loader = torch.utils.data.DataLoader(
            self.dataset, 
            batch_size=self.config.batch_size, 
            sampler=self.train_sampler
        )
        validation_loader = torch.utils.data.DataLoader(
            self.dataset, 
            batch_size=self.config.batch_size, 
            sampler=self.validation_sampler
        )
        test_loader = torch.utils.data.DataLoader(
            self.dataset, 
            batch_size=self.config.batch_size, 
            sampler=self.test_sampler
        )
        return train_loader, validation_loader, test_loader

class DiseaseCNN(nn.Module):
    """CNN model for disease classification"""
    def __init__(self, num_classes: int, config: DiseaseClassifierConfig):
        super(DiseaseCNN, self).__init__()
        self.config = config
        
        channels = [3, 32, 64, 128, 256]
        layers = []

        for i in range(len(channels) - 1):
            in_ch = channels[i]
            out_ch = channels[i + 1]

            layers.extend([
                nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.BatchNorm2d(out_ch),
                nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.BatchNorm2d(out_ch),
                nn.MaxPool2d(2)
            ])

        self.conv_layers = nn.Sequential(*layers)
        self.dense_layers = nn.Sequential(
            nn.Dropout(config.dropout_rate),
            nn.Linear(50176, 1024),
            nn.ReLU(),
            nn.Dropout(config.dropout_rate),
            nn.Linear(1024, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_layers(x)
        x = x.view(-1, 50176)
        x = self.dense_layers(x)
        return x

class DiseaseClassifier:
    """Main class for disease classification"""
    def __init__(self, config: Optional[DiseaseClassifierConfig] = None):
        self.config = config or DiseaseClassifierConfig()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.warning(f"Using device: {self.device}")
        
        self.model = None
        self.dataset = DiseaseDataset(self.config)
        self.disease_info = None

    def train(self, dataset_path: str) -> Tuple[np.ndarray, np.ndarray]:
        """Train the model"""
        self.dataset.load_dataset(dataset_path)
        train_loader, validation_loader, _ = self.dataset.get_data_loaders()
        
        num_classes = len(self.dataset.dataset.class_to_idx)
        self.model = DiseaseCNN(num_classes, self.config).to(self.device)
        
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config.learning_rate)

        train_losses = np.zeros(self.config.epochs)
        validation_losses = np.zeros(self.config.epochs)

        for epoch in range(self.config.epochs):
            t0 = datetime.now()
            
            # Training
            self.model.train()
            train_loss = []
            for inputs, targets in train_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                train_loss.append(loss.item())
            
            # Validation
            self.model.eval()
            validation_loss = []
            with torch.no_grad():
                for inputs, targets in validation_loader:
                    inputs, targets = inputs.to(self.device), targets.to(self.device)
                    outputs = self.model(inputs)
                    loss = criterion(outputs, targets)
                    validation_loss.append(loss.item())

            train_losses[epoch] = np.mean(train_loss)
            validation_losses[epoch] = np.mean(validation_loss)

            dt = datetime.now() - t0
            logger.info(
                f"Epoch: {epoch+1}/{self.config.epochs} "
                f"Train Loss: {train_losses[epoch]:.3f} "
                f"Validation Loss: {validation_losses[epoch]:.3f} "
                f"Duration: {dt}"
            )

        return train_losses, validation_losses

    def save_model(self) -> None:
        """Save the trained model"""
        if self.model is None:
            raise ValueError("Model has not been trained yet")
        torch.save(self.model.state_dict(), self.config.model_save_path)

    def load_model(self, num_classes: int) -> None:
        """Load a trained model"""
        self.model = DiseaseCNN(num_classes, self.config)
        self.model.load_state_dict(torch.load(self.config.model_save_path))
        self.model.eval()
        self.model.to(self.device)

    def evaluate(self) -> Dict[str, float]:
        """Evaluate the model on train, validation and test sets"""
        if self.model is None:
            raise ValueError("Model has not been trained or loaded yet")

        _, _, test_loader = self.dataset.get_data_loaders()
        accuracies = {}
        
        for name, loader in [("Train", self.dataset.train_sampler),
                           ("Validation", self.dataset.validation_sampler),
                           ("Test", test_loader)]:
            n_correct = 0
            n_total = 0
            
            self.model.eval()
            with torch.no_grad():
                for inputs, targets in loader:
                    inputs, targets = inputs.to(self.device), targets.to(self.device)
                    outputs = self.model(inputs)
                    _, predictions = torch.max(outputs, 1)
                    n_correct += (predictions == targets).sum().item()
                    n_total += targets.shape[0]
            
            accuracies[name] = n_correct / n_total
            logger.info(f"{name} Accuracy: {accuracies[name]:.3f}")

        return accuracies

    def predict(self, image_path: str) -> str:
        """Predict disease from a single image"""
        if self.model is None:
            raise ValueError("Model has not been trained or loaded yet")
        
        if self.disease_info is None:
            self.disease_info = pd.read_csv(self.config.disease_info_path, encoding="cp1252")

        image = Image.open(image_path)
        image = image.resize((self.config.image_size, self.config.image_size))
        input_data = TF.to_tensor(image)
        input_data = input_data.view((-1, 3, self.config.image_size, self.config.image_size))
        input_data = input_data.to(self.device)

        self.model.eval()
        with torch.no_grad():
            output = self.model(input_data)
            output = output.cpu().numpy()
            index = np.argmax(output)
            
        return self.disease_info["disease_name"][index]

def plot_losses(train_losses: np.ndarray, validation_losses: np.ndarray) -> None:
    """Plot training and validation losses"""
    plt.plot(train_losses, label="train_loss")
    plt.plot(validation_losses, label="validation_loss")
    plt.xlabel('No of Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

# Example usage
if __name__ == "__main__":
    # Create configuration
    config = DiseaseClassifierConfig(
        image_size=224,
        batch_size=64,
        epochs=5
    )

    # Initialize classifier
    classifier = DiseaseClassifier(config)

    # Train the model
    train_losses, validation_losses = classifier.train("Dataset")

    # Save the model
    classifier.save_model()

    # Plot losses
    plot_losses(train_losses, validation_losses)

    # Evaluate the model
    accuracies = classifier.evaluate()

    # Make a prediction
    prediction = classifier.predict("path_to_image.jpg")
    logger.info(f"Predicted disease: {prediction}")
    