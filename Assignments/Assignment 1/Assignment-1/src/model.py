import torch
import os
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


class DogCatClassifier(nn.Module):
    def __init__(self):
        super(DogCatClassifier, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding="same")  # 256*256
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)  # 128 * 128

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding="same")
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(64 * 64 * 64, 256)
        self.fc2 = nn.Linear(256, 64)
        self.fc3 = nn.Linear(64, 1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        x = torch.sigmoid(x)
        return x.squeeze(1)


class Trainer:
    def __init__(self, root_path: str):
        # Define transformations for data augmentation and normalization
        self.transform = transforms.Compose(
            [
                transforms.Resize((256, 256)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )
        self.root = root_path
        self.threshold = 0.5
        # Initialize the model, loss function, and optimizer
        self.model = DogCatClassifier()
        self.criterion = nn.BCELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)

    def _load_data(self):
        self.train_images = datasets.ImageFolder(
            root=os.path.join(self.root, "train"),
            transform=self.transform,
        )
        self.val_images = datasets.ImageFolder(
            root=os.path.join(self.root, "val"), transform=self.transform
        )

    def _extract_data(self):
        self.train_loader = DataLoader(self.train_images, batch_size=32, shuffle=True)
        self.val_loader = DataLoader(self.val_images, batch_size=32, shuffle=False)

    def _eval(self):
        self.model.eval()
        total_val_loss = 0
        correct_val_predictions = 0
        total_val_samples = 0

        with torch.no_grad():
            for val_inputs, val_labels in self.val_loader:
                val_outputs = self.model(val_inputs)
                val_loss = self.criterion(
                    val_outputs, val_labels.type(torch.FloatTensor)
                )

                total_val_loss += val_loss.item()
                val_predicted = val_outputs > self.threshold
                correct_val_predictions += (val_predicted == val_labels).sum().item()
                total_val_samples += val_labels.size(0)

        val_loss = total_val_loss / len(self.val_loader)
        val_accuracy = correct_val_predictions / total_val_samples
        return val_loss, val_accuracy

    def _train(self):
        self.model.train()
        total_loss = 0
        correct_predictions = 0
        total_samples = 0

        for inputs, labels in self.train_loader:
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels.type(torch.FloatTensor))
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            predicted = outputs > self.threshold
            correct_predictions += (predicted == labels).sum().item()
            total_samples += labels.size(0)

        epoch_loss = total_loss / len(self.train_loader)
        epoch_accuracy = correct_predictions / total_samples
        return epoch_loss, epoch_accuracy

    def train(self, num_epochs: int = 20):
        self._load_data()
        self._extract_data()

        for epoch in range(num_epochs):
            train_loss, train_accuracy = self._train()
            print(
                f"Training - Epoch [{epoch + 1}/{num_epochs}], Loss: {train_loss:.4f}, Accuracy: {train_accuracy:.4f}"
            )
            val_loss, val_accuracy = self._eval()
            print(
                f"Validation - Epoch [{epoch + 1}/{num_epochs}], Loss: {val_loss:.4f}, Accuracy: {val_accuracy:.4f}"
            )

    def save(self, save_path: str):
        torch.save(self.model.state_dict(), save_path)


if __name__ == "__main__":
    trainer = Trainer(root_path="input")
    trainer.train()
    trainer.save(save_path="model/dog_cat_classifier.pth")