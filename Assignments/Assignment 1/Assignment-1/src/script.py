import pandas as pd
import torch
import matplotlib.pyplot as plt
import matplotlib.image as img

df = pd.read_csv('C:/Users/chira/Projects/AIML Lab/Assignments/Assignment 1/Assignment-1/input/train.csv')
df1 = df.sample(frac = 1)
df1.reset_index(inplace = True, drop = True)

def train_val_test_split(df, val_size, test_size):
    val_size = int(len(df) * val_size)
    test_size = int(len(df) * test_size)
    train_data, val_data, test_data = df[test_size+val_size:], df[val_size:test_size+val_size], df[:test_size]
    train_data.reset_index(inplace = True, drop = True)
    val_data.reset_index(inplace = True, drop = True)
    test_data.reset_index(inplace = True, drop = True)
    return train_data, val_data, test_data

def getData(data, val_size, test_size):
    train, val, test = train_val_test_split(data, val_size, test_size)
    train_x = torch.from_numpy(train.iloc[:,1:].values.reshape(-1,1,28,28)).float()
    train_y = torch.nn.functional.one_hot(torch.from_numpy(train.iloc[:,0].values.reshape(-1,1)).squeeze(1).long(), num_classes=10)
    val_x = torch.from_numpy(val.iloc[:,1:].values.reshape(-1,1,28,28)).float()
    val_y = torch.nn.functional.one_hot(torch.from_numpy(val.iloc[:,0].values.reshape(-1,1)).squeeze(1).long(), num_classes=10)
    test_x = torch.from_numpy(test.iloc[:,1:].values.reshape(-1,1,28,28)).float()
    test_y = torch.nn.functional.one_hot(torch.from_numpy(test.iloc[:,0].values.reshape(-1,1)).squeeze(1).long(), num_classes=10)
    return train_x, train_y, val_x, val_y, test_x, test_y

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(64 * 7 * 7, 256)  # Adjust input size based on the size of your input tensor
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 10)

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
        x = self.relu(x)
        x = self.fc4(x)
        x = torch.softmax(x, dim=1)
        return x

class Trainer:
    def __init__(self, train_data_tensor, train_label_tensor, val_data_tensor, val_label_tensor, test_data_tensor, test_label_tensor):
        self.transform = None
        self.train_data_tensor = train_data_tensor
        self.train_label_tensor = train_label_tensor
        self.val_data_tensor = val_data_tensor
        self.val_label_tensor = val_label_tensor
        self.test_data_tensor = test_data_tensor
        self.test_label_tensor = test_label_tensor
        self.threshold = 0.5

        self.model = Classifier()
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)

    def _extract_data(self):
        self.train_loader = DataLoader(TensorDataset(self.train_data_tensor, self.train_label_tensor), batch_size=32, shuffle=True)
        self.val_loader = DataLoader(TensorDataset(self.val_data_tensor, self.val_label_tensor), batch_size=32, shuffle=False)
        self.test_loader = DataLoader(TensorDataset(self.test_data_tensor, self.test_label_tensor), batch_size=32, shuffle=False)

    def _eval(self, loader):
        self.model.eval()
        total_loss = 0
        correct_predictions = 0
        total_samples = 0

        with torch.no_grad():
            for inputs, labels in loader:
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels.type(torch.FloatTensor))

                total_loss += loss.item()
                predicted = outputs > self.threshold
                correct_predictions += (predicted == labels).sum().item()
                total_samples += labels.size(0)

        loss = total_loss / len(loader)
        accuracy = correct_predictions / total_samples
        return loss, accuracy

    def _train(self):
        self.model.train()
        total_loss = 0
        correct_predictions = 0
        total_samples = 0

        for inputs, labels in self.train_loader:
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            labels = labels.view(-1, 1) if labels.dim() == 1 else labels
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

    def train(self, num_epochs=40):
        self._extract_data()

        for epoch in range(num_epochs):
            train_loss, train_accuracy = self._train()
            print(
                f"Training - Epoch [{epoch + 1}/{num_epochs}], Loss: {train_loss:.4f}, Accuracy: {train_accuracy:.4f}"
            )
            val_loss, val_accuracy = self._eval(self.val_loader)
            print(
                f"Validation - Epoch [{epoch + 1}/{num_epochs}], Loss: {val_loss:.4f}, Accuracy: {val_accuracy:.4f}"
            )

        test_loss, test_accuracy = self._eval(self.test_loader)
        print(f"Testing - Loss: {test_loss:.4f}, Accuracy: {test_accuracy:.4f}")

    def save(self, save_path: str):
        torch.save(self.model.state_dict(), save_path)


if __name__ == "__main__":
    train_x, train_y, val_x, val_y, test_x, test_y = getData(df1, val_size = 0.1, test_size = 0.1)
    trainer = Trainer(train_x, train_y, val_x, val_y, test_x, test_y)
    
    trainer.train()
    trainer.save(save_path="model/classifier_model.pth")
    
