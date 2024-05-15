import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from PoseHead import PoseHead
from torchvision.transforms import ToTensor
import random

random.seed(42)

class CustomDataset(Dataset):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.data = []
        self.read_data()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        category, keypoints = self.data[idx]
        return torch.tensor(keypoints, dtype=torch.float32), torch.tensor(category, dtype=torch.long)

    def read_data(self):
        file_list = sorted([f for f in os.listdir(self.data_dir) if os.path.isfile(os.path.join(self.data_dir, f))])
        for file_name in file_list:
            file_path = os.path.join(self.data_dir, file_name)
            self.read_file(file_path)

    def read_file(self, file_path):
        with open(file_path, 'r') as file:
            lines = file.readlines()
            for line in lines:
                line_data = line.strip().split()
                category = int(line_data[0])
                keypoints = [[float(line_data[i]), float(line_data[i+1])] for i in range(1, len(line_data), 2)]
                self.data.append((category, keypoints))


def train_model(dataloader, model, criterion, optimizer, num_epochs=200):
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in dataloader:
            labels = F.one_hot(labels, num_classes=2).float()
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
        epoch_loss = running_loss / len(dataloader.dataset)
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss:.4f}")
    # print(outputs)
    return model


if __name__ == "__main__":
    data_dir = '../datasets/bird_pose/labels'
    dataset = CustomDataset(data_dir)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

    model = PoseHead()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.0001)

    model = train_model(dataloader, model, criterion, optimizer)
    torch.save(model.state_dict(), "posehead.pt")
