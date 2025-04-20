#!/usr/bin/env python3
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader


# ──────── Network Definition ──────────────────────────────────────────────────
class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 8, 3, padding=1)
        self.pool  = nn.MaxPool2d(2, 2)
        self.fc1   = nn.Linear(8 * 14 * 14, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = x.view(-1, 8 * 14 * 14)
        return self.fc1(x)

# ──────── Argument Parsing ────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="Train SimpleCNN on FashionMNIST")
parser.add_argument("--epochs",        type=int,   default=1,
                    help="number of training epochs")
parser.add_argument("--batch-size",    type=int,   default=32,
                    help="mini‑batch size")
parser.add_argument("--learning-rate", type=float, default=0.001,
                    help="SGD learning rate")
parser.add_argument("--device",        type=str,   default="cpu",
                    help="training device, e.g. 'cpu' or 'cuda'")
parser.add_argument("--data-root",     type=str,   default="./data",
                    help="location where FashionMNIST is already downloaded")
args = parser.parse_args()

# ──────── Data Loading ────────────────────────────────────────────────────────
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_dataset = torchvision.datasets.FashionMNIST(
    root=args.data_root,
    train=True,
    download=False,     # already handled in setup()
    transform=transform
)
train_loader = DataLoader(
    train_dataset,
    batch_size=args.batch_size,
    shuffle=True
)

# ──────── Model, Optimizer, Loss ─────────────────────────────────────────────
device = torch.device(args.device)
model = SimpleCNN().to(device)
optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate)
criterion = nn.CrossEntropyLoss()

# ──────── Training Loop ──────────────────────────────────────────────────────
for epoch in range(1, args.epochs + 1):
    total_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch}/{args.epochs}, Loss: {avg_loss:.4f}")
