#!/usr/bin/env python3
import argparse

import torch
import torch.nn as nn
import torch.optim as optim


def main():
    parser = argparse.ArgumentParser(description='Simple PyTorch Training')
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--learning-rate', type=float, default=0.01)
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--dataset', type=str, default='')
    args = parser.parse_args()

    # Create dummy data: 500 samples, 10 features, binary labels
    data = torch.randn(500, 10)
    labels = torch.randint(0, 2, (500,))

    # Model: single linear layer
    model = nn.Linear(10, 2)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.learning_rate)

    for epoch in range(1, args.epochs + 1):
        optimizer.zero_grad()
        outputs = model(data)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        print(f'Epoch {epoch}/{args.epochs}, Loss: {loss.item():.4f}')

if __name__ == '__main__':
    main()
