
import torch
import torch.nn as nn
from compile_test import test


class SimpleNet(nn.Module):
    def __init__(self, a, b, c):
        super().__init__()
        self.fc1 = nn.Linear(a, b)
        self.relu = nn.ReLU()

        self.fc2 = nn.Linear(b, b)
        self.relu1 = nn.ReLU()
        self.fc3 = nn.Linear(b, b)
        self.relu2 = nn.ReLU()
        self.fc4 = nn.Linear(b, c)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu1(self.fc2(x))
        x = self.relu2(self.fc3(x))
        x = self.fc4(x)

        return x

if __name__ == "__main__":
    model = SimpleNet(20, 50, 10)
    model.eval()
    print(model(torch.zeros((20,), dtype=torch.float32)))
    test(model, [torch.randn((20,), dtype=torch.float32) for i in range(50)])
