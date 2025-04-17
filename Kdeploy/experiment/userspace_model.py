
import torch
import torch.nn as nn
import time

from parse_and_generate import *

NUM_INFERENCES = 40 

class SimpleNet(nn.Module):
    def __init__(self, n):
        super().__init__()
        self.fc1 = nn.Linear(12, n)
        self.relu = nn.ReLU()

        self.fc2 = nn.Linear(n, n)
        self.relu1 = nn.ReLU()
        self.fc3 = nn.Linear(n, n)
        self.relu2 = nn.ReLU()
        self.fc4 = nn.Linear(n, 2)


    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu1(self.fc2(x))
        x = self.relu2(self.fc3(x))
        x = self.fc4(x)

        return x


if __name__ == "__main__":
    for i in [8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]:
        model = SimpleNet(i)

        compiled = torch.compile(model)
        data = torch.randn((12,), dtype=torch.float32);
        compiled(data)
        data = torch.randn((12,), dtype=torch.float32);

        print("============= SIZE %d ================" % i)
        for j in range(NUM_INFERENCES):
            t1 = time.time_ns()
            out = compiled(data)
            t2 = time.time_ns()
            diff = (t2 - t1) 
            print(diff)
