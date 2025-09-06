
import torch
import torch.nn as nn
from gen_kernel_module import build


class SimpleNet(nn.Module):
    def __init__(self, n):
        super().__init__()
        self.fc1 = nn.Linear(2, n)
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
    model = SimpleNet(2)
    model.eval()

    #compiled = torch.compile(model, fullgraph=True, backend="inductor")
    data = torch.ones((2,), dtype=torch.float32)
    print(data, model(data))
    print(100 * data, model(100 * data))
    print([-1, 1], model(torch.tensor([-1, 1], dtype=torch.float32)))
    #get_primals(model, data)

    #print(compiled(data))
    build(model, data, 'main.c')
