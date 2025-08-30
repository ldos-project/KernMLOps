
import torch
import torch.nn as nn


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

def get_primals(model, x):
    primals = []

    children = list(model.children())
    for i, layer in enumerate(children):
        if hasattr(layer, "weight"):
            primals.append(layer.weight)
        if hasattr(layer, "bias"):
            primals.append(layer.bias)
        # Insert input after first layer (matches most Inductor wrappers)
        if i == 0:
            primals.append(x)

    for i in range(len(primals)):
        p = torch.flatten(primals[i]).tolist()
        for j in range(len(p)):
            print("primals[%d][%d] = %.4f;" % (i, j, p[j]))


if __name__ == "__main__":
    torch._inductor.config.cpu_backend = "triton"
    model = SimpleNet(2)

    compiled = torch.compile(model, fullgraph=True, backend="inductor")
    data = torch.ones((2,), dtype=torch.float32)
    get_primals(model, data)
    input()

    print(compiled(data))

    print("\n\n================")
    print(compiled(data))
    '''
    sample_input = (torch.ones((2,), dtype=torch.float32),);
    exported = torch.export.export(model, sample_input)
    print(exported)
    '''
