import torch
import torch.nn as nn


# Define the CNN model (same as before)
class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(128 * 8 * 8, 128)
        self.fc2 = nn.Linear(128, 10)  # Assume 10 classes (e.g., CIFAR-10)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = self.pool(torch.relu(self.conv3(x)))
        x = x.view(-1, 128 * 8 * 8)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def gen_primals_init_code(model, x):
    primals = []
    s = "float** allocate_primals() {\n"

    children = list(model.children())
    for i, layer in enumerate(children):
        if hasattr(layer, "weight"):
            primals.append(layer.weight)
        if hasattr(layer, "bias"):
            primals.append(layer.bias)
        # Insert input after first layer (matches most Inductor wrappers)
        if i == 0:
            primals.append(x)

    s += "\tfloat** primals = malloc(%d * sizeof(float*));\n" % len(primals)
    for i in range(len(primals)):
        p = torch.flatten(primals[i]).tolist()
        s += "\tprimals[%d] = malloc(%d * sizeof(float));\n" % (i, len(p))
        for j in range(len(p)):
            s += "\tprimals[%d][%d] = %.4f;\n" % (i, j, p[j])

    s += "\treturn primals;\n}\n"

    return s


if __name__ == "__main__":
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    torch._inductor.config.cpu_backend = "triton"
    device = torch.device("cpu")

    model = CNNModel()
    model.to(device)
    model.eval()  # Set the model to evaluation mode

    compiled = torch.compile(model, fullgraph=True, backend="inductor")
    data = torch.ones(1, 3, 64, 64).to(device)  # Batch size 1, 3 channels (RGB), 64x64 size
    with open("allocate_primals.c", 'w') as f:
        f.write(gen_primals_init_code(model, data))

    print(compiled(data))
