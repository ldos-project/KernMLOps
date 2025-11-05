
import os
import sys
import time

import torch
import torch.nn as nn
from gen_kernel_module import TorchKernelDeployer
from module_test import query_kernel_module

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

# ./user_kernel_comparison.py [output_file] [U | K]
if __name__ == "__main__":
    for i in [8, 16, 32, 64, 128]:#, 256, 512, 1024, 2048, 4096]:
        model = SimpleNet(i)
        data = torch.randn((12,), dtype=torch.float32)
        expected_out = model(data)

        with open(sys.argv[2], 'w') as out:
            if sys.argv[1] == "U":
                compiled = torch.compile(model)
                compiled(data)
                data = torch.randn((12,), dtype=torch.float32)

                out.write("============= USER SIZE %d ================\n" % i)
                for j in range(NUM_INFERENCES):
                    t1 = time.time_ns()
                    out = compiled(data)
                    t2 = time.time_ns()
                    diff = (t2 - t1)
                    out.write(str(diff) + "\n")

            elif sys.argv[1] == "K":
                try:
                    module = TorchKernelDeployer(model, data.shape)
                    module.build()
                    os.system("cd build; make; sudo insmod my_module.ko")

                    out.write("============= KERN SIZE %d ================\n" % i)
                    for j in range(NUM_INFERENCES):
                        out, time = query_kernel_module(data, expected_out.shape[0])
                        out.write(str(time) + "\n")

                finally:
                    os.system("sudo rmmod my_module")
