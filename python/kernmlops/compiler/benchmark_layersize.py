
import os
import sys
import time

import torch
import torch.nn as nn

from .gen_kernel_module import TorchKernelDeployer
from .module_test import query_kernel_module

NUM_INFERENCES = 40

class SimpleNet(nn.Module):
    def __init__(self, n):
        super().__init__()
        self.l1 = nn.Linear(16, n)
        self.relu = nn.ReLU()
        self.l2 = nn.Linear(n, 2)

    def forward(self, x):
        x = self.relu(self.l1(x))
        x = self.l2(x)
        return x

# ./user_kernel_comparison.py [output_file] [PG | PC | TU | TK]
if __name__ == "__main__":
    torch._dynamo.config.cache_size_limit = 32
    with open(sys.argv[2], 'w') as out:
        for i in [4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32786]:
            print("!!!!!", i)
            model = SimpleNet(i)
            data = torch.randn((16,), dtype=torch.float32)
            expected_out = model(data)

            if sys.argv[1] == "TU": # triton user
                module = TorchKernelDeployer(model, data.shape)
                module.build()
                os.system("cd build; make user;")

                out.write("============= SIZE %d ================\n" % i)
                for j in range(NUM_INFERENCES):
                    os.system("cd build; ./user > abc.txt;")
                    t = 0
                    with open("build/abc.txt") as f:
                        t = float(f.read())
                        out.write(str(t) + "\n")

            elif sys.argv[1] == "PG": # pytorch gpu
                model.to("cuda")
                data = data.to("cuda")
                compiled = torch.compile(model)
                compiled(data)

                out.write("============= SIZE %d ================\n" % i)
                for j in range(NUM_INFERENCES):
                    data = torch.randn((16,), dtype=torch.float32)
                    t1 = time.time_ns()
                    data = data.to("cuda")
                    o = compiled(data)
                    t2 = time.time_ns()
                    diff = (t2 - t1)
                    out.write(str(diff) + "\n")

            elif sys.argv[1] == "PC": # pytorch cpu
                os.environ["CUDA_VISIBLE_DEVICES"] = ""
                compiled = torch.compile(model)
                compiled(data)

                out.write("============= SIZE %d ================\n" % i)
                for j in range(NUM_INFERENCES):
                    data = torch.randn((16,), dtype=torch.float32)
                    t1 = time.time_ns()
                    o = compiled(data)
                    t2 = time.time_ns()
                    diff = (t2 - t1)
                    out.write(str(diff) + "\n")

            elif sys.argv[1] == "TK": # triton kernel
                try:
                    module = TorchKernelDeployer(model, data.shape)
                    module.build()
                    os.system("cd build; sudo make; sudo insmod my_module.ko")

                    out.write("============= SIZE %d ================\n" % i)
                    for j in range(NUM_INFERENCES):
                        o, time = query_kernel_module(data, expected_out.shape[0], measure_time=True)
                        out.write(str(time) + "\n")

                finally:
                    os.system("sudo rmmod my_module")
