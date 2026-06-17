# How to Generate Kernel Modules with KCompiler

## Preparing the environment

If you are on a cloudlab machine we have prepared then run `. /opt/uv/env`

Within the repo root run `. .venv/bin/activate`

First create a file similar to sample_nn.py

```python
import pathlib
import torch
import torch.nn as nn
try:
    from .gen_kernel_module import TorchKernelDeployer
except ImportError:
    from gen_kernel_module import TorchKernelDeployer

lass SimpleNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(16, 8)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(8, 2)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        return self.fc2(x)


def main():
    model = SimpleNet().eval()
    input_shape = (16,)
    output_dir = pathlib.Path("build")

    module = TorchKernelDeployer(model, input_shape)
    module.build(output_dir)

    print(f"Generated files in {output_dir}")
    print("User: cd build && make user && ./user")
    print("Kernel: cd build && make && sudo insmod my_module.ko")

if __name__ == "__main__":
    main()
```

Then inside the top level described virtual environment run:
`python sample_nn.py`.
After that to run the user-space variant (this is just to test) run 
`cd build && make user && ./user; popd`.

This will print out the result directly.

To run the kernel version:
```
cd build && make && sudo insmod my_module.ko; popd
```

To see the result run `sudo dmesg` you should see:
```
[Timestamp] Hello: <some number>`
```

You can then make ioctl calls to the char device at <some number> and it will run the desired module calls.

We included a sample in `ioctl_test.c` that is automatically created in the build directory.

You can do this with:

```bash
cd build && make ioctl_test; ./ioctl_test /dev/model_run; popd
```
