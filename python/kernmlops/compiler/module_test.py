
import ctypes
import fcntl
import os
import time

import numpy as np
import torch
from gen_kernel_module import TorchKernelDeployer


def query_kernel_module(inp, out_size, measure_time=False):
    DEVICE = "/dev/model_run"

    IOCTL_PROC = 0xc0086d01

    # struct model_data {
    #   float *in;
    #   float *out;
    #   int input_size;
    # };
    class ModelData(ctypes.Structure):
        _fields_ = [
            ("in", ctypes.POINTER(ctypes.c_float)),
            ("out", ctypes.POINTER(ctypes.c_float)),
            ("input_size", ctypes.c_int),
        ]

    inp_flat = np.ravel(inp).astype(np.float32)
    n = inp_flat.size

    # create ctypes arrays
    in_arr = (ctypes.c_float * n)(*inp_flat)
    out_arr = (ctypes.c_float * out_size)()

    # build ioctl struct
    data = ModelData(in_arr, out_arr, n)

    # open device and call ioctl
    with open(DEVICE, "wb", buffering=0) as fd:
        t1 = time.time_ns()
        fcntl.ioctl(fd, IOCTL_PROC, data)
        t2 = time.time_ns()

    output = torch.from_numpy(np.ctypeslib.as_array(out_arr))
    if not measure_time:
        return output
    return output, t2 - t1

def test(model, inputs):
    try:
        module = TorchKernelDeployer(model, inputs[0].shape)
        module.build()
        os.system("cd build; make; sudo insmod my_module.ko")

        torch.set_printoptions(precision=5)  # number of decimal places
        for input in inputs:
            expected = model(input)
            output_size = 1
            for n in expected.shape:
                output_size *= n
            result = query_kernel_module(input, output_size)
            if not torch.allclose(expected, result, atol=1e-4, rtol=1e-4):
                print(f"FAILED with input {input}")
                print(f"\tExpected: {expected}")
                print(f"\tOutput: {result}")
            else:
                print("Passed!")
        print("Tests done!")
    finally:
        os.system("sudo rmmod my_module")


if __name__ == '__main__':
    while True:
        s = input("? ")
        nums = [float(i) for i in s.split(" ")]
        if len(nums) == 1:
            inp = torch.zeros((int(nums[0]),), dtype=torch.float32)
        else:
            inp = torch.tensor(nums, dtype=torch.float32)
        print(query_kernel_module(inp, 10))
