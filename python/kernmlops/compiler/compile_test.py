
import ctypes
import fcntl
import os

import numpy as np
import torch
from gen_kernel_module import build


def query_kernel_module(inp):
    DEVICE = "/dev/model_run"

    IOCTL_PROC = 0xc0086d01

    # struct model_data {
    #   float *in;
    #   float *out;
    #   int n;
    # };
    class ModelData(ctypes.Structure):
        _fields_ = [
            ("in", ctypes.POINTER(ctypes.c_float)),
            ("out", ctypes.POINTER(ctypes.c_float)),
            ("n", ctypes.c_int),
        ]

    n = len(inp)

    # create ctypes arrays
    in_arr = (ctypes.c_float * n)(*inp)
    out_arr = (ctypes.c_float * n)()

    # build ioctl struct
    data = ModelData(in_arr, out_arr, n)

    # open device and call ioctl
    with open(DEVICE, "wb", buffering=0) as fd:
        fcntl.ioctl(fd, IOCTL_PROC, data)

    return torch.from_numpy(np.ctypeslib.as_array(out_arr))

def test(model, inputs):
    build(model, inputs[0])
    os.system("cd build; make; sudo insmod my_module.ko")

    torch.set_printoptions(precision=5)  # number of decimal places

    try:
        for input in inputs:
            expected = model(input)
            result = query_kernel_module(input)
            if not torch.allclose(expected, result, atol=1e-4, rtol=1e-4):
                print(f"FAILED with input {input}")
                print(f"\tExpected: {expected}")
                print(f"\tOutput: {result}")
            else:
                print(f"Passed with input {input}")
        print("Tests done!")
    finally:
        os.system("sudo rmmod my_module")
