import ctypes
import os
import socket
import struct

import numpy as np
import torch
import torch.nn as nn
from bcc import libbcc

# Netlink settings (must match the kernel sender)
NETLINK_USER = 31
UPCALL_HDR_FMT = "I"
UPCALL_HDR_SIZE = struct.calcsize(UPCALL_HDR_FMT)
MODEL_INPUT_DIM = 16

PIN_PATH = "/sys/fs/bpf/infer_output"
MAP_NAME = "infer_output"
OUT_BUF_SIZE = 512
MAX_ENTRIES = 1  # as declared in C map

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(16, 128),
            nn.ReLU(),
            nn.Linear(128, 8)
        )

    def forward(self, x):
        return self.layers(x)

def run_inference(model, payload_bytes):
    # Interpret payload_bytes as int32 array (kernel sends 16 ints)
    n_i32 = len(payload_bytes) // 4
    arr = np.frombuffer(payload_bytes[:n_i32*4], dtype=np.int32).astype(np.float32)

    arr = arr[:MODEL_INPUT_DIM].reshape(1, MODEL_INPUT_DIM)
    x = torch.from_numpy(arr)

    with torch.no_grad():
        out = model(x)
    out_np = out.detach().cpu().numpy().astype(np.float32)

    return out_np

def open_pinned_map(pin_path=PIN_PATH):
    # Use libbcc C API to get fd for pinned object
    fd = libbcc.lib.bpf_obj_get(pin_path.encode("utf-8"))
    if fd < 0:
        raise RuntimeError("bpf_obj_get failed for " + pin_path)
    return fd

def update_map_bytes(fd, key_index, byte_buf):
    # Build ctypes for key and value
    key = ctypes.c_uint32(key_index)
    # value type is unsigned char[OUT_BUF_SIZE]
    class Val(ctypes.Structure):
        _fields_ = [("data", ctypes.c_ubyte * OUT_BUF_SIZE)]
    val = Val()
    # zero then copy
    for i in range(OUT_BUF_SIZE):
        val.data[i] = 0
    for i, b in enumerate(byte_buf[:OUT_BUF_SIZE]):
        val.data[i] = b
    # call into libbcc C API
    ret = libbcc.lib.bpf_update_elem(fd, ctypes.byref(key), ctypes.byref(val), 0)
    if ret != 0:
        raise RuntimeError("bpf_update_elem failed: " + str(ret))

def main():
    model = Model()
    fd = open_pinned_map(PIN_PATH)
    print("Got pinned map fd:", fd)

    # create netlink socket
    s = socket.socket(socket.AF_NETLINK, socket.SOCK_RAW, NETLINK_USER)
    s.bind((os.getpid(), 0))
    s.sendto(b"REGISTER", (0, 0))
    print("Registered with kernel via netlink; waiting for messages")

    try:
        while True:
            data = s.recv(65536)
            if not data:
                continue
            if len(data) < UPCALL_HDR_SIZE:
                print("Short netlink packet, skipping")
                continue
            (payload_len,) = struct.unpack(UPCALL_HDR_FMT, data[:UPCALL_HDR_SIZE])
            payload = data[UPCALL_HDR_SIZE:UPCALL_HDR_SIZE + payload_len]
            print("Received payload length:", payload_len)

            out_bytes = run_inference(model, payload).tobytes()
            update_map_bytes(fd, 0, out_bytes)
            print("Wrote inference output ({} bytes) into pinned map".format(len(out_bytes)))

    except KeyboardInterrupt:
        print("exiting")
    finally:
        s.close()

if __name__ == "__main__":
    main()
