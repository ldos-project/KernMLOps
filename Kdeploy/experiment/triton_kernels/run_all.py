
import os
import time

sizes = [8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]

for i in sizes:
    print("================ SIZE %d ================" % i, flush=True)
    for j in range(40):
        os.system("sudo insmod my_module.ko nn_size=%d" % i)
        time.sleep(0.01)
        os.system("sudo rmmod my_module")
        os.system("sudo dmesg | tail -n 2 | grep 'ns'")


