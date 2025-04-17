
import os
import time

sizes = [8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]

mkfile = ""
with open("Makefile", "r") as f:
    mkfile = f.read()

for i in sizes:
    with open("Makefile", "w") as f:
        f.write(mkfile.replace("[tmp]", str(i)))

    os.system("make")

with open("Makefile", "w") as f:
    f.write(mkfile)


for i in sizes:
    print("============== SIZE %d ==============" % i)
    for j in range(40):
        os.system("sudo insmod model_%d.ko" % i)
        time.sleep(0.01)
        os.system("sudo rmmod model_%d" % i)
        os.system("sudo dmesg | tail -n 2 | grep 'ns' ")


