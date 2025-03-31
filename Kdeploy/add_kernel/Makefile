
KERNEL_MODULE_SRC := main
USER_SRC := user
TGT_KERNEL_NAME := add_kernel

obj-m += my_module.o
my_module-objs := $(KERNEL_MODULE_SRC).o $(TGT_KERNEL_NAME).o
PWD := $(CURDIR)

all: my_module.ko

$(TGT_KERNEL_NAME).asm: $(TGT_KERNEL_NAME).py
	TRITON_CPU_BACKEND=1 TRITON_KERNEL_DUMP=1 TRITON_ALWAYS_COMPILE=1 TRITON_DUMP_DIR=$(PWD)/dump python3 $<
	mv $(PWD)/dump/*/$@ .
	rm -rf dump
	python3 cleanup.py $@

$(TGT_KERNEL_NAME).o: $(TGT_KERNEL_NAME).asm
	as -o $@ $<
	touch .$@.cmd

my_module.ko: $(TGT_KERNEL_NAME).o $(KERNEL_MODULE_SRC).c
	$(MAKE) -C /lib/modules/$(shell uname -r)/build M=$(PWD) modules

user: $(TGT_KERNEL_NAME).o $(USER_SRC).c
	gcc $(USER_SRC).c $(TGT_KERNEL_NAME).o -o user

clean:
	$(MAKE) -C /lib/modules/$(shell uname -r)/build M=$(PWD) clean
	rm -f $(TGT_KERNEL_NAME).o $(TGT_KERNEL_NAME).asm .*.cmd user massif.out
