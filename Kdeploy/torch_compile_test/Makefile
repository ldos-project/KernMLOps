
KERNEL_MODULE_SRC := main
USER_SRC := user

TRITON_KERNELS_ASM := $(wildcard *.asm)
O_FILES := $(TRITON_KERNELS:=.o)

PWD := $(CURDIR)

obj-m += my_module.o
my_module-objs := $(KERNEL_MODULE_SRC).o $(O_FILES)
CFLAGS_main.o += -mhard-float

all: my_module.ko

$(O_FILES): %.o: %.asm
	clang -o $@ -c $<
	touch .$@.cmd

my_module.ko: $(O_FILES) $(KERNEL_MODULE_SRC).c
	$(MAKE) -C /lib/modules/$(shell uname -r)/build M=$(PWD) modules

user: $(O_FILES) $(USER_SRC).c
	gcc $(USER_SRC).c $(O_FILES) -o user -g

clean:
	$(MAKE) -C /lib/modules/$(shell uname -r)/build M=$(PWD) clean
	rm -f $(O_FILES) $(TRITON_KERNELS_ASM) .*.cmd user
