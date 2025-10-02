KERNEL_MODULE_SRC := main
USER_SRC := main

TRITON_KERNELS_ASM := $(wildcard *.asm)
O_FILES := $(TRITON_KERNELS_ASM:.asm=.o)

SLEEF_OBJS := sleefsimdsp.c_2.o rempitab.c.o common.c.o #common.c.o commonxx.cpp.o dispscalar.c.o rempitab.c.o sleefsimddp.c.o sleefsimdsp.c.o

SLEEF_CMDS := $(addprefix .,$(addsuffix .cmd, $(SLEEF_OBJS)))

PWD := $(CURDIR)

obj-m += my_module.o
my_module-objs := $(KERNEL_MODULE_SRC).o $(O_FILES) $(SLEEF_OBJS) shim.o
CFLAGS_main.o += -DKERNEL_MODE -mhard-float

all: my_module.ko

$(O_FILES): %.o: %.asm
	clang -o $@ -c $<
	touch .$@.cmd

$(SLEEF_CMDS): .%.o.cmd: %.o
	touch $@

my_module.ko: $(O_FILES) $(KERNEL_MODULE_SRC).c shim.c $(SLEEF_CMDS)
	$(MAKE) -C /lib/modules/$(shell uname -r)/build M=$(PWD) modules

user: $(O_FILES) $(USER_SRC).c
	gcc $(USER_SRC).c $(O_FILES) -o user -g

clean:
	$(MAKE) -C /lib/modules/$(shell uname -r)/build M=$(PWD) clean
	rm -f $(O_FILES) $(TRITON_KERNELS_ASM) .*.cmd user
