KERNEL_MODULE_SRC := main
USER_SRC := main

MODULE_DIR := $(if $(KBUILD_EXTMOD),$(KBUILD_EXTMOD),$(CURDIR))

TRITON_KERNELS_ASM := $(notdir $(wildcard $(MODULE_DIR)/*.asm))
O_FILES := $(TRITON_KERNELS_ASM:.asm=.o)

SLEEF_OBJS := sleefsimdsp.c_2.o rempitab.c.o common.c.o #common.c.o commonxx.cpp.o dispscalar.c.o rempitab.c.o sleefsimddp.c.o sleefsimdsp.c.o
SLEEF_LIB := libsleef.a
SLEEF_EXTRACTED := .sleef_extracted

SLEEF_CMDS := $(addprefix .,$(addsuffix .cmd, $(SLEEF_OBJS)))

PWD := $(CURDIR)

obj-m += my_module.o
my_module-objs := $(KERNEL_MODULE_SRC).o $(O_FILES) $(SLEEF_OBJS) shim.o
CFLAGS_main.o += -DKERNEL_MODE -mhard-float -msse2

all: my_module.ko

$(O_FILES): %.o: %.asm
	clang -o $@ -c $<
	touch .$@.cmd


$(SLEEF_EXTRACTED): $(SLEEF_LIB) extract_sleef.sh
	rm -rf extracted_files
	./extract_sleef.sh
	cp $(addprefix extracted_files/,$(SLEEF_OBJS)) .
	touch $@

$(SLEEF_OBJS): $(SLEEF_EXTRACTED)

$(SLEEF_CMDS): .%.o.cmd: %.o
	touch $@

my_module.ko: $(O_FILES) $(SLEEF_OBJS) $(KERNEL_MODULE_SRC).c shim.c $(SLEEF_CMDS)
	$(MAKE) -C /lib/modules/$(shell uname -r)/build M=$(PWD) modules


user: $(O_FILES) $(USER_SRC).c $(SLEEF_LIB)
	gcc $(USER_SRC).c $(O_FILES) $(SLEEF_LIB) -o user -g

ioctl_test: test_ioctl.c
	gcc test_ioctl.c -o ioctl_test -g

clean:
	$(MAKE) -C /lib/modules/$(shell uname -r)/build M=$(PWD) clean
	rm -rf extracted_files $(SLEEF_EXTRACTED)
	rm -f $(O_FILES) $(TRITON_KERNELS_ASM) $(SLEEF_OBJS) .*.cmd user ioctl_test
