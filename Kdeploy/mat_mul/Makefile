
PY_FILE := matmul_kernel.py
TRITON_KERNELS := matmul_kernel
KERNEL_MODULE_SRC := main
USER_SRC := user

TRITON_KERNELS_ASM := $(TRITON_KERNELS:=.asm)
TRITON_KERNELS_O := $(TRITON_KERNELS:=.o)

O_FILES := $(TRITON_KERNELS_O) run_on_stack.o

obj-m += my_module.o
my_module-objs := $(KERNEL_MODULE_SRC).o $(O_FILES)
CFLAGS_main.o += -mhard-float

PWD := $(CURDIR)

all: my_module.ko

$(TRITON_KERNELS_ASM): %.asm: $(PY_FILE)
	TRITON_CPU_BACKEND=1 TRITON_KERNEL_DUMP=1 TRITON_ALWAYS_COMPILE=1 TRITON_DUMP_DIR=$(PWD)/dump python3 $<
	mv $(PWD)/dump/*/*.asm .
	rm -rf dump
	for file in *.asm; do \
		python3 cleanup.py "$$file"; \
	done

$(O_FILES): %.o: %.asm
	as -o $@ $<
	touch .$@.cmd

my_module.ko: $(O_FILES) $(KERNEL_MODULE_SRC).c
	$(MAKE) -C /lib/modules/$(shell uname -r)/build M=$(PWD) modules

user: $(O_FILES) $(USER_SRC).c
	gcc $(USER_SRC).c $(O_FILES) -o user -g -L$(HOME)/ldos-ml-compiler/triton-cpu/python/triton/_C -lTritonCPURuntime -Wl,-rpath,$(HOME)/ldos-ml-compiler/triton-cpu/python/triton/_C

clean:
	$(MAKE) -C /lib/modules/$(shell uname -r)/build M=$(PWD) clean
	rm -f $(TRITON_KERNELS_O) $(TRITON_KERNELS_ASM) .*.cmd user
