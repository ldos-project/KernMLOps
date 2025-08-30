
PY_FILE := userspace_model.py
TRITON_KERNELS := triton_poi_fused_addmm_2 triton_poi_fused_relu_0 triton_poi_fused_relu_1
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
	TORCH_LOGS=output_code TORCHINDUCTOR_CACHE_DIR=dump python3 $<
	mv $(PWD)/dump/triton/*/*/*.asm .
	# rm -rf dump

$(O_FILES): %.o: %.asm
	clang -o $@ -c $<
	touch .$@.cmd

my_module.ko: $(O_FILES) $(KERNEL_MODULE_SRC).c
	$(MAKE) -C /lib/modules/$(shell uname -r)/build M=$(PWD) modules

user: $(O_FILES) $(USER_SRC).c
	gcc $(USER_SRC).c $(O_FILES) -o user -g

clean:
	$(MAKE) -C /lib/modules/$(shell uname -r)/build M=$(PWD) clean
	rm -f $(TRITON_KERNELS_O) $(TRITON_KERNELS_ASM) .*.cmd user
