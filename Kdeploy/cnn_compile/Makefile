
PY_FILE := userspace_model.py
TRITON_KERNELS := triton_poi_fused_0.asm triton_poi_fused_max_pool2d_with_indices_5.asm triton_poi_fused_relu_4.asm triton_poi_fused_1.asm triton_poi_fused_max_pool2d_with_indices_7.asm triton_poi_fused_relu_6.asm triton_poi_fused_2.asm triton_poi_fused_max_pool2d_with_indices_9.asm triton_poi_fused_relu_8.asm triton_poi_fused_3.asm triton_poi_fused_relu_10.asm
KERNEL_MODULE_SRC := main
USER_SRC := user

TRITON_KERNELS_ASM := $(TRITON_KERNELS:=.asm)
TRITON_KERNELS_O := $(TRITON_KERNELS:=.o)

O_FILES := $(TRITON_KERNELS_O)

obj-m += my_module.o
my_module-objs := $(KERNEL_MODULE_SRC).o $(O_FILES)
CFLAGS_main.o += -mhard-float

PWD := $(CURDIR)

all: my_module.ko

$(TRITON_KERNELS_ASM): %.asm: $(PY_FILE)
	TRITON_LOG_LEVEL=debug TORCH_LOGS=output_code TORCHINDUCTOR_CACHE_DIR=dump python3 $<
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
