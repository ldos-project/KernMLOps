
all: addmm

addmm:
	TRITON_CPU_BACKEND=1 TRITON_KERNEL_DUMP=1 TRITON_ALWAYS_COMPILE=1 TRITON_DUMP_DIR=dump python3 addmm.py
	mv dump/*/*.asm .
	rm -rf dump
