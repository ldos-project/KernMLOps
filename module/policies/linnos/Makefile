obj-m += raid1_policy.o
KBUILD_EXTRA_SYMBOLS := $(realpath ../../fstore/Module.symvers)
KBUILD ?= $(PWD)/../../kbuild
ROOT_DIR := $(dir $(realpath $(lastword $(MAKEFILE_LIST))))
ccflags-y += -msse

echo:
	@echo obj-m ${obj-m}
	@echo KBUILD ${KBUILD}
	@echo KBUILD_EXTRA_SYMBOLS ${KBUILD_EXTRA_SYMBOLS}
	@echo ROOT_DIR ${ROOT_DIR}
	@echo ccflags-y ${ccflags-y}

default: linnos-inference-capture

linnos-inference-capture.o: linnos-inference-capture.bpf.c
	clang -O2 -g -target bpf -c $^ -o $@ \
		-D__TARGET_ARCH_x86_64 \
		-I/usr/include/ \
  	-I/usr/src/linux-headers-$(shell uname -r)/include \
  	-I/usr/src/linux-headers-$(shell uname -r)/arch/x86/include \
  	-I/usr/src/linux-headers-$(shell uname -r)/arch/x86/include/generated/

linnos-inference-capture: linnos-inference-capture.cpp
	g++ -std=gnu++2b -o $@ $< -lbpf -I../../fstore/

raid1_policy.ko: raid1_policy.c
	+${MAKE} KBUILD_EXTRA_SYMBOLS=${KBUILD_EXTRA_SYMBOLS} -C ${KBUILD} M=${ROOT_DIR} modules

run: linnos-inference-capture raid1_policy.ko linnos-inference-capture.o
	sudo ./$<
