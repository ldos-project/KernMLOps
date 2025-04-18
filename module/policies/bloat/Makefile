obj-m += bloat_policy.o
KBUILD_EXTRA_SYMBOLS := $(realpath ../../fstore/Module.symvers)
KBUILD ?= $(PWD)/../../kbuild
ROOT_DIR := $(dir $(realpath $(lastword $(MAKEFILE_LIST))))
ccflags-y += -msse

default: run

echo:
	@echo obj-m ${obj-m}
	@echo KBUILD ${KBUILD}
	@echo KBUILD_EXTRA_SYMBOLS ${KBUILD_EXTRA_SYMBOLS}
	@echo ROOT_DIR ${ROOT_DIR}
	@echo ccflags-y ${ccflags-y}


bloat_probe.o: bloat.ebpf.c
	clang -O2 -g -target bpf -c $^ -o $@ \
		-D__TARGET_ARCH_x86_64 \
		-I/usr/include/ \
  	-I/usr/src/linux-headers-$(shell uname -r)/include \
  	-I/usr/src/linux-headers-$(shell uname -r)/arch/x86/include \
  	-I/usr/src/linux-headers-$(shell uname -r)/arch/x86/include/generated/

bloat_policy_deployer: bloat_policy_deployer.cpp
	g++ -std=gnu++2b -o $@ $< -lbpf -I../../fstore/

raid1_policy.ko: raid1_policy.c
	+${MAKE} KBUILD_EXTRA_SYMBOLS=${KBUILD_EXTRA_SYMBOLS} -C ${KBUILD} M=${ROOT_DIR} modules

run: bloat_policy_deployer bloat_probe.o
	sudo ./$<

clean:
	-rm bloat_policy_deployer
	-rm bloat_policy_deployer.o
	-rm bloat_probe.o
	-rm *module*
	-rm *.mod*
	-rm *.o*
	-rm .*.o*
	-rm *.ko*
	-rm .*.cmd
	-rm Module.symvers
