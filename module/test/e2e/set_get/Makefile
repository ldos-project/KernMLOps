BUILD ?= build
ROOT_DIR := $(dir $(realpath $(lastword $(MAKEFILE_LIST))))
OUT_DIR := ${BUILD}/$(notdir ${ROOT_DIR:/=})/

KERNEL_MOD_OUT = get_set.ko

obj-m += get_set.o

echo:
	@echo BUILD ${BUILD}
	@echo ROOT_DIR ${ROOT_DIR}
	@echo OUT_DIR ${OUT_DIR}
	@echo KBUILD ${KBUILD}
	@echo KERNEL_MOD_OUT ${KERNEL_MOD_OUT}

${OUT_DIR}${KERNEL_MOD_OUT}: ${OUT_DIR}%.ko : %.c
	+${MAKE} -C ${KBUILD} M=${ROOT_DIR} MO=${OUT_DIR} modules

clean:
	+${MAKE} -C ${KBUILD} M=${ROOT_DIR} MO=${OUT_DIR} clean
