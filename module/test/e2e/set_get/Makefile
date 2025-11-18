BUILD ?= build
ROOT_DIR := $(dir $(realpath $(lastword $(MAKEFILE_LIST))))
OUT_DIR := ${BUILD}/$(notdir ${ROOT_DIR:/=})

TEST_SRC := $(wildcard *.cpp)
BLD_OUT := $(patsubst %.cpp,%,${TEST_SRC})
FULL_OUT := $(addprefix ${OUT_DIR}/,${BLD_OUT})

KERNEL_MOD_OUT = set_get.ko

KERNEL_MOD_NAME = $(patsubst %.ko,%,${KERNEL_MOD_OUT})

TMP_KERNEL_MOD_OUT = $(addprefix /tmp/,${KERNEL_MOD_NAME})

KBUILD_EXTRA_SYMBOLS := $(realpath ../../../fstore/Module.symvers)
obj-m += set_get.o

default: ${OUT_DIR}/${KERNEL_MOD_OUT}

echo:
	@echo BUILD ${BUILD}
	@echo ROOT_DIR ${ROOT_DIR}
	@echo OUT_DIR ${OUT_DIR}
	@echo KBUILD ${KBUILD}
	@echo TEST_SRC ${TEST_SRC}
	@echo BLD_OUT ${BLD_OUT}
	@echo FULL_OUT ${FULL_OUT}
	@echo KERNEL_MOD_OUT ${KERNEL_MOD_OUT}
	@echo TMP_KERNEL_MOD_OUT ${TMP_KERNEL_MOD_OUT}
	@echo KBUILD_EXTRA_SYMBOLS ${KBUILD_EXTRA_SYMBOLS}

${OUT_DIR}:
	mkdir -p $@

${OUT_DIR}/${KERNEL_MOD_OUT}: ${OUT_DIR}/%.ko : %.c ../../../fstore/fstore.h | ${OUT_DIR}
	+${MAKE} KBUILD_EXTRA_SYMBOLS=${KBUILD_EXTRA_SYMBOLS} -C ${KBUILD} M=${ROOT_DIR} modules
	cp $*.ko ${OUT_DIR}/${KERNEL_MOD_OUT}
	+${MAKE} -C . clean

${TMP_KERNEL_MOD_OUT} : /tmp/% : ${OUT_DIR}/%.ko
	-sudo rmmod $*
	sudo insmod $<
	touch $@

undeploy:
	-sudo rmmod ${KERNEL_MOD_NAME}
	-rm ${TMP_KERNEL_MOD_OUT}

${FULL_OUT}: ${OUT_DIR}/% : %.cpp ../../../fstore/fstore.h | ${OUT_DIR}
	${CXX} -O3 -I/usr/src/linux-headers-$(shell uname -r)/include/ \
		-std=gnu++2b $< -o $@

test: ${TMP_KERNEL_MOD_OUT} ${FULL_OUT}
	@$(foreach path,${FULL_OUT}, printf "$(shell basename $(path)) ... "; \
		sudo $(path) && printf "pass\n" || printf "fail\n";)

clean:
	+${MAKE} -C ${KBUILD} M=${ROOT_DIR} MO=${OUT_DIR} clean
