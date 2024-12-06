
KBUILD ?= $(shell pwd)/kbuild
KHEADS ?= $(shell pwd)/kheads
MODULE ?= $(shell pwd)/module
KERNEL ?= $(shell pwd)/linux

BUILD := $(realpath build)

SUDO ?= ""

DIRS = ${KBUILD} ${MODULE} ${BUILD}

.PHONY: test

all: fstore/fstore.ko

/tmp/deploy: fstore/fstore.ko
	-sudo rmmod fstore
	sudo insmod $<
	touch $@

undeploy:
	-sudo rmmod fstore
	rm -f /tmp/deploy

kdev_test: ${KHEADS} | build
	+${MAKE} BUILD=${BUILD} -C test kdev_test

unit_test: ${KHEADS} /tmp/deploy
	+${MAKE} BUILD=${BUILD} -C test unit_test

test: kdev_test unit_test

echo:
	@echo KBUILD ${KBUILD}
	@echo KERNEL ${KERNEL}
	@echo MODULE ${MODULE}
	@echo KHEADS ${KHEADS}
	@echo SUDO ${SUDO}
	@echo BUILD ${BUILD}
	@echo test:
	@${MAKE} BUILD=${BUILD} -C test echo

${DIRS}:
	mkdir -p $@

${KBUILD}/.config: config/config-6.12.0+ | ${KBUILD}
	cp $< $@
	+yes ""| ${MAKE} -C ${KERNEL} O=${KBUILD} olddefconfig

${KBUILD}/vmlinux: ${KBUILD}/.config
	+${MAKE} -C ${KBUILD}
	touch $@

${KHEADS}: ${KBUILD}/vmlinux
	+${SUDO} ${MAKE} -C ${KBUILD} headers_install ARCH=$(shell uname -m) INSTALL_HDR_PATH=$@
	sudo touch $@

${MODULE}/lib: ${KBUILD}/vmlinux ${KHEADS} | ${MODULE}
	+${SUDO} ${MAKE} -C ${KBUILD} INSTALL_MOD_PATH=${MODULE} INSTALL_MOD_STRIP=1 modules_install
	${SUDO} touch $@

fstore/fstore.ko: ${KBUILD}/vmlinux ${MODULE}/lib fstore/fstore.c fstore/fstore.h
	+${MAKE} -C fstore

clean:
	+${MAKE} -C fstore clean

Clean: clean
	+{MAKE} -C ${KBUILD} clean
