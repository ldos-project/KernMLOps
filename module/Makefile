
KBUILD ?= $(shell pwd)/kbuild
KHEADS ?= $(shell pwd)/kheads
MODULE ?= $(shell pwd)/module
KERNEL ?= $(shell pwd)/linux

BUILD := build

REAL_BUILD = $(realpath build)

EXPORT_VARS := $(if ${BUILD}, BUILD=${REAL_BUILD}) KBUILD=${KBUILD}

LATEST_STABLE_KERNEL := $(shell ls /boot | grep -E '^vmlinuz+-[0-9]+\.[0-9]+\.[0-9]+-[0-9]+-generic' | sort -V | tail -n 1)

LATEST_STABLE_CONFIG := $(patsubst vmlinuz-%, config-%, $(LATEST_STABLE_CONFIG))

SUDO ?= ""

DIRS = ${KBUILD} ${MODULE} ${BUILD}

.PHONY: test

all: ${MODULE}/lib

/tmp/deploy: fstore/fstore.ko
	-sudo rmmod fstore
	sudo insmod $<
	touch $@

undeploy:
	+${MAKE} -C test undeploy
	-sudo rmmod fstore
	rm -f /tmp/deploy

kdev_test: ${KHEADS} | build
	+${MAKE} ${EXPORT_VARS} -C test kdev_test

unit_test: ${KHEADS} /tmp/deploy
	+${MAKE} ${EXPORT_VARS} -C test unit_test

e2e_test: ${KHEADS} /tmp/deploy
	+${MAKE} ${EXPORT_VARS} -C test e2e_test

test: kdev_test unit_test e2e_test

echo:
	@echo KBUILD ${KBUILD}
	@echo KERNEL ${KERNEL}
	@echo MODULE ${MODULE}
	@echo KHEADS ${KHEADS}
	@echo SUDO ${SUDO}
	@echo BUILD ${BUILD}
	@echo REAL_BUILD ${REAL_BUILD}
	@echo EXPORT_VARS ${EXPORT_VARS}
	@echo test:
	@${MAKE} ${EXPORT_VARS} -C test echo

${DIRS}:
	mkdir -p $@

${KBUILD}/.config: /boot/${LATEST_STABLE_CONFIG} config/config_edits.txt | ${KBUILD}
	cp $< $@
	../config/apply_config.sh ${KERNEL}/scripts/config config/config_edits.txt $@
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
