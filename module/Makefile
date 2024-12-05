
KBUILD ?= $(shell pwd)/kbuild
KHEADS ?= $(shell pwd)/kheads
MODULE ?= $(shell pwd)/module
KERNEL ?= $(shell pwd)/linux

SUDO ?= ""

TEST_SRC = $(wildcard test/*.cpp)
TEST_BLD = $(patsubst %.cpp,%,$(subst test/,build/,${TEST_SRC}))

.PHONY: test

all: fstore/fstore.ko

/tmp/deploy: fstore/fstore.ko
	sudo insmod $<
	touch $@

undeploy:
	sudo rmmod fstore
	rm -f /tmp/deploy

${TEST_BLD}: build/% : test/%.cpp ${KHEADS}
	echo $^
	${CXX} -O3 -I/usr/src/linux-headers-$(shell uname -r)/include/ -std=gnu++2b $< -o $@

test: /tmp/deploy ${TEST_BLD}
	@$(foreach path,${TEST_BLD}, printf "$(shell basename $(path)) ... "; ${SUDO} $(path) && printf "pass\n" || printf "fail\n";)

echo:
	@echo TEST_BLD ${TEST_BLD}
	@echo TEST_SRC ${TEST_SRC}
	@echo KBUILD ${KBUILD}
	@echo KERNEL ${KERNEL}
	@echo MODULE ${MODULE}
	@echo KHEADS ${KHEADS}
	@echo SUDO ${SUDO}

.tmp/fstore_dirs:
	mkdir -p .tmp
	mkdir -p ${KBUILD}
	mkdir -p ${MODULE}
	mkdir -p build
	touch $@

${KBUILD}/.config: config/config-6.12.0+ .tmp/fstore_dirs
	cp $< $@
	+yes ""| ${MAKE} -C ${KERNEL} O=${KBUILD} olddefconfig

${KBUILD}/vmlinux: ${KBUILD}/.config
	+${MAKE} -C ${KBUILD}
	touch $@

${KHEADS}: ${KBUILD}/vmlinux
	+${SUDO} ${MAKE} -C ${KBUILD} headers_install ARCH=$(shell uname -m) INSTALL_HDR_PATH=$@
	sudo touch $@

${MODULE}/lib: ${KBUILD}/vmlinux .tmp/fstore_dirs /usr/src/linux-headers-6.12.0+
	+${SUDO} ${MAKE} -C ${KBUILD} INSTALL_MOD_PATH=${MODULE} INSTALL_MOD_STRIP=1 modules_install
	${SUDO} rm ${MODULE}/lib/modules/6.12.0+/build
	${SUDO} ln -s /usr/src/linux-headers-6.12.0+/ ${MODULE}/lib/modules/6.12.0+/build
	${SUDO} touch $@

fstore/fstore.ko: ${KBUILD}/vmlinux ${MODULE}/lib fstore/fstore.c
	+${MAKE} -C fstore

.tmp/fstore_copyin: fstore/fstore.ko
	sudo guestmount -a ubuntu.img -m /dev/sda3 mnt
	sudo cp $< mnt/home/kermit/
	sudo fusermount -u mnt/
	touch $@

.tmp/fstore_test: test/
	sudo guestmount -a ubuntu.img -m /dev/sda3 mnt
	sudo cp -R $< mnt/home/kermit/
	sudo fusermount -u mnt/
	touch $@

load_test: /tmp/fstore_test /tmp/fstore_copyin

clean:
	+${MAKE} -C fstore clean
	sudo fusermount -u mnt/

Clean: clean
	+{MAKE} -C ${KBUILD} clean
