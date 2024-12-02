
KBUILD ?= $(shell pwd)/kbuild
KERNEL ?= $(shell pwd)/linux
MODULE ?= $(shell pwd)/kmod

SUDO ?= ""

all: fstore/fstore.ko

echo:
	echo ${KBUILD}
	echo ${KERNEL}
	echo ${MODULE}
	echo ${SUDO}

.tmp/fstore_dirs:
	mkdir -p .tmp
	mkdir -p ${KBUILD}
	mkdir -p ${MODULE}
	touch $@

${KBUILD}/.config: config/config-6.12.0+
	cp $< $@
	+yes ""| ${MAKE} -C ${KERNEL} O=${KBUILD} olddefconfig

${KBUILD}/vmlinux: ${KBUILD}/.config
	+${MAKE} -C ${KBUILD}

${MODULE}/lib: ${KBUILD}/vmlinux .tmp/fstore_dirs
	+${SUDO} ${MAKE} -C ${KBUILD} INSTALL_MOD_PATH=${MODULE} INSTALL_MOD_STRIP=1 modules_install

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
