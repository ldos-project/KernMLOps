obj-m += fstore.o
PWD := $(CURDIR)
KBUILD ?= $(PWD)/../kbuild

all:
	$(MAKE) -C ${KBUILD} M=${PWD} modules

clean:
	$(MAKE) -C ${KBUILD} M=${PWD} clean
