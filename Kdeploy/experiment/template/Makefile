
obj-m += model_[tmp].o
CFLAGS_model_[tmp].o += -mhard-float

ccflags-y += -std=gnu99

PWD := $(CURDIR)

all:
	$(MAKE) -C /lib/modules/$(shell uname -r)/build M=$(PWD) modules

clean:
	$(MAKE) -C /lib/modules/$(shell uname -r)/build M=$(PWD) clean
