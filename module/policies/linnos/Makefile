
default: linnos-inference-capture

linnos-inference-capture.o: linnos-inference-capture.bpf.c
	clang -O2 -g -target bpf -c $^ -o $@ \
		-D__TARGET_ARCH_x86_64 \
		-I/usr/include/ \
  	-I/usr/src/linux-headers-$(shell uname -r)/include \
  	-I/usr/src/linux-headers-$(shell uname -r)/arch/x86/include \
  	-I/usr/src/linux-headers-$(shell uname -r)/arch/x86/include/generated/

linnos-inference-capture: linnos-inference-capture.cpp linnos-inference-capture.o
	g++ -o $@ $< -lbpf -I../../fstore/
