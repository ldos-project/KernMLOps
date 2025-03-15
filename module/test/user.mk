BUILD ?= build/
ROOT_DIR := $(CURDIR)
OUT_DIR := ${BUILD}/$(notdir ${ROOT_DIR:/=})

TEST_SRC := $(wildcard *.cpp)
BLD_OUT := $(patsubst %.cpp,%,${TEST_SRC})
FULL_OUT := $(addprefix ${OUT_DIR}/,${BLD_OUT})

echo:
	@echo BUILD ${BUILD}
	@echo ROOT_DIR ${ROOT_DIR}
	@echo OUT_DIR ${OUT_DIR}
	@echo TEST_SRC ${TEST_SRC}
	@echo BLD_OUT ${BLD_OUT}
	@echo FULL_OUT ${FULL_OUT}

${OUT_DIR}:
	mkdir -p $@

${FULL_OUT}: ${OUT_DIR}/% : %.cpp ../../fstore/fstore.h | ${OUT_DIR}
	${CXX} -O3 -I/usr/src/linux-headers-$(shell uname -r)/include/ \
		-std=gnu++2b $< -o $@

test: ${FULL_OUT}
	@$(foreach path,${FULL_OUT}, printf "$(shell basename $(path)) ... "; sudo $(path) && printf "pass\n" || printf "fail\n";)
