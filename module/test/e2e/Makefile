BUILD ?= build
ROOT_DIR := $(dir $(realpath $(lastword $(MAKEFILE_LIST))))
OUT_DIR := ${BUILD}/$(notdir ${ROOT_DIR:/=})/
EXPORT_BUILD := $(if ${OUT_DIR}, BUILD=${OUT_DIR})
DIRS := $(wildcard ${ROOT_DIR}/*/.)

echo:
	@echo BUILD ${BUILD}
	@echo ROOT_DIR ${ROOT_DIR}
	@echo OUT_DIR ${OUT_DIR}
	@echo EXPORT_BUILD ${EXPORT_BUILD}
	@echo DIRS ${DIRS}
	@$(foreach path,${DIRS},${MAKE} ${EXPORT_BUILD} -C ${path} echo)

test:
	@$(foreach path,${DIRS},${MAKE} ${EXPORT_BUILD} -C ${path} test)
