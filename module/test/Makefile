BUILD ?= build
ROOT_DIR := $(CURDIR)
OUT_DIR := ${BUILD}/$(notdir ${ROOT_DIR:/=})

EXPORT_VARS := $(if ${BUILD}, BUILD=${OUT_DIR}) KBUILD=${KBUILD}
echo:
	@echo BUILD ${BUILD}
	@echo ROOT_DIR ${ROOT_DIR}
	@echo OUT_DIR ${OUT_DIR}
	@echo kdev:
	@${MAKE} ${EXPORT_VARS} -C kdev echo
	@echo unit:
	@${MAKE} ${EXPORT_VARS} -C unit echo
	@echo e2e:
	@${MAKE} ${EXPORT_VARS} -C e2e echo


kdev_test:
	@${MAKE} ${EXPORT_VARS} -C kdev test

unit_test:
	@${MAKE} ${EXPORT_VARS} -C unit test

e2e_test:
	@${MAKE} ${EXPORT_VARS} -C e2e test

undeploy:
	+${MAKE} -C e2e undeploy

test: kdev_test unit_test
