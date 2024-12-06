ADDBUILD := $(if ${BUILD}, BUILD=${BUILD})
echo:
	@echo BUILD ${BUILD}
	@echo kdev:
	@${MAKE} ${ADDBUILD} -C kdev echo
	@echo unit:
	@${MAKE} ${ADDBUILD} -C unit echo
	@echo e2e:
	@${MAKE} ${ADDBUILD} -C e2e echo


kdev_test:
	@${MAKE} ${ADDBUILD} -C kdev test

unit_test:
	@${MAKE} ${ADDBUILD} -C unit test

e2e_test:
	@${MAKE} ${ADDBUILD} -C e2e test

test: kdev_test unit_test
