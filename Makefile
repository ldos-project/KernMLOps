SHELL := /bin/bash

UNAME ?= $(shell whoami)
UID ?= $(shell id -u)
GID ?= $(shell id -g)
USER_PYTHON ?= $(shell readlink -nf $(shell which python))
KERNEL_VERSION ?= $(shell uname -r)

# Container variables
KERNEL_DEV_HEADERS_DIR ?= /usr/src/kernels/${KERNEL_VERSION}
KERNEL_DEV_SPECIFIC_HEADERS_MOUNT ?=
KERNEL_DEV_MODULES_DIR ?= /lib/modules/${KERNEL_VERSION}

# Kernel dev header locations are different on red hat and ubuntu
# First ensure the user has not overridden this field
ifeq (${KERNEL_DEV_HEADERS_DIR}, /usr/src/kernels/${KERNEL_VERSION})
	ifeq ("$(wildcard ${KERNEL_DEV_HEADERS_DIR})","")
	    KERNEL_DEV_HEADERS_DIR = /usr/src/linux-headers-${KERNEL_VERSION}
	endif
endif

# Ubuntu distributions have generic and specific headers that are needed
# First ensure the user has not overridden this field
ifeq (${KERNEL_DEV_SPECIFIC_HEADERS_MOUNT},)
# Now check if the headers are the "generic" version
	ifneq ("$(shell echo ${KERNEL_DEV_HEADERS_DIR} | grep generic)","")
			KERNEL_DEV_SPECIFIC_HEADERS_DIR = /usr/src/linux-headers-$(shell echo "${KERNEL_VERSION}" | sed 's|\(.*\)-.*|\1|')
			KERNEL_DEV_SPECIFIC_HEADERS_MOUNT = -v ${KERNEL_DEV_SPECIFIC_HEADERS_DIR}/:${KERNEL_DEV_SPECIFIC_HEADERS_DIR}:ro
	endif
endif

BASE_IMAGE_NAME ?= kernmlops
BCC_IMAGE_NAME ?= ${BASE_IMAGE_NAME}-bcc
IMAGE_NAME ?= ${UNAME}-${BASE_IMAGE_NAME}
SRC_DIR ?= $(shell pwd)
VERSION ?= $(shell git log --pretty="%h" -1 Dockerfile.dev requirements.txt)

CONTAINER_SRC_DIR ?= /KernMLOps
CONTAINER_WORKDIR ?= ${CONTAINER_SRC_DIR}
CONTAINER_HOSTNAME ?= $(shell hostname)-docker
CONTAINER_CONTEXT ?= default
CONTAINER_CPUSET ?=
CONTAINER_CMD ?= bash -l
INTERACTIVE ?= i

# Benchmarking variables
COLLECTION_BENCHMARK ?= faux
COLLECTION_POLL_RATE ?= .5
BENCHMARK_DIR ?= /home/${UNAME}/kernmlops-benchmark

# Provisioning variables
PROVISIONING_USER ?= ${UNAME}
PROVISIONING_HOST ?= localhost
PROVISIONING_PORT ?= 22
PROVISIONING_TARGET ?= ${PROVISIONING_HOST}:${PROVISIONING_PORT}

# Developer variables that should be set as env vars in startup files like .profile
KERNMLOPS_CONTAINER_MOUNTS ?=
KERNMLOPS_CONTAINER_ENV ?=


# Dependency and code quality commands
dependencies: dependencies-asdf

dependencies-asdf:
	@echo "Updating asdf plugins..."
	@asdf plugin update --all >/dev/null 2>&1 || true
	@echo "Adding new asdf plugins..."
	@awk '$$1 != "#" {print $$1}' ./.tool-versions | xargs -I % asdf plugin-add % >/dev/null 2>&1 || true
	@echo "Installing asdf tools..."
	@awk '$$1 != "#" {print $$1}' ./.tool-versions | xargs -I{} bash -c 'asdf install {}'
	@echo "Updating local environment to use proper tool versions..."
	@awk '$$1 != "#" {print $$1, $$2}' ./.tool-versions | xargs -I{} bash -c 'asdf local {}'
	@asdf reshim
	@echo "Done!"

hooks:
	@pre-commit install --hook-type pre-commit
	@pre-commit install-hooks

pre-commit:
	@pre-commit run -a

lint:
	ruff check
	ruff check --select I
	pyright

format:
	ruff check --fix
	ruff check --select I --fix


# Python commands
collect:
	@${MAKE} -e CONTAINER_CMD="bash -lc 'make benchmark-linux-build'" docker

collect-data:
	@python python/kernmlops collect -v \
	-p ${COLLECTION_POLL_RATE} \
	-d ${BENCHMARK_DIR} \
	--benchmark ${COLLECTION_BENCHMARK}

benchmark-linux-build:
	@python python/kernmlops collect -v \
	-p ${COLLECTION_POLL_RATE} \
	-d ${BENCHMARK_DIR} \
	--benchmark linux-build

benchmark-linux-build-baseline:
	@python python/kernmlops collect -v \
	-p ${COLLECTION_POLL_RATE} \
	-d ${BENCHMARK_DIR} \
	--no-hooks \
	--benchmark linux-build

dump:
	@python python/kernmlops collect dump


# Provisioning commands
provision-benchmarks:
	@echo "[install-benchmarks]" > hosts
	@echo "${PROVISIONING_TARGET}" >> hosts
	@ansible-playbook benchmark/provisioning/site.yml -u ${PROVISIONING_USER} -i ./hosts -e benchmark_dir=${BENCHMARK_DIR} \
	|| echo "Ensure that 'make provision-benchmarks-admin' has been run by your system administrator"

provision-benchmarks-admin:
	@echo "[install-benchmarks]" > hosts
	@echo "${PROVISIONING_TARGET}" >> hosts
	ansible-playbook benchmark/provisioning/site.yml -u ${PROVISIONING_USER} -i ./hosts -e benchmark_dir=${BENCHMARK_DIR} -K

provision-development:
	@echo "[development]" > hosts
	@echo "${PROVISIONING_TARGET}" >> hosts
	ansible-playbook benchmark/provisioning/site.yml -u ${PROVISIONING_USER} -i ./hosts -K

# Docker commands
docker-image:
	@${MAKE} docker-image-dependencies
	docker --context ${CONTAINER_CONTEXT} build \
	--build-arg BUILD_IMAGE=${BCC_IMAGE_NAME}:latest \
	--build-arg SRC_DIR=${CONTAINER_SRC_DIR} \
	--build-arg UNAME=${UNAME} \
	--build-arg IS_CI=false \
	--build-arg UID=${UID} \
	--build-arg GID=${GID} \
	-t ${IMAGE_NAME}:${VERSION} \
	--file Dockerfile.dev \
	--target dev .

docker-image-dependencies:
	@mkdir -p data/curated
	docker --context ${CONTAINER_CONTEXT} build \
	-t ${BCC_IMAGE_NAME}:latest \
	--file Dockerfile.dev \
	--target bcc .

docker:
	@if [ ! -d "${KERNEL_DEV_HEADERS_DIR}" ]; then \
		echo "Kernel dev headers not installed: ${KERNEL_DEV_HEADERS_DIR}" && exit 1; \
	fi
	@if [ ! -d "${KERNEL_DEV_MODULES_DIR}" ]; then \
		echo "Kernel dev headers not installed: ${KERNEL_DEV_MODULES_DIR}" && exit 1; \
	fi
	@if [ ! -d "${BENCHMARK_DIR}" ]; then \
		echo "User benchmarks not installed: ${BENCHMARK_DIR}" \
		&& echo "Try 'make provision-benchmarks'" && exit 1; \
	fi
	@docker --context ${CONTAINER_CONTEXT} run --rm \
	-v ${SRC_DIR}/:${CONTAINER_SRC_DIR} \
	-v ${KERNEL_DEV_HEADERS_DIR}/:${KERNEL_DEV_HEADERS_DIR}:ro \
	-v ${KERNEL_DEV_MODULES_DIR}/:${KERNEL_DEV_MODULES_DIR}:ro \
	-v ${BENCHMARK_DIR}/:/home/${UNAME}/kernmlops-benchmark \
	-v ${BENCHMARK_DIR}/:${BENCHMARK_DIR} \
	-v /sys/kernel/:/sys/kernel \
	${KERNEL_DEV_SPECIFIC_HEADERS_MOUNT} \
	${KERNMLOPS_CONTAINER_MOUNTS} \
	${KERNMLOPS_CONTAINER_ENV} \
	${CONTAINER_CPUSET} \
	--pid=host \
	--privileged \
	--hostname=${CONTAINER_HOSTNAME} \
	--workdir=${CONTAINER_WORKDIR} ${CONTAINER_OPTS} -${INTERACTIVE}t \
	${IMAGE_NAME}:${VERSION} \
	${CONTAINER_CMD} || true


# Miscellaneous commands

clean-docker-images:
	docker --context ${CONTAINER_CONTEXT} image list \
	--filter "label=creator=${UNAME}" \
	--filter "label=project=KernMLOps" \
	--format "table {{.Repository}}:{{.Tag}}" | tail -n +2 | uniq \
	| xargs -I % docker --context ${CONTAINER_CONTEXT} rmi %

set-capabilities:
	sudo setcap CAP_BPF,CAP_SYS_ADMIN,CAP_DAC_READ_SEARCH,CAP_SYS_RESOURCE,CAP_NET_ADMIN,CAP_SETPCAP,CAP_PERFMON=+eip ${USER_PYTHON}

revoke-capabilities:
	sudo setcap CAP_BPF,CAP_SYS_ADMIN,CAP_DAC_READ_SEARCH,CAP_SYS_RESOURCE,CAP_NET_ADMIN,CAP_SETPCAP,CAP_PERFMON=-eip ${USER_PYTHON}

vmlinux-header:
	bpftool btf dump file /sys/kernel/btf/vmlinux format c > python/data_collection/bpf/vmlinux.h

clear-page-cache:
	sync && echo 3 > /proc/sys/vm/drop_caches
