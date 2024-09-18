SHELL := /bin/bash

USER_PYTHON ?= $(shell readlink -nf $(shell which python))

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

collect-data:
	@python python/data_collection

lint:
	ruff check
	ruff check --select I
	pyright

format:
	ruff check --fix
	ruff check --select I --fix

set-capabilities:
	sudo setcap CAP_BPF,CAP_SYS_ADMIN,CAP_DAC_READ_SEARCH,CAP_SYS_RESOURCE,CAP_NET_ADMIN,CAP_SETPCAP=+eip ${USER_PYTHON}

revoke-capabilities:
	sudo setcap CAP_BPF,CAP_SYS_ADMIN,CAP_DAC_READ_SEARCH,CAP_SYS_RESOURCE,CAP_NET_ADMIN,CAP_SETPCAP=-eip ${USER_PYTHON}

vmlinux-header:
	bpftool btf dump file /sys/kernel/btf/vmlinux format c > python/data_collection/bpf/vmlinux.h
