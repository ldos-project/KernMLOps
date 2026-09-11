# Use alongside Makefile until it includes this fragment:
# make -C policy -f Makefile -f original_bcc.mk BUILD=build/container original-bcc-source
# Network access and dependency installation belong only in an ephemeral,
# unprivileged container using the existing tewaro-kernmlops:590b89227 image.
ORIGINAL_BCC_URL := https://git.launchpad.net/ubuntu/+source/bpfcc
ORIGINAL_BCC_TAG := applied/0.29.1+ds-1ubuntu7
ORIGINAL_BCC_COMMIT := 4752b9906825bfc3b86e4590293fc55d7a5db4f8
ORIGINAL_BCC_ROOT := $(abspath $(BUILD)/original_bcc)
ORIGINAL_BCC_SOURCE := $(ORIGINAL_BCC_ROOT)/source
ORIGINAL_BCC_CMAKE := $(ORIGINAL_BCC_ROOT)/cmake
ORIGINAL_BCC_PREFIX := $(ORIGINAL_BCC_ROOT)/install
ORIGINAL_BCC_PATCH := $(abspath bcc_frontend.patch)
ORIGINAL_BCC_JOBS ?= 2

.PHONY: original-bcc-source
original-bcc-source:
	@test "$(abspath $(BUILD))" = "$(CURDIR)/build/container" || { \
		printf 'Use BUILD=build/container for the isolated BCC build.\n' >&2; exit 1; }
	ls -ld "$(BUILD)"
	mkdir -p "$(ORIGINAL_BCC_ROOT)"
	@if test ! -e "$(ORIGINAL_BCC_SOURCE)"; then \
		test -f /.dockerenv || { printf 'Fetch BCC only inside the build container.\n' >&2; exit 1; }; \
		ls -ld "$(ORIGINAL_BCC_ROOT)" && \
		git clone --no-checkout --depth=1 --branch "$(ORIGINAL_BCC_TAG)" \
			"$(ORIGINAL_BCC_URL)" "$(ORIGINAL_BCC_SOURCE)" && \
		test "$$(git -C "$(ORIGINAL_BCC_SOURCE)" rev-parse 'HEAD^{commit}')" = "$(ORIGINAL_BCC_COMMIT)" && \
		git -C "$(ORIGINAL_BCC_SOURCE)" checkout --detach "$(ORIGINAL_BCC_COMMIT)"; \
	fi
	@test "$$(git -C "$(ORIGINAL_BCC_SOURCE)" rev-parse HEAD)" = "$(ORIGINAL_BCC_COMMIT)" || { \
		printf 'BCC source identity mismatch; existing source was not reset.\n' >&2; exit 1; }
	@set -o pipefail; if test ! -e "$(ORIGINAL_BCC_ROOT)/pristine"; then \
		ls -ld "$(ORIGINAL_BCC_ROOT)" && \
		mkdir "$(ORIGINAL_BCC_ROOT)/pristine" && \
		git -C "$(ORIGINAL_BCC_SOURCE)" archive "$(ORIGINAL_BCC_COMMIT)" \
			src/cc/bpf_module.h src/cc/bpf_module.cc src/cc/frontends/clang/loader.cc \
			debian/changelog debian/patches CMakeLists.txt src/cc/CMakeLists.txt \
			| tar -x -C "$(ORIGINAL_BCC_ROOT)/pristine"; \
	fi
	@for file in src/cc/bpf_module.h src/cc/bpf_module.cc src/cc/frontends/clang/loader.cc; do \
		test "$$(git hash-object "$(ORIGINAL_BCC_ROOT)/pristine/$$file")" = \
			"$$(git -C "$(ORIGINAL_BCC_SOURCE)" rev-parse "$(ORIGINAL_BCC_COMMIT):$$file")" || { \
			printf 'Pristine BCC file differs from the pinned commit: %s\n' "$$file" >&2; exit 1; }; \
	done
	@set -o pipefail; git -C "$(ORIGINAL_BCC_SOURCE)" show -s --format=fuller "$(ORIGINAL_BCC_COMMIT)" \
		| tee "$(ORIGINAL_BCC_ROOT)/source-identity.txt"
	@test "$$(dpkg-parsechangelog -l "$(ORIGINAL_BCC_SOURCE)/debian/changelog" -S Version)" = "0.29.1+ds-1ubuntu7"

.PHONY: original-bcc-patch original-bcc-build original-bcc-deps
original-bcc-patch: original-bcc-source $(ORIGINAL_BCC_PATCH)
	@if git -C "$(ORIGINAL_BCC_SOURCE)" apply --reverse --check "$(ORIGINAL_BCC_PATCH)" 2>/dev/null; then \
		printf 'BCC frontend patch already applied.\n'; \
	elif git -C "$(ORIGINAL_BCC_SOURCE)" apply --check "$(ORIGINAL_BCC_PATCH)"; then \
		git -C "$(ORIGINAL_BCC_SOURCE)" apply "$(ORIGINAL_BCC_PATCH)"; \
	else \
		printf 'BCC patch conflicts with existing edits; source was not reset.\n' >&2; exit 1; \
	fi
	git -C "$(ORIGINAL_BCC_SOURCE)" diff --check
	git -C "$(ORIGINAL_BCC_SOURCE)" diff -- src/cc/bpf_module.h src/cc/bpf_module.cc \
		src/cc/frontends/clang/loader.cc > "$(ORIGINAL_BCC_ROOT)/frontend.diff"

# Install these only in the disposable container, never on the host or in an image.
# Build as the workspace owner afterwards, in that same container.
original-bcc-deps:
	@test -f /.dockerenv && test "$$(id -u)" = 0 || { \
		printf 'Dependency installation requires root inside an ephemeral container.\n' >&2; exit 1; }
	ls -ld /var/lib/apt/lists /var/cache/apt/archives /etc/apt/sources.list.d
	apt-get -o Dir::Etc::sourcelist=sources.list.d/ubuntu.sources -o Dir::Etc::sourceparts=- update
	DEBIAN_FRONTEND=noninteractive apt-get \
		-o Dir::Etc::sourcelist=sources.list.d/ubuntu.sources -o Dir::Etc::sourceparts=- \
		install -y --no-install-recommends --no-upgrade \
		libclang-18-dev=1:18.1.3-1ubuntu1 libclang-cpp18-dev=1:18.1.3-1ubuntu1

original-bcc-build: original-bcc-patch
	@test -s "$(abspath bcc_reduce.hpp)" || { \
		printf 'Missing policy/bcc_reduce.hpp: the reducer owner must supply it before building.\n' >&2; exit 1; }
	@test -f /.dockerenv && test "$$(id -u)" != 0 || { \
		printf 'Build BCC as the workspace owner inside the unprivileged container.\n' >&2; exit 1; }
	@test "$$($(LLVM_CONFIG) --version)" = 18.1.3
	@test "$$(dpkg-query -W -f='$${Version}' libbpfcc)" = "0.29.1+ds-1ubuntu7"
	@test -f /usr/lib/llvm-18/include/clang/Frontend/CompilerInstance.h && \
		test -f /usr/lib/llvm-18/lib/libclang-cpp.so || { \
		printf 'Run original-bcc-deps in this ephemeral container first.\n' >&2; exit 1; }
	ls -ld "$(ORIGINAL_BCC_ROOT)"
	mkdir -p "$(ORIGINAL_BCC_CMAKE)" "$(ORIGINAL_BCC_PREFIX)"
	# Prevent upstream CMake from writing the user global git configuration.
	env -u DESTDIR GIT_CONFIG_GLOBAL=/dev/null PYTHONDONTWRITEBYTECODE=1 \
		cmake -S "$(ORIGINAL_BCC_SOURCE)" -B "$(ORIGINAL_BCC_CMAKE)" \
		-DCMAKE_BUILD_TYPE=Release -DCMAKE_C_COMPILER=/usr/bin/clang-18 \
		-DCMAKE_CXX_COMPILER=/usr/bin/clang++-18 \
		-DCMAKE_CXX_FLAGS="-I$(abspath .)" \
		-DLLVM_DIR=/usr/lib/llvm-18/lib/cmake/llvm \
		-DENABLE_LLVM_SHARED=ON -Dlibclang-shared=/usr/lib/llvm-18/lib/libclang-cpp.so \
		-DCMAKE_USE_LIBBPF_PACKAGE=ON -DNO_BLAZESYM=ON \
		-DENABLE_TESTS=OFF -DENABLE_EXAMPLES=OFF -DENABLE_MAN=OFF \
		-DENABLE_LLVM_NATIVECODEGEN=OFF -DCMAKE_DISABLE_FIND_PACKAGE_LuaJIT=ON \
		-DREVISION=0.29.1 -DREVISION_LAST=0.29.1 \
		-DCMAKE_INSTALL_PREFIX="$(ORIGINAL_BCC_PREFIX)" -DCMAKE_INSTALL_LIBDIR=lib \
		-DPYTHON_CMD=/usr/bin/python3 -DPYTHON_PREFIX="$(ORIGINAL_BCC_PREFIX)" \
		-DPY_SKIP_DEB_LAYOUT=ON \
		-DPYTHON_FLAGS="--single-version-externally-managed --install-lib=$(ORIGINAL_BCC_PREFIX)/lib/python3/site-packages" \
		-DCMAKE_EXPORT_COMPILE_COMMANDS=ON
	env -u DESTDIR GIT_CONFIG_GLOBAL=/dev/null PYTHONDONTWRITEBYTECODE=1 \
		cmake --build "$(ORIGINAL_BCC_CMAKE)" --parallel "$(ORIGINAL_BCC_JOBS)"
	env -u DESTDIR PYTHONDONTWRITEBYTECODE=1 cmake --install "$(ORIGINAL_BCC_CMAKE)"
	# Import only: do not construct BPF, create maps, or load programs here.
	env LD_LIBRARY_PATH="$(ORIGINAL_BCC_PREFIX)/lib" \
		PYTHONPATH="$(ORIGINAL_BCC_PREFIX)/lib/python3/site-packages" PYTHONDONTWRITEBYTECODE=1 \
		/usr/bin/python3 -c 'import bcc; from bcc.libbcc import lib; assert bcc.__file__.startswith("$(ORIGINAL_BCC_PREFIX)/"); assert not lib.bpf_module_rw_engine_enabled(); print(bcc.__file__)'
	sha256sum "$(ORIGINAL_BCC_PATCH)" "$(abspath bcc_reduce.hpp)" \
		"$(abspath original_reduce.cpp)" "$(abspath original_reduce.hpp)" \
		"$(ORIGINAL_BCC_PREFIX)/lib/libbcc.so.0" > "$(ORIGINAL_BCC_ROOT)/build.sha256"

ORIGINAL_BCC_EVIDENCE ?= $(ORIGINAL_BCC_ROOT)/evidence/frontend-v1
.PHONY: original-bcc-export
original-bcc-export:
	@test -f "$(ORIGINAL_BCC_PREFIX)/lib/libbcc.so.0"
	ls -ld "$(ORIGINAL_BCC_ROOT)"
	mkdir -p "$(ORIGINAL_BCC_EVIDENCE)"
	@set -e; for stream in rss perf; do for transport in capture mpmo; do \
		env LD_LIBRARY_PATH="$(ORIGINAL_BCC_PREFIX)/lib" \
			PYTHONPATH="$(ORIGINAL_BCC_PREFIX)/lib/python3/site-packages:$(abspath ../python/kernmlops)" \
			PYTHONDONTWRITEBYTECODE=1 $(PYTHON) inspect_original.py \
			--stream $$stream --transport $$transport \
			--prefix "$(ORIGINAL_BCC_EVIDENCE)/$$stream-$$transport"; \
	done; done

$(ORIGINAL_BCC_ROOT)/liboriginal_queue.so: original_queue.c
	ls -ld "$(ORIGINAL_BCC_ROOT)"
	$(CC) -O2 -shared -fPIC -std=c11 -Wall -Wextra -Werror $< -o "$@"

ORIGINAL_BCC_BUNDLE_EVIDENCE ?= $(ORIGINAL_BCC_ROOT)/evidence/bundle-v1
.PHONY: original-bcc-bundle
original-bcc-bundle: $(ORIGINAL_BCC_ROOT)/liboriginal_queue.so
	@test -f "$(ORIGINAL_BCC_PREFIX)/lib/libbcc.so.0"
	$(PYTHON) original_manifest.py --directory "$(ORIGINAL_BCC_ROOT)"
	ls -ld "$(ORIGINAL_BCC_ROOT)"
	mkdir -p "$(ORIGINAL_BCC_BUNDLE_EVIDENCE)"
	@set -e; for stream in rss perf; do for transport in capture mpmo; do \
		env LD_LIBRARY_PATH="$(ORIGINAL_BCC_PREFIX)/lib" \
			PYTHONPATH="$(ORIGINAL_BCC_PREFIX)/lib/python3/site-packages:$(abspath ../python/kernmlops)" \
			PYTHONDONTWRITEBYTECODE=1 $(PYTHON) inspect_original.py \
			--stream $$stream --transport $$transport \
			--selection "$(ORIGINAL_BCC_ROOT)/$$stream.selection.json" \
			--prefix "$(ORIGINAL_BCC_BUNDLE_EVIDENCE)/$$stream-$$transport"; \
	done; done
	$(MAKE) BUILD="$(BUILD)" original-bcc-manifest

.PHONY: original-bcc-manifest
original-bcc-manifest:
	$(PYTHON) original_manifest.py --directory "$(ORIGINAL_BCC_ROOT)" --reports "$(ORIGINAL_BCC_BUNDLE_EVIDENCE)"
