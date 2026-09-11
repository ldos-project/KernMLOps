SHELL := /bin/bash
.DEFAULT_GOAL := all
.DELETE_ON_ERROR:

LLVM_CONFIG ?= llvm-config-18
LLVM_BINDIR = $(shell $(LLVM_CONFIG) --bindir)
CLANG = $(LLVM_BINDIR)/clang
CXX = $(LLVM_BINDIR)/clang++
OPT = $(LLVM_BINDIR)/opt
OBJDUMP = $(LLVM_BINDIR)/llvm-objdump
PYTHON ?= python3
BUILD ?= build
SOURCE ?= bpf/syscall.bpf.c
PROGRAM ?= capture_syscall
EVENT ?= policy_event
FIELDS ?= syscall_nr,doubled_syscall_nr
EXTRACT_FLAGS ?=
MULTIARCH := $(shell $(CC) -dumpmachine)
BPF_CFLAGS := -target bpf -g -O1 -I/usr/include/$(MULTIARCH)

RUNNER ?=
TARGET ?= extract

.PHONY: all doctor frontend extract test prepare-load run-load test-load inspect image container container-test container-test-load
all: extract

$(BUILD):
	ls -d .
	mkdir -p "$@"

$(BUILD)/field-extract: extract.cpp | $(BUILD)
	@test "$$($(LLVM_CONFIG) --version | cut -d. -f1)" = 18 || { printf 'LLVM 18 development tools are required; use make container.\n'; exit 1; }
	$(CXX) $$($(LLVM_CONFIG) --cxxflags) -std=c++17 -O2 -Wall -Wextra -Wno-unused-parameter \
		$< -o $@ $$($(LLVM_CONFIG) --ldflags --link-shared --libs core irreader bitwriter passes --system-libs)

frontend: | $(BUILD)
	$(CLANG) $(BPF_CFLAGS) -Xclang -disable-llvm-passes -emit-llvm -c "$(SOURCE)" -o $(BUILD)/input.bc
	$(LLVM_BINDIR)/llvm-dis $(BUILD)/input.bc -o $(BUILD)/input.ll

# Source and field selections are runtime Make inputs, so never reuse stale output.
extract: $(BUILD)/field-extract frontend
	rm -f $(BUILD)/selected.bc $(BUILD)/selected.bpf.o $(BUILD)/selected.schema.json
	$(BUILD)/field-extract --input $(BUILD)/input.bc --output $(BUILD)/selected.bc \
		--schema $(BUILD)/selected.schema.json --program "$(PROGRAM)" --event "$(EVENT)" --fields "$(FIELDS)" $(EXTRACT_FLAGS)
	$(LLVM_BINDIR)/llvm-dis $(BUILD)/selected.bc -o $(BUILD)/selected.ll
	$(CLANG) -target bpf -g -O2 -c $(BUILD)/selected.bc -o $(BUILD)/selected.bpf.o

test: $(BUILD)/field-extract
	$(PYTHON) tests/test_extract.py --build "$(BUILD)" --clang "$(CLANG)" \
		--objdump "$(OBJDUMP)" --llvm-dis "$(LLVM_BINDIR)/llvm-dis" --multiarch "$(MULTIARCH)" -v

$(BUILD)/load: tests/load.c | $(BUILD)
	$(CC) -std=c11 -O2 -g -Wall -Wextra -Werror $< -o $@ -lbpf -lelf -lz

prepare-load: $(BUILD)/load
	$(MAKE) extract SOURCE=bpf/syscall.bpf.c PROGRAM=capture_syscall EVENT=policy_event FIELDS=syscall_nr,doubled_syscall_nr

run-load:
	$(RUNNER) $(BUILD)/load $(BUILD)/selected.bpf.o

test-load: prepare-load
	$(MAKE) run-load

inspect: extract
	$(OBJDUMP) -dr --section-headers $(BUILD)/selected.bpf.o

image:
	$(MAKE) -C .. docker-image

container: image
	$(MAKE) -C .. docker INTERACTIVE=i CONTAINER_HOST_HEADERS=0 CONTAINER_PRIVILEGED=0 \
		CONTAINER_OPTS="--user $$(id -u):$$(id -g)" \
		CONTAINER_CMD='make -C policy BUILD=build/container SOURCE="$(SOURCE)" PROGRAM="$(PROGRAM)" EVENT="$(EVENT)" FIELDS="$(FIELDS)" EXTRACT_FLAGS="$(EXTRACT_FLAGS)" $(TARGET)'

container-test:
	$(MAKE) container TARGET=test

container-test-load:
	$(MAKE) container TARGET=prepare-load
	$(MAKE) -C .. docker INTERACTIVE=i CONTAINER_HOST_HEADERS=0 CONTAINER_SOURCE_MODE=ro \
		CONTAINER_OPTS="--network none --read-only --ulimit memlock=-1:-1" \
		CONTAINER_CMD='make -C policy BUILD=build/container run-load'

doctor:
	$(LLVM_CONFIG) --version
	$(CXX) $$($(LLVM_CONFIG) --cxxflags) -fsyntax-only -x c++ -include llvm/IR/Module.h /dev/null
	$(CLANG) -fsyntax-only -x c -include linux/bpf.h -include bpf/bpf_helpers.h /dev/null
	$(CC) -fsyntax-only -x c -include bpf/libbpf.h /dev/null

.PHONY: test-block-io prepare-block-io run-block-io container-test-block-io container-test-block-io-load
test-block-io: $(BUILD)/field-extract
	$(PYTHON) tests/test_block_io.py --build "$(BUILD)" --clang "$(CLANG)" \
		--objdump "$(OBJDUMP)" --llvm-dis "$(LLVM_BINDIR)/llvm-dis" --multiarch "$(MULTIARCH)" -v

$(BUILD)/block-io-load: tests/block_io_load.c | $(BUILD)
	$(CC) -std=c11 -O2 -g -Wall -Wextra -Werror $< -o $@ -lbpf -lelf -lz

prepare-block-io: test-block-io $(BUILD)/block-io-load
	ls -d "$(BUILD)/block_io"
	mkdir -p "$(BUILD)/block_io/workload"

BLOCK_IO_WORKDIR ?= $(BUILD)/block_io/workload
run-block-io:
	$(RUNNER) $(BUILD)/block-io-load $(BUILD)/block_io/latency.bpf.o "$(BLOCK_IO_WORKDIR)"

container-test-block-io:
	$(MAKE) container TARGET=test-block-io

# Requires explicit permission: attaches host block tracepoints and writes one
# temporary 8 MiB file on the separately mounted, disposable workload directory.
container-test-block-io-load:
	$(MAKE) container TARGET=prepare-block-io
	$(MAKE) -C .. docker INTERACTIVE=i CONTAINER_HOST_HEADERS=0 CONTAINER_SOURCE_MODE=ro \
		CONTAINER_OPTS="--network none --read-only --ulimit memlock=-1:-1" \
		KERNMLOPS_CONTAINER_MOUNTS='-v $(CURDIR)/build/container/block_io/workload:/policy-io:rw' \
		CONTAINER_CMD='make -C policy BUILD=build/container BLOCK_IO_WORKDIR=/policy-io run-block-io'

.PHONY: test-ycsb-results
test-ycsb-results:
	env PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=../python/kernmlops:../python/tests \
		$(PYTHON) -m unittest test_ycsb test_redis_results test_collect_results -v

.PHONY: test-ycsb-ttest
test-ycsb-ttest:
	env PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. \
		$(PYTHON) -m unittest discover -s tests -p test_ycsb_ttest.py -v

$(BUILD)/check-dtlb: check_dtlb.c | $(BUILD)
	$(CC) -O2 -Wall -Wextra -Werror $< -o $@

.PHONY: check-dtlb
check-dtlb: $(BUILD)/check-dtlb
	$(BUILD)/check-dtlb

.PHONY: rss-tlb test-rss-tlb matrix-prepare matrix-run matrix-status
$(BUILD)/rss_tlb: | $(BUILD)
	ls -d "$(BUILD)"
	mkdir -p "$@"

$(BUILD)/rss_tlb/librss_tlb.so: rss_tlb_native.c | $(BUILD)/rss_tlb
	$(CC) -O2 -g -shared -fPIC -std=c11 -Wall -Wextra -Werror $< -o $@ -lbpf -lelf -lz

rss-tlb: $(BUILD)/field-extract $(BUILD)/rss_tlb/librss_tlb.so
	$(CLANG) $(BPF_CFLAGS) -Xclang -disable-llvm-passes -emit-llvm -c bpf/rss.bpf.c -o $(BUILD)/rss_tlb/rss.full.bc
	$(BUILD)/field-extract --input $(BUILD)/rss_tlb/rss.full.bc --output $(BUILD)/rss_tlb/rss.selected.bc \
		--schema $(BUILD)/rss_tlb/rss.schema.json --program rss_emit --event rss_event \
		--fields timestamp_ns,count_pages,tgid,member --preserve-other-programs
	$(CLANG) -target bpf -g -O2 -c $(BUILD)/rss_tlb/rss.selected.bc -o $(BUILD)/rss_tlb/rss.selected.bpf.o
	$(CLANG) $(BPF_CFLAGS) -Xclang -disable-llvm-passes -emit-llvm -c bpf/dtlb.bpf.c -o $(BUILD)/rss_tlb/dtlb.full.bc
	$(BUILD)/field-extract --input $(BUILD)/rss_tlb/dtlb.full.bc --output $(BUILD)/rss_tlb/dtlb.selected.bc \
		--schema $(BUILD)/rss_tlb/dtlb.schema.json --program dtlb_sample --event dtlb_event \
		--fields timestamp_ns,misses,accesses,miss_enabled_ns,miss_running_ns,access_enabled_ns,access_running_ns,cpu
	$(CLANG) -target bpf -g -O2 -c $(BUILD)/rss_tlb/dtlb.selected.bc -o $(BUILD)/rss_tlb/dtlb.selected.bpf.o
	$(PYTHON) rss_tlb_manifest.py $(BUILD)/rss_tlb

MATRIX_DIR ?= data/experiments/redis-rss-dtlb
MATRIX_REPEATS ?= 30
MATRIX_SEED ?= 17
MATRIX_PLAN_ARGS ?=
matrix-prepare:
	env PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=../python/kernmlops $(PYTHON) run_ycsb_matrix.py \
		--directory "$(MATRIX_DIR)" --repeats $(MATRIX_REPEATS) --seed $(MATRIX_SEED) --limit 0 $(MATRIX_PLAN_ARGS)

matrix-run:
	env PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=../python/kernmlops $(PYTHON) run_ycsb_matrix.py \
		--directory "$(MATRIX_DIR)" --repeats $(MATRIX_REPEATS) --seed $(MATRIX_SEED) --resume $(MATRIX_PLAN_ARGS) $(MATRIX_ARGS)

matrix-status:
	$(PYTHON) -m json.tool ../$(MATRIX_DIR)/manifest.json

test-rss-tlb:
	env PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=../python/kernmlops:. $(PYTHON) -m unittest \
		discover -s tests -p test_rss_tlb.py -v

.PHONY: test-ycsb-matrix summarize-ycsb-matrix
test-ycsb-matrix:
	env PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=../python/kernmlops:.:tests $(PYTHON) -m unittest \
		test_ycsb_matrix test_ycsb_matrix_summary -v

summarize-ycsb-matrix:
	env PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=../python/kernmlops:. $(PYTHON) summarize_ycsb_matrix.py --directory $(MATRIX_DIR) $(SUMMARY_ARGS)

include original_bcc.mk
include original_reduce.mk

.PHONY: test-original-collector test-original-matrix compare-original-matrix
test-original-collector:
	env PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=../python/kernmlops $(PYTHON) -m unittest \
		discover -s tests -p test_original_collector.py -v

test-original-matrix:
	env PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=../python/kernmlops:.:tests $(PYTHON) -m unittest \
		test_original_matrix -v

compare-original-matrix:
	env PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=../python/kernmlops:. $(PYTHON) \
		compare_original_matrix.py --directory "$(MATRIX_DIR)" $(SUMMARY_ARGS)

.PHONY: plot-never-overall test-never-overall
plot-never-overall:
	env PYTHONDONTWRITEBYTECODE=1 $(PYTHON) plot_never_overall.py --directory "$(MATRIX_DIR)/analysis"

test-never-overall:
	env PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. $(PYTHON) -m unittest \
		discover -s tests -p test_never_overall.py -v

ORIGINAL_TRANSPORT ?= capture
.PHONY: preflight-original
preflight-original:
	env PYTHONDONTWRITEBYTECODE=1 LD_LIBRARY_PATH="$(ORIGINAL_BCC_PREFIX)/lib" \
		PYTHONPATH="$(ORIGINAL_BCC_PREFIX)/lib/python3/site-packages:$(abspath ../python/kernmlops)" \
		$(PYTHON) preflight_rss_tlb.py --original $(ORIGINAL_TRANSPORT) \
		--manifest "$(ORIGINAL_BCC_ROOT)/manifest.json"

.PHONY: preflight-selected-control
preflight-selected-control:
	env PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=../python/kernmlops \
		$(PYTHON) preflight_rss_tlb.py --selected-control

BLOAT_DIR ?= data/derived/bloat/full-probes-20260908
.PHONY: bloat-audit bloat-output-dir
bloat-output-dir:
	ls -ld ../data
	mkdir -p ../data/derived/bloat

bloat-audit:
	env PYTHONDONTWRITEBYTECODE=1 $(PYTHON) bloat/audit.py --output "$(BLOAT_DIR)/audit" $(BLOAT_ARGS)

.PHONY: bloat-correlate bloat-sensitivity test-bloat
bloat-correlate:
	env PYTHONDONTWRITEBYTECODE=1 $(PYTHON) bloat/analyze.py \
		--audit-directory "$(BLOAT_DIR)/audit" --output "$(BLOAT_DIR)/correlations" $(BLOAT_ARGS)

bloat-sensitivity:
	env PYTHONDONTWRITEBYTECODE=1 $(PYTHON) bloat/sensitivity.py \
		--input-dir "$(BLOAT_DIR)/correlations" --output "$(BLOAT_DIR)/phase-sensitivity" $(BLOAT_ARGS)

test-bloat:
	env PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=bloat:tests $(PYTHON) -m unittest \
		discover -s tests -p 'test_bloat*.py' -v
