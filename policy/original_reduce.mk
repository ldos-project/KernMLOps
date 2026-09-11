# Offline only; consumes frozen original BCC exports, never compiles new probes.
ORIGINAL_REDUCE_ROOT := $(abspath $(BUILD)/original_reduce)
ORIGINAL_REDUCE_FIXTURES ?= $(abspath $(BUILD)/original_bcc/evidence/frontend-v4)

$(BUILD)/original-reduce: original_reduce.cpp original_reduce.hpp | $(BUILD)
	$(CXX) $$($(LLVM_CONFIG) --cxxflags) -std=c++17 -O2 -Wall -Wextra -Wno-unused-parameter \
		-DKERNMLOPS_ORIGINAL_REDUCE_MAIN original_reduce.cpp -o "$@" \
		$$($(LLVM_CONFIG) --ldflags --link-shared --libs core irreader bitwriter passes --system-libs)

.PHONY: original-reduce test-original-reduce
original-reduce: $(BUILD)/original-reduce

test-original-reduce: $(BUILD)/original-reduce
	$(PYTHON) tests/test_original_reduce.py --build "$(BUILD)" \
		--fixtures "$(ORIGINAL_REDUCE_FIXTURES)" --llvm-bindir "$(LLVM_BINDIR)" -v
