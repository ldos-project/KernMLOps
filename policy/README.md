# Policy Field Extractor

The libbpf field extractor in `policy/` uses LLVM 18, with a syscall fixture and a
block-I/O port; it does not accept BCC syntax verbatim. The opt-in
[original-source RSS/DTLB path](#original-source-rssdtlb) instead compacts the
authoritative BCC/Kermit sources. Existing collector defaults remain unchanged.

## Usage

Run all builds, extraction, inspection, and tests through Make, from the repository root.
The container targets use the repository's existing `Dockerfile.dev`, `docker-image`,
and `docker` Make targets. The shared development image includes LLVM 18 and libbpf
build dependencies; no separate policy image or host package installation is needed:

```sh
make -C policy container
make -C policy container FIELDS=syscall_nr
make -C policy container TARGET=inspect
```

The default source is `bpf/syscall.bpf.c`, program `capture_syscall` attached at
`raw_tp/sys_enter`, and event `policy_event`. Selecting the default fields
`syscall_nr,doubled_syscall_nr` reduces its 32-byte payload to 8 bytes.

The repository is mounted at the shared runner's `CONTAINER_SRC_DIR` (normally
`/KernMLOps`), and policy commands run from its `policy/` directory. Sources and
project headers elsewhere in the repository are accessible using relative paths.
Policy compiler targets set `CONTAINER_PRIVILEGED=0` and `CONTAINER_HOST_HEADERS=0`, so
they do not need BPF privileges and host include directories do not
hide the container's LLVM/libbpf headers. Collector commands retain host headers
by default. Container command failures propagate through Make.
Native builds need LLVM/Clang 18, LLVM/libbpf development packages, and C++ headers.
Tests also need Python 3; live tests need libelf/zlib development packages.

```sh
make -C policy extract SOURCE=bpf/syscall.bpf.c PROGRAM=capture_syscall EVENT=policy_event FIELDS=syscall_nr,doubled_syscall_nr
make -C policy inspect
```

`container` accepts the same selection variables. Inspection rebuilds the selection;
repeat custom variables when inspecting a nondefault selection.

## Design

1. Clang compiles source with `-g` and LLVM optimization passes disabled into
   unoptimized `input.bc`, also disassembled as `input.ll`.
2. `field-extract` is a standalone LLVM 18 C++ transformer, not a stock `opt` plugin.
   It uses debug types to project selected fields into a new, naturally aligned event.
   Output follows source declaration order, not the order supplied in `FIELDS`.
3. Generic LLVM passes remove dead dataflow, with helper-aware effect summaries and
   cleanup. Helper 113 (`bpf_probe_read_kernel`) reads its source and writes its
   destination; removal requires an unused status and unread, nonescaping stack bytes.
   Helpers 5, 8, 14, and 15 (time, CPU, PID/TGID, UID/GID) are removed when unused.
4. The transformer writes `selected.bc` and `selected.schema.json`; Clang's BPF backend
   compiles the selected bitcode into `selected.bpf.o`.

Selection changes output, not every source use: aliases and local values needed by
retained computations or effects remain. Unknown and map effects are kept; state-map
layouts are never shrunk, and ring-buffer capacity is unchanged. The payload contains
only selected fields and ABI-required padding; instruction count and temporary stack
usage are not guaranteed to be globally minimal.

## Supported Boundaries

- One selected entry program with exactly one `bpf_ringbuf_output` site, submitting
  one whole stack-allocated event struct. Output in non-inlined subprograms is rejected.
- Multiple attached programs require `--preserve-other-programs` (passed through
  `EXTRACT_FLAGS` in Make). Companion programs and their output layouts remain intact;
  their names are recorded in the JSON schema. This is not cross-program state pruning.
- The entire event must be a non-packed, fixed-layout struct of signed or unsigned
  8/16/32/64-bit integer scalars, including unselected members. Arrays, pointers,
  bitfields, packed structs, and other field types are rejected.
- Matching full debug information is required. Field selection must be nonempty,
  with unique, existing member names. Multiple emit sites in the selected program are rejected.
- Source plus compact event size must not exceed 512 bytes. This is a conservative
  temporary-stack budget before optimization, not proof of total stack usage or
  kernel verifier acceptance.

## Outputs And Consumers

Container artifacts are in `policy/build/container/`; native artifacts are in
`policy/build/`: `field-extract`, `input.bc`, `input.ll`, `selected.bc`, `selected.ll`,
`selected.bpf.o`, and `selected.schema.json`.

The JSON schema describes the generated event size, alignment, byte order, and each
selected field's name, offset, size, and signedness. Existing consumers must adapt their decoders to
this layout; regenerating a libbpf skeleton alone does not change event decoding.
Use the JSON as the payload contract: LLVM may omit stack-only event types from BTF.
Kernel/map BTF and retained CO-RE information are preserved independently of that schema.

Exact probe-overhead measurements are not preserved. Smaller/faster events can change
timestamps and drop behavior, even with unchanged ring-buffer capacity. Event timing
must not be treated as evidence of unchanged probe overhead.

## Tests

```sh
make -C policy container-test
make -C policy container-test-load
make -C policy test
make -C policy test-load
```

The compiler tests check compact layouts, padding, signedness, shared dependencies,
helper removal, retained state/error handling, CO-RE relocations, and rejected inputs.
The live test loads and attaches the default selection and checks actual 8-byte events.
Both suites have passed on the development host. The example reads the low 32 bits of
a syscall argument through its address and targets little-endian hosts.

`container-test-load` builds as the invoking user, then runs only the loader through
the shared privileged container runner with a read-only source mount and networking disabled.
It briefly attaches to the host's syscall tracepoint and detaches on exit. It requires
host BPF support and a Docker daemon permitted to grant those privileges. Native
`test-load` needs equivalent BPF/tracing privileges; `RUNNER` can supply a privilege wrapper.

## Block I/O

`bpf/block_io.bpf.c` retains the normal probe's issue/completion logic, all original
event fields, `device_queue`, and `started_4k_ios`. The adaptation replaces BCC/Kermit
syntax with libbpf maps, CO-RE reads, and ring-buffer output. The minimal kernel types
and flag constants in `bpf/block_io.h` target Linux 6.12's `request->q->disk` path;
older `rq_disk` layouts and different flag-bit definitions are not supported.

The compiler/replay test needs no kernel tracing privileges:

```sh
make -C policy container-test-block-io
```

It compiles the real CO-RE port and extracts three selections. A second build uses
the same probe logic with local request layouts and replaces helper addresses with
native test stubs. Full and extracted IR receive identical requests, timestamps,
and map behavior over 25 operations. Tests compare selected payload bytes, complete
companion records, event suppression, and both state maps, including output failures,
map insertion failures, orphan completions, and special segment-count cases.
This is equivalence to the unextracted libbpf port, not to the original transport.

Measured results for the selected program (companion code is not included in these counts):

| Selection | Payload Bytes | BPF Instructions | Kernel Read Calls |
| --- | --- | --- | --- |
| Completion: `block_latency_us,block_io_latency_us` | 56 -> 16 | 188 -> 165 | 14 -> 13 |
| Issue: `queue_length_segments,queue_length_4ks` | 48 -> 8 | 186 -> 158 | 12 -> 11 |
| Completion: `device,block_io_end_uptime_us` | 56 -> 16 | 188 -> 147 | 14 -> 11 |

Artifacts are in `build/container/block_io/`: `original.bpf.o`, `latency.bpf.o`,
`queue.bpf.o`, `identity.bpf.o`, their schemas/disassemblies/replay logs, and `report.json`.
For example, `latency.bpf.o` emits a 16-byte completion stream but still has the
48-byte issue stream. Queue maintenance and both attachments remain necessary for
that retained companion output. The extractor does not yet produce a latency-only
object with no queue maps or issue program.

The separately authorized live test loads both block tracepoints, reads kernel BTF,
and writes/fsyncs up to 8 MiB through a newly created regular file:

```sh
make -C policy container-test-block-io-load
```

Explain and obtain permission before running it. Only the disposable workload directory
is mounted writable in addition to the shared runner's existing mounts; the repository
source mount is read-only. The file is unlinked immediately after creation, and the
writer closes its inherited tracing descriptors before I/O. Cancellation detaches the
parent's links without waiting indefinitely for blocked storage. The test uses `fstatfs`
on its own file, not host mount-table inspection, and rejects RAM-backed filesystems.

Live loading and real 48/16-byte issue/completion records have been verified on the
development kernel. Streams include background I/O; without request identity in the
two-field completion payload, the live check cannot attribute each record to the test
file or assert equality between separately timed live events. Exact values are checked
by deterministic replay instead.

## Files

- `Makefile`: build/test/inspection targets, forwarding container work to the root Makefile.
- `../Dockerfile.dev`: the shared development image and policy toolchain dependencies.
- `extract.cpp`: standalone LLVM transformer and JSON schema generation.
- `bpf/syscall.bpf.c`: syscall event fixture.
- `bpf/block_io.bpf.c`, `bpf/block_io.h`: the block-I/O logic and required types.
- `tests/test_extract.py`, `tests/load.c`: compiler tests and the libbpf live-load test.
- `tests/test_block_io.py`, `tests/block_io_replay.c`, `tests/block_io_load.c`: block-I/O validation.

## Redis/YCSB Results

Redis benchmark collections also save the YCSB client's own results. No t-test or
other statistical analysis is performed automatically.

```text
<output_dir>/{baseline|curated}/redis/<collection_id>/
    ycsb_invocations.end.parquet
    ycsb_metrics.end.parquet
    ycsb/0000.load.stdout.log
    ycsb/0000.load.stderr.log
    ycsb/0001.run.stdout.log
    ycsb/0001.run.stderr.log
    ...
```

Each actual load/run launch gets a unique invocation ID, including failed launches.
The invocation table records phase, inner/outer iteration, command, observed start/end
times, exit status, parse status, and collection-relative raw-log paths. `run_label`
preserves the collection prefix when supplied; it does not infer training/inference
from the output directory or change the selected probes.

The metric table preserves every final `[section], metric, value` record, not a
whitelist: throughput, runtime, operation counts, latency summaries/percentiles,
return statuses, GC statistics, and additional exporter records. Names and raw values
are retained alongside a nullable numeric value and unit. Duplicate metric names are
separate rows; missing metrics are not zero. Progress/status text stays in the raw logs.

Only raw logging and ordinary Python invocation bookkeeping happen during the
benchmark. Parsing and DataFrame construction occur once after all load/run and repeat
cycles finish, the measured interval ends, and collection workers stop. The existing
collector then writes both final tables in the Redis collection directory. Existing
probe-side live collection is unchanged by this YCSB-specific deferral.

Join metrics to invocation metadata on `(collection_id, invocation_id)` when
postprocessing, and select the desired phase/operation/metric explicitly. Check both
`execution_state` and `parse_state`: a zero process exit code need not imply that
YCSB produced measurements. Partial results and raw logs are retained on failures.

Focused tests use dummy child processes, not Redis or kernel instrumentation:

```sh
make docker INTERACTIVE=i CONTAINER_HOST_HEADERS=0 CONTAINER_PRIVILEGED=0 \
  CONTAINER_SOURCE_MODE=ro CONTAINER_OPTS='--user $(UID):$(GID) --network none' \
  CONTAINER_CMD='make -C policy test-ycsb-results'
```

For a real Redis/YCSB run, use the existing benchmark configuration and the normal
container environment rather than a separate smoke workload:

```sh
make docker INTERACTIVE=i \
  CONTAINER_CMD="bash -lc 'make collect-data KERNMLOPS_CONFIG_FILE=config/redis_always.yaml COLLECTION_PREFIX=redis-always-ycsb'"
```

Explain the config's requested permissions before execution: it clears page caches,
sets THP to `always` and overcommit to `never_check`, and attaches its configured
kernel probes. Those settings are not restored by the benchmark. YCSB must already
be installed; its normal launcher may invoke Maven to resolve its runtime classpath.
The command reuses the existing container image without rebuilding it.

This configuration was validated with one successful load and one successful run;
both raw outputs and their final metric/invocation tables were saved in the Redis
collection directory. Operation-level errors remain recorded even when YCSB exits
with status zero.

## Comparing YCSB Runs

`ycsb_ttest.py` is a separate offline utility. It reads saved collection directories
containing `ycsb_invocations.end.parquet` and `ycsb_metrics.end.parquet`; it does not
start benchmarks, change probe settings, or modify the captured results.

Compare two collections by passing their actual directory paths:

```sh
make docker INTERACTIVE=i CONTAINER_HOST_HEADERS=0 CONTAINER_PRIVILEGED=0 \
  CONTAINER_SOURCE_MODE=ro CONTAINER_OPTS='--user $(UID):$(GID) --network none' \
  CONTAINER_CMD='python policy/ycsb_ttest.py --left data/curated/redis/RUN_A --right data/curated/redis/RUN_B'
```

The default is the transaction (`run`) phase. Use `--phase load` or `--phase all`
to include other phases. CSV is printed to stdout; `--output path.csv` saves it to
a new file instead (its parent must exist and be writable). Existing files are
never overwritten.

Each row matches the benchmark name, phase, outer/inner iteration, section,
metric name, and unit. The report includes sample counts, means, between-run
standard deviations, right-minus-left differences, percentage changes, t-statistics,
raw p-values, and Holm-adjusted p-values across the valid tests in that report.
There is no automatic significance or acceptability threshold.

**One collection per side is not enough for a per-metric t-test.** Current YCSB
exports contain one summary value per metric per invocation, not its underlying
request samples or variance. The utility still reports values and differences,
but leaves t/p values null with `status=insufficient_samples`.

Supply independent replicate collections on each side to enable two-sided Welch
tests (at least two per side, with more usually needed for useful precision):

```sh
make docker INTERACTIVE=i CONTAINER_HOST_HEADERS=0 CONTAINER_PRIVILEGED=0 \
  CONTAINER_SOURCE_MODE=ro CONTAINER_OPTS='--user $(UID):$(GID) --network none' \
  CONTAINER_CMD='python policy/ycsb_ttest.py --left data/curated/redis/TRAINING_1 data/curated/redis/TRAINING_2 --right data/curated/redis/INFERENCE_1 data/curated/redis/INFERENCE_2'
```

The statistical sample count is the number of collections, not YCSB's `Operations`
count. Repeat positions, load/run phases, and different metrics are never pooled.
For percentile metrics, the test compares run-level percentile summaries, not
individual request latencies or a pooled latency distribution. The caller is
responsible for independent runs and comparable experimental conditions.

Missing values are not zero. Failed/partially parsed invocations, duplicate metric
records, non-finite values, or missing metrics block inference for the affected
row and are explained in `status`/`notes`. Recorded YCSB arguments must match after
the launcher prefix; otherwise `configuration_mismatch` is reported. Collection IDs
cannot repeat within or across groups. If both samples have zero variance, the
t-test is undefined and is reported as `zero_variance`, not a manufactured p-value.
Operation-level error counts from otherwise completed invocations remain metrics.

Python callers can use `compare_ycsb_runs(left, right, phase="run")` for a DataFrame
or `welch_ttest(left_numbers, right_numbers)` for a standalone numeric comparison.
Both use the same right-minus-left sign convention; a negative latency change and
a positive throughput change have different performance meanings.

Run the synthetic-data, SciPy-reference, input-integrity, and CLI tests without
Redis, network access, or kernel privileges:

```sh
make docker INTERACTIVE=i CONTAINER_HOST_HEADERS=0 CONTAINER_PRIVILEGED=0 \
  CONTAINER_SOURCE_MODE=ro CONTAINER_OPTS='--user $(UID):$(GID) --network none' \
  CONTAINER_CMD='make -C policy test-ycsb-ttest'
```

## Historical RSS/DTLB Matrix

The completed historical experiment is in `data/experiments/redis-rss-dtlb-20260908/`.
It contains 180 separate Redis benchmark collections: 30 per combination of THP
`always`/`never` and `none`/`all`/`reduced` instrumentation. The order is randomized
within each six-condition round using seed 17. YCSB workload properties come from
`config/redis_always.yaml` and `config/redis_never.yaml` without size/mix changes.

- `none`: no collector hooks; native YCSB result capture still runs.
- `all`: all 12 existing registered hooks except `cbmm`. The legacy collectors and
  their event choices remain unchanged; `cbmm` is never instantiated or attached.
- `reduced`: only `extracted_rss_dtlb`, a libbpf collector loading LLVM-extracted
  RSS and paired DTLB objects. It does not also enable the default hooks.

`bpf/rss.bpf.c` retains RSS component counts and TGID attribution, reducing its
32-byte event to 24 bytes. Member 2 is swap and must not be summed into resident
RSS; this is a component stream, not an independently sampled aggregate RSS value.
The raw/formatted tracepoint bridge refreshes each owner and accounts for missing
or unavailable ownership rather than silently publishing stale/zero identities.

`bpf/dtlb.bpf.c` uses grouped, system-wide per-CPU DTLB READ ACCESS and READ MISS
counters. Access interrupts trigger paired snapshots at a requested 1,000 Hz.
The extracted event is 64 bytes instead of 72, retaining CPU, cumulative 64-bit
counts, and enabled/running times needed for the ratio. Each valid interval computes:

```text
scaled_misses   = delta(misses)   * delta(miss_enabled)   / delta(miss_running)
scaled_accesses = delta(accesses) * delta(access_enabled) / delta(access_running)
dtlb_missrate   = scaled_misses / scaled_accesses
```

First samples, resets, and zero/unscheduled spans have null ratios. Counter state
is carried across output chunks. Ratios are CPU-wide, not Redis-only. Ring-output
losses and unattributed RSS events are recorded in `policy_status`; bridge and
counter-read failures invalidate a collection. Cumulative DTLB snapshots can span
missed notifications, but the telemetry streams are not claimed to be lossless.

This compares the chosen full and reduced collection pipelines, including their
backend/transport differences, not an isolated LLVM-only speedup. In particular,
the legacy full perf hook exposes its existing miss counters, while the reduced
collector adds the access denominator for the explicitly requested ratio.

The native source, compiler layouts, object/library hashes, generated configs,
collector source hashes, registry snapshot, and run ordering are frozen before
execution. The runner stops on an invalid run, preserves every attempt, resumes
only after revalidating completed output, and refuses `cbmm` in every condition.
It uses fresh collector processes, never parallel workloads or internal YCSB
iterations as independent repetitions. `manifest.json` tracks progress and
`results.jsonl` links each job to its normal Redis collection directory.

Build and test without kernel privileges using the existing container:

```sh
make docker INTERACTIVE=i CONTAINER_HOST_HEADERS=0 CONTAINER_PRIVILEGED=0 \
  CONTAINER_OPTS='--user $(UID):$(GID)' \
  CONTAINER_CMD='make -C policy BUILD=build/container rss-tlb test-rss-tlb test-ycsb-matrix'
```

For a NEW experiment using this historical libbpf path, choose a new `MATRIX_DIR`.
Freeze the plan with `make -C policy matrix-prepare MATRIX_DIR=...`, then execute
through `make docker` with `make -C policy matrix-run MATRIX_DIR=...`. Both targets
accept `MATRIX_REPEATS` and `MATRIX_SEED`. `MATRIX_ARGS="--limit 6"` runs one round;
omitting the limit resumes all remaining jobs. Do not rebuild/change the frozen
collector inputs mid-experiment.

Explain host permissions before executing: each configured run clears page caches,
sets THP, changes overcommit, and attaches probes/PMU events. The runner snapshots
and restores the initial THP/overcommit settings on orderly exit, failure, or handled
interrupt; cache contents cannot be restored. Forced termination can bypass cleanup.

The completed matrix's offline outputs are under `analysis/`:

- `raw_metrics.csv`: every captured YCSB metric with condition and repetition labels.
- `overheads.csv`: per-round overhead relative to no probes at the same THP setting.
- `overall_overhead.png` and `latency_overhead.png`: raw and normalized values,
  individual runs, distributions, and sample counts; no acceptance thresholds.
- `coverage.json`: confirms 30/30 in all six cells and identifies missing/invalid data.
- `ttest-*.csv`: nine optional Welch reports, requested with `SUMMARY_ARGS=--ttests`.

```sh
make docker INTERACTIVE=i CONTAINER_HOST_HEADERS=0 CONTAINER_PRIVILEGED=0 \
  CONTAINER_OPTS='--user $(UID):$(GID)' \
  CONTAINER_CMD='make -C policy summarize-ycsb-matrix MATRIX_DIR=data/experiments/redis-rss-dtlb-20260908 SUMMARY_ARGS=--ttests'
```

Postprocessing acquires a shared experiment lock and refuses an active runner.
It uses the latest attempt only, never substitutes an older success for a failed
retry, and does not alter raw collection files. Overhead percentages are normalized
within each THP/repetition pair, so their medians are not ratios of group medians.

## Original-Source RSS/DTLB

Privileged `capture` and `mpmo` preflights and the 120-run original-source matrix
have completed successfully; results are in `data/experiments/redis-original-llvm-20260909/`.
This path is separate from the historical replacement probes above and the draft
`--mpmo-only` path; use `--original-only` for the new experiment.

Authoritative inputs are `mm_trace_rss_stat.bpf.c` and the source generated by
`PerfBPFHook(events=[DTLBPerfTable, DTLBAccessPerfTable])`; `PERF_HANDLER` is unchanged.
These are not new handwritten probes. DTLB READ ACCESS is now also registered in
the normal/full `perf` collector, using the existing handler and 1,000 Hz sampling.
The archived ALL controls predate this addition and still lack the ACCESS counter.
Private Ubuntu BCC 0.29.1
(`0.29.1+ds-1ubuntu7`, commit `4752b9906825bfc3b86e4590293fc55d7a5db4f8`) matches the
installed library's source and LLVM 18.1.3. `bcc_frontend.patch` adds only 16 lines:
an optional hook before JIT/map creation. `original_reduce.cpp` performs LLVM field
selection, record compaction, and map/BTF plus BCC metadata updates. Required RSS
bridge/ownership and shared perf-read dependencies remain; this is not state pruning.

| Stream | Original -> Compact Payload | MPMO Slot Value |
| --- | --- | --- |
| RSS | 32 -> 24 bytes | 8 state + 24 payload = 32 bytes |
| MISS / ACCESS | 40 -> 32 bytes each | 8 state + 8 CPU metadata + 32 payload = 48 bytes |

RSS retains `tgid,ts,member,counter_value`; its 32-byte bridge value is unchanged.
Perf retains `ts_uptime_us,count,enabled_time_us,running_time_us`: `count` stays
`u32`, all perf times remain microseconds, and RSS timestamps remain nanoseconds.
MPMO has 32 slots per queue. `KERMIT_CPU_METADATA` records the last writer's CPU
only for the new MPMO perf build; the `kermit.h` default of 4096 slots is unchanged.
Capture preserves perf CPU attribution through its callback CPU argument.

The existing collector thread, configured with `poll_rate: 0.1`, reads physical
slot 0 once per RSS/miss/access queue per MPMO poll, without scanning other slots.
There is no inference model or aggregation. Capture saves normal `policy_rss`, raw
`original_perf`, and final `policy_status`; MPMO saves only `policy_status`, not an
event stream. Both record generated/compiled source, pre/post LLVM IR, generated
schemas, and loaded BPF bytecode hashes under the bundle's `evidence/collections/`.
Original miss/access counts are asynchronous, system-wide per CPU (requested
1,000 Hz each): neither mode computes a miss/access ratio. Compact records retain
inputs for future aggregation and preserve CPU attribution; that aggregation is not
implemented. Original producer/bridge/read failures are not observable, so status
is not evidence of losslessness or zero errors.

Opt in under `collector_config.generic` (exclusive with `extracted_rss_tlb`);
`hooks: []` is required. Normal selection defaults remain unchanged:

```yaml
hooks: []
poll_rate: 0.1
original_rss_tlb: /KernMLOps/policy/build/container/original_bcc/manifest.json
original_rss_tlb_transport: capture  # or mpmo
```

`Makefile` includes `original_bcc.mk` and `original_reduce.mk`. All commands below
run from the repository root through `make docker`, using only the existing image.
Build/export recipes document fresh setup, not steps to rerun for the validated
bundle. Never rebuild historical `rss_tlb` artifacts or change frozen matrix inputs.
`original-bcc-build` invokes `original-bcc-source` to pin/verify the private source.
Install Clang development dependencies temporarily as root in an unprivileged
container, then build as the workspace UID into `build/container/original_bcc/install`
in that same container, never into the global BCC installation:

```sh
make docker INTERACTIVE=i CONTAINER_HOST_HEADERS=0 CONTAINER_PRIVILEGED=0 \
  CONTAINER_OPTS='--user 0:0' \
  CONTAINER_CMD='bash -lc "make -C policy BUILD=build/container original-bcc-deps && exec setpriv --reuid=$(UID) --regid=$(GID) --clear-groups make -C policy BUILD=build/container build/container original-bcc-build"'
```

Compile-only bundle export stops before BPF loading. Supplementary group 0 permits
tracefs reads without capabilities; keep all four isolation options shown below.
`original-bcc-bundle` also invokes `original-bcc-manifest` to freeze generated schemas:

```sh
make docker INTERACTIVE=i CONTAINER_HOST_HEADERS=0 CONTAINER_PRIVILEGED=0 \
  CONTAINER_OPTS='--user $(UID):$(GID) --group-add 0 --cap-drop ALL --network none' \
  KERNMLOPS_CONTAINER_MOUNTS='-v /opt/kermit/linux:/kernel-source:ro -v /opt/kermit/kbuild:/kernel-build:ro' \
  KERNMLOPS_CONTAINER_ENV='-e BCC_KERNEL_SOURCE=/kernel-source -e BCC_KERNEL_BUILD=/kernel-build' \
  CONTAINER_CMD='make -C policy BUILD=build/container original-bcc-bundle'
```

Use the same unprivileged runner with targets `test-original-collector`,
`test-original-matrix`, and `test-original-reduce` instead of `original-bcc-bundle`.
The reducer tests consume frozen original frontend IR (default
`build/container/original_bcc/evidence/frontend-v4`), not newly handwritten probes.

Real preflights and matrix runs require user-approved privileged root containers.
Keep the normal benchmark mounts and the **default `EXTERNAL_BENCHMARK_DIR`**;
do not substitute a build directory, which would hide the installed YCSB workload.
Pass host identity for output ownership, plus the matching kernel source/build:

```sh
make docker INTERACTIVE=i CONTAINER_HOST_HEADERS=0 CONTAINER_PRIVILEGED=1 \
  CONTAINER_OPTS='--user 0:0' \
  KERNMLOPS_CONTAINER_MOUNTS='-v /opt/kermit/linux:/kernel-source:ro -v /opt/kermit/kbuild:/kernel-build:ro' \
  KERNMLOPS_CONTAINER_ENV='-e UNAME=$(UNAME) -e SUDO_UID=$(UID) -e SUDO_GID=$(GID) -e BCC_KERNEL_SOURCE=/kernel-source -e BCC_KERNEL_BUILD=/kernel-build' \
  CONTAINER_CMD='make -C policy BUILD=build/container preflight-original ORIGINAL_TRANSPORT=capture'
```

Repeat with `ORIGINAL_TRANSPORT=mpmo`; both preflights have passed. Preflight only
attaches/checks probes, not Redis or VM settings. Matrix execution also clears caches
and changes THP/overcommit; the restoration and forced-termination caveats above apply.
The runner sets private `PYTHONPATH`/`LD_LIBRARY_PATH` only for original collector
children. The runner and normal collectors continue using the package BCC unchanged.

Freeze then run the four new cells: THP `always`/`never` crossed with internal modes
`original-reduced` (capture) / `original-mpmo`, 30 repeats each, seed 17, 120 new runs.
Keep the same `MATRIX_PLAN_ARGS` for preparation and execution; omit `matrix-prepare`
when resuming an existing plan:

```sh
make docker INTERACTIVE=i CONTAINER_HOST_HEADERS=0 CONTAINER_PRIVILEGED=1 \
  CONTAINER_OPTS='--user 0:0' \
  KERNMLOPS_CONTAINER_MOUNTS='-v /opt/kermit/linux:/kernel-source:ro -v /opt/kermit/kbuild:/kernel-build:ro' \
  KERNMLOPS_CONTAINER_ENV='-e UNAME=$(UNAME) -e SUDO_UID=$(UID) -e SUDO_GID=$(GID) -e BCC_KERNEL_SOURCE=/kernel-source -e BCC_KERNEL_BUILD=/kernel-build' \
  CONTAINER_CMD="make -C policy matrix-prepare matrix-run MATRIX_DIR=data/experiments/redis-original-llvm-20260909 MATRIX_REPEATS=30 MATRIX_SEED=17 MATRIX_PLAN_ARGS='--original-only --manifest /KernMLOps/policy/build/container/original_bcc/manifest.json'"
```

Reuse only the 120 historical `none`/`all` controls from
`data/experiments/redis-rss-dtlb-20260908`; exclude its 60 replacement-`reduced` runs.
Do not mutate that experiment, its artifacts, or raw data. Historical `all` lacks
ACCESS, and old/new runs are different time cohorts. There is no inference model
and no isolated LLVM causal claim. YCSB parsing still occurs at collection end,
after the entire benchmark and worker shutdown, not inside its measured interval.

After stopping the matrix, `compare_original_matrix.py` produces raw plots/metrics,
coverage, ratio-of-means overheads against historical `none`, and ten Welch reports
(five contrasts per THP, Holm correction within each report). Never pair old/new
replicate indices; these overheads are not historical paired-round medians. Outputs
go only into the new matrix's `analysis/`. Require all 240 valid candidate/control
collections for a complete comparison, without implying they are available yet:

```sh
make docker INTERACTIVE=i CONTAINER_HOST_HEADERS=0 CONTAINER_PRIVILEGED=0 \
  CONTAINER_OPTS='--user $(UID):$(GID) --cap-drop ALL --network none' \
  CONTAINER_CMD='make -C policy compare-original-matrix MATRIX_DIR=data/experiments/redis-original-llvm-20260909 SUMMARY_ARGS=--require-complete'
```

## Selected-Probes Control

`--selected-control-only` runs the original, full-record capture probes without
the LLVM field-reduction pass. It deploys only the RSS raw/formatted tracepoints
and DTLB READ MISS / READ ACCESS handlers. ITLB, flush, walk-duration, other hooks,
and `cbmm` are excluded. This uses the installed BCC and normal capture callbacks;
no reduction manifest or private compiler build is required.

The opt-in collector filter is:

```yaml
collector_config:
  generic:
    hooks: [perf, mm_rss_stat]
    perf_events: [dtlb_misses, dtlb_accesses]
```

Omitting `perf_events` preserves the normal perf registry; an explicit list selects
only those registered counters. The selected control retains all source fields,
including PID/TGID, and saves `mm_rss_stat`, `dtlb_misses`, and `dtlb_accesses`.
The runner validates full records in all three tables and rejects extra probe tables.

The separate control matrix has 60 runs: 30 each for THP `always` and `never`, with
the same workload and seed 17. Use the read-only source/build header mounts below
so normal BCC discovers the matching kernel headers without the kheaders fallback.
The privileged run changes/restores THP and overcommit and clears page caches;
page-cache contents cannot be restored. `preflight-selected-control` provides a
short attachment check without Redis or VM-setting changes.

```sh
make docker INTERACTIVE=i CONTAINER_HOST_HEADERS=0 \
  KERNMLOPS_CONTAINER_MOUNTS='-v /opt/kermit/linux:/lib/modules/$(KERNEL_VERSION)/source:ro -v /opt/kermit/kbuild:/lib/modules/$(KERNEL_VERSION)/build:ro' \
  KERNMLOPS_CONTAINER_ENV='-e UNAME=$(UNAME) -e SUDO_UID=$(UID) -e SUDO_GID=$(GID)' \
  CONTAINER_CMD='bash -lc "make -C policy matrix-prepare matrix-run MATRIX_DIR=data/experiments/redis-selected-control-20260909 MATRIX_REPEATS=30 MATRIX_SEED=17 MATRIX_PLAN_ARGS=--selected-control-only"'
```

After completion, add the control to the existing comparison. This writes only to
the control directory's `analysis/`, leaving both earlier experiments untouched:

```sh
make docker INTERACTIVE=i CONTAINER_HOST_HEADERS=0 CONTAINER_PRIVILEGED=0 \
  CONTAINER_OPTS='--user $(UID):$(GID) --cap-drop ALL --network none' \
  CONTAINER_CMD="make -C policy compare-original-matrix MATRIX_DIR=data/experiments/redis-original-llvm-20260909 SUMMARY_ARGS='--control-directory data/experiments/redis-selected-control-20260909 --require-complete'"
```

The combined comparison has 300 collections and 18 Welch reports, including
selected-control versus each earlier mode within THP. Means, sample standard
deviations, and raw/Holm-adjusted p-values remain per metric. Replicate numbers
across cohorts are not paired. This control matches the reduced modes' deployed
probe/counter set, but uses stock full-record capture processing; cohort timing
and consumer differences still prevent attributing differences solely to LLVM.
