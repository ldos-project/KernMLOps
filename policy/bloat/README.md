# Derived Bloat Correlations

This is offline exploratory analysis, not a collector, classifier, or deployed
forecasting model. Code is isolated here; generated data lives separately under
`data/derived/bloat/full-probes-20260908/`. Raw captures, frozen experiment plans,
and benchmark-performance reports are never modified.

## Inputs And Target

The source is the `all` cells of `redis-rss-dtlb-20260908`: 30 `always` and 30
`never` collections, with all twelve recorded hooks except CBMM. Other cohorts
and the historical replacement-reduced probes are not pooled into this analysis.

The audit confirms Redis identity against startup logs, process metadata, and
leader lifetime events. All 60 captures contain anonymous and file RSS component
updates, but none contains a shared-memory RSS observation. Therefore the approved
target is explicitly **observed anonymous + file RSS excess**, not complete RSS:

```text
observed_rss(t) = anonymous_pages(t) + file_pages(t), converted to MiB
excess(t) = observed_rss_always(t) - median_never_reference(phase, elapsed_time)
```

Unknown shmem and swap are not filled with zero. Swap is not resident RSS.
Component histories use only observations at or before the evaluation time;
missing initial levels and conflicting same-time values remain unknown. Signed
excess values are retained. This proxy does not measure unused hugepage bytes or
establish a causal THP effect.

## Windowed Data

The primary pass uses the YCSB `run` invocation separately from `load`, a 1-second
grid, 1-second trailing feature windows, and a 1-second horizon. Invocation times
include launcher/Maven preparation: these are not exact operation-progress bins.
Final reported YCSB runtime is audit-only and is not a feature or normalization.

Only `always` rows enter correlations. The 30 `never` runs form an exploratory
fixed median reference at matching phase/elapsed times, with at least 15 distinct
reference runs required. No cross-collection telemetry features are joined.
Reference IQR bands are descriptive dispersion, not confidence intervals or a
counterfactual for a particular always run. Future validation would need reference
fitting inside training splits.

`windows.parquet` keeps grouping/attribution metadata separate from `f__` feature
columns and the labels `excess_now_mib`, `excess_1s_mib`, and
`excess_change_1s_mib`. The latter two use future RSS only as a label; features do
not. RSS-derived features are explicitly target-linked diagnostics.

The full-probe screen includes scheduling, perf counters, file/unmap activity,
zswap, PMD scan observations, and system memory context where usable. Important
restrictions are recorded in the feature catalog and per-collection quality JSON:

- Perf is system-wide per CPU, not Redis-specific. Use interval deltas and
  enabled/running correction. Drop ambiguous counter declines/resets, long gaps,
  and invalid intervals instead of inventing wrap counts. Keep CPU coverage.
- Historical `all` lacks DTLB accesses; no miss/access ratio is manufactured.
- Event counts mean observed events, not loss-corrected or proven complete counts.
  Missing tables are not imputed as zero; eventless means remain unknown.
- Madvise success-only rows lack completion times. Merged collapse and block-I/O
  outputs have unreliable pairing/availability. They are excluded from the causal
  feature set, with explicit reasons, rather than presented as early signals.
- Scheduling and zswap durations use completion times. Scan handler time is not
  scan/collapse runtime; requested unmap ranges are not reclaimed RSS.
- Memory snapshots use BOOTTIME, while BPF uses MONOTONIC. A memory value is used
  only once the next serial snapshot's BOOTTIME timestamp upper-bounds completion
  of its read. This is conservative delayed context, not exact clock calibration;
  BOOTTIME includes suspend time and exact historical lag is unknown.

## Statistics

Pearson and average-tie Spearman correlations are computed per collection/phase,
not by pooling telemetry rows. Raw associations and descriptive partial
associations are both reported. Partial associations control elapsed time; future
level/change additionally controls current excess. These whole-phase OLS fits are
diagnostics, not deployable transforms or trained predictive validation.

Summaries give equal weight to runs and include median/mean correlation, support,
sign consistency, and a 95% bootstrap interval for the median correlation. At
least eight finite windows per run and ten valid runs are required for inference.

`p_value` is a two-sided **independent-run sign test** of whether positive and
negative run correlations are equally likely, excluding exact-zero ties. It is
not a pooled Pearson p-value, a Welch test, or a test of forecast accuracy. Holm
correction includes every valid feature/target/method/adjustment test in the pass,
including target-linked diagnostics. CI/p-values condition on the fixed estimated
reference and exclude reference uncertainty.

## Shared-Phase Sensitivity

The follow-up is explicitly post-hoc: r00-r14 always runs estimate per-variable
phase/time medians; r15-r29 always runs are centered against these fixed profiles.
Profiles require eight calibration runs at each point. The correlation engine
then screens the residual series. This helps identify associations driven by a
common startup/workload profile, but is not a confirmatory holdout or trained
predictor. Its uncertainty is conditional on the fitted calibration profiles.

The sensitivity directory also contains raw-input diagnostics, including the
fraction of future windows with unchanged Redis RSS, changes caused solely by the
never reference, and a persistence baseline. Baseline error is summarized with
equal weight per run, not by pooling all windows.

## Running

Prepare the dedicated output parent as the workspace user:

```sh
make -C policy bloat-output-dir
```

Run the workflow in the existing container, with the source archive read-only and
only the derived-data subtree writable. No privileges, network, new measurements,
or model dependencies are needed:

```sh
make docker INTERACTIVE=i CONTAINER_HOST_HEADERS=0 CONTAINER_PRIVILEGED=0 \
  CONTAINER_SOURCE_MODE=ro \
  CONTAINER_OPTS='--user $(UID):$(GID) --network none --cap-drop=ALL' \
  KERNMLOPS_CONTAINER_MOUNTS='-v $(SRC_DIR)/data/derived/bloat:$(CONTAINER_SRC_DIR)/data/derived/bloat:rw' \
  CONTAINER_CMD='make -C policy bloat-audit bloat-correlate bloat-sensitivity'
```

Outputs are deliberately not overwritten. For a new analysis version, select a
new `BLOAT_DIR`; do not delete or rewrite raw inputs. `BLOAT_ARGS` forwards options
to the selected target: use `--phase load` only with `bloat-correlate` for a separate
phase analysis, not with the audit or sensitivity targets. `test-bloat` runs
the synthetic/unit tests; the archive smoke test is opt-in.

## Outputs

- `audit/`: per-run identities, component coverage, phase bounds, table inventory,
  and quality findings in CSV/Parquet plus `audit.json`.
- `correlations/`: windowed observations/features/labels, never reference and IQR,
  feature catalog, per-run correlations, summary statistics, provenance manifest,
  quality JSON, RSS/reference plots, heatmap, and illustrative feature traces.
- `phase-sensitivity/`: calibration profiles, explicitly residualized windows,
  separate correlation results, persistence/variation diagnostics, and provenance.

Large correlations, especially on small changes in a mostly static RSS level,
are not by themselves evidence that impending bloat can be predicted. Report
physical effect sizes and the shared-phase sensitivity alongside p-values.
