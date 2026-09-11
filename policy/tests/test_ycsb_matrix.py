"""Unprivileged matrix tests: mocked collectors, plus optional real Parquet reads."""

import importlib.util
import json
import os
import signal
import subprocess
import sys
import tempfile
import unittest
from collections import Counter
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

SCRIPT = Path(__file__).resolve().parents[1] / "run_ycsb_matrix.py"
SPEC = importlib.util.spec_from_file_location("matrix_under_test", SCRIPT)
matrix = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(matrix)
REGISTRY_NAMES = matrix.registry_names
READ_ROWS = matrix.read_rows

HOOKS = [
    "file_data",
    "memory_usage",
    "process_metadata",
    "quanta_runtime",
    "block_io",
    "perf",
    "collapse_huge_pages",
    "madvise",
    "unmap_range",
    "mm_rss_stat",
    "process_trace",
    "zswap_runtime",
]
UUID = "12345678-1234-4321-abcd-123456789abc"


def fake_configuration(raw):
    config = deepcopy(raw)
    config["benchmark_config"]["redis"].setdefault("repeat", 1)
    config["benchmark_config"]["redis"].setdefault("outer_repeat", 1)
    generic = config["collector_config"]["generic"]
    generic.setdefault("extracted_rss_tlb", None)
    generic.setdefault("extracted_rss_tlb_mpmo", None)
    generic.setdefault("original_rss_tlb", None)
    generic.setdefault("original_rss_tlb_transport", "capture")
    generic.setdefault("perf_events", None)
    declared = generic["hooks"]
    for selector, hook, transport in (
        ("extracted_rss_tlb", matrix.EXTRACTED_HOOK, "ringbuf"),
        ("extracted_rss_tlb_mpmo", matrix.MPMO_HOOK, "mpmo"),
    ):
        if generic[selector] is not None:
            bundle = json.loads(Path(generic[selector]).read_text())
            if bundle.get("transport", "ringbuf") != transport:
                raise ValueError("factory rejected bundle transport")
            declared = [hook]
    metadata = {"declared_hooks": declared, "verification": "declaration_only"}
    if generic["original_rss_tlb"] is not None:
        metadata["manifest"] = fake_original_manifest(generic["original_rss_tlb"])
        metadata["declared_hooks"] = [
            "original_rss_dtlb_" + generic["original_rss_tlb_transport"]
        ]
    if generic["perf_events"] is not None:
        metadata = {
            "hooks": generic["hooks"].copy(),
            "perf_events": generic["perf_events"].copy(),
        }
    return config, metadata


def fake_original_manifest(path):
    """Stand-in for the separately implemented passive bundle API, not BCC loading."""
    manifest = json.loads(Path(path).read_text())
    if manifest.get("kind") != "original-bcc-llvm":
        raise ValueError("not an original bundle")
    for item in manifest["files"].values():
        if (
            matrix.digest((Path(path).parent / item["name"]).read_bytes())
            != item["sha256"]
        ):
            raise ValueError("original file hash mismatch")
    for source, sha in manifest["sources"].items():
        if matrix.digest(Path(source).read_bytes()) != sha:
            raise ValueError("original source hash mismatch")
    return manifest


def original_status(cid, transport, evidence):
    status = {
        "collection_id": cid,
        "origin": "original-bcc-llvm",
        "transport": transport,
        "rss_schema_bytes": 24,
        "dtlb_schema_bytes": 32,
        "callback_errors": 0,
        "compiler_evidence": str(evidence),
        "polls": 10,
        "counter_scope": "system-wide per CPU; independently sampled cumulative DTLB READ MISS + READ ACCESS; original u32 counts; no inference, aggregation, or model",
        "collector_error": "",
    }
    for stream in ("rss", "dtlb_misses", "dtlb_accesses"):
        status.update(
            {
                f"{stream}_received": 3,
                f"{stream}_empty": 5,
                f"{stream}_busy": 2,
                f"lost_{stream}": 0,
            }
        )
    if transport == "mpmo":
        status.update(data_slots=32, read_slot=0)
    return status


def mpmo_status(cid):
    return {
        "collection_id": cid,
        "transport": "mpmo",
        "data_slots": 32,
        "read_slot": 0,
        "polls": 10,
        "rss_slot0_reads": 4,
        "dtlb_slot0_reads": 3,
        "rss_empty": 5,
        "dtlb_empty": 5,
        "rss_busy": 1,
        "dtlb_busy": 2,
        "rss_checksum": (1 << 64) - 1,
        "dtlb_checksum": (1 << 63) + 1,
        "reader_errors": 0,
        "rss_emit_failures": 3,
        "rss_owner_misses": 0,
        "dtlb_emit_failures": 5,
        "dtlb_read_errors": 0,
        "rss_unattributed": 7,
        "rss_schema_bytes": 24,
        "dtlb_schema_bytes": 64,
    }


class MPMOStatusTests(unittest.TestCase):
    def test_pure_validation_preserves_uint64_counters_and_allows_diagnostics(self):
        rows = [mpmo_status("test")]
        before = deepcopy(rows)
        self.assertTrue(matrix.validate_mpmo_status(rows, "test"))
        self.assertEqual(rows, before)
        rows[0].update(
            polls=(1 << 64) - 1,
            rss_slot0_reads=(1 << 64) - 1,
            dtlb_slot0_reads=(1 << 64) - 1,
            rss_empty=0,
            dtlb_empty=0,
            rss_busy=0,
            dtlb_busy=0,
        )
        self.assertTrue(matrix.validate_mpmo_status(rows, "test"))

    def test_single_row_identity_transport_slots_and_schemas_are_required(self):
        status = mpmo_status("test")
        for rows in ([], [status, status], [None]):
            with (
                self.subTest(rows=rows),
                self.assertRaisesRegex(ValueError, "one MPMO"),
            ):
                matrix.validate_mpmo_status(rows, "test")
        for name, value in (
            ("collection_id", "other"),
            ("transport", "ringbuf"),
            ("data_slots", 31),
            ("read_slot", 1),
            ("rss_schema_bytes", 32),
            ("dtlb_schema_bytes", 72),
        ):
            with self.subTest(field=name), self.assertRaises(ValueError):
                matrix.validate_mpmo_status([{**status, name: value}], "test")
        for name in status:
            incomplete = status.copy()
            del incomplete[name]
            with self.subTest(missing=name), self.assertRaises(ValueError):
                matrix.validate_mpmo_status([incomplete], "test")

    def test_every_numeric_field_requires_nonnegative_uint64_not_bool_or_float(self):
        status = mpmo_status("test")
        for name, value in status.items():
            if type(value) is not int:
                continue
            for invalid in (-1, 1 << 64, True, False, 1.0, float("nan"), "1", None):
                with (
                    self.subTest(field=name, value=invalid),
                    self.assertRaisesRegex(ValueError, f"{name}.*uint64"),
                ):
                    matrix.validate_mpmo_status([{**status, name: invalid}], "test")

    def test_polls_reads_accounting_and_hard_errors(self):
        status = mpmo_status("test")
        cases = [({"polls": 0}, "positive polls")]
        for stream in ("rss", "dtlb"):
            cases.extend(
                [
                    (
                        {
                            f"{stream}_slot0_reads": 0,
                            f"{stream}_empty": 10,
                            f"{stream}_busy": 0,
                        },
                        f"positive {stream}_slot0_reads",
                    ),
                    ({f"{stream}_empty": 6}, f"{stream}.*must equal polls"),
                    ({f"{stream}_busy": 10}, f"{stream}.*must equal polls"),
                ]
            )
        cases.extend(
            ({name: 1}, f"{name} must be zero")
            for name in ("reader_errors", "rss_owner_misses", "dtlb_read_errors")
        )
        for change, message in cases:
            with (
                self.subTest(change=change),
                self.assertRaisesRegex(ValueError, message),
            ):
                matrix.validate_mpmo_status([{**status, **change}], "test")


class MatrixTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        (self.root / "config").mkdir()
        self.manifest = self.root / matrix.MANIFEST
        self.manifest.parent.mkdir(parents=True)
        self.manifest.write_text('{"format": "extracted-rss-dtlb"}')
        self.mpmo_manifest = (
            self.manifest.parent.with_name("rss_tlb_mpmo") / "manifest.json"
        )
        self.mpmo_manifest.parent.mkdir()
        self.mpmo_manifest.write_text(
            '{"transport": "mpmo", "data_slots": 32, "read_slot": 0}'
        )
        self.original_manifest = (
            self.manifest.parent.with_name("original") / "manifest.json"
        )
        self.original_manifest.parent.mkdir()
        library = self.original_manifest.parent / "library.so"
        library.write_bytes(b"frozen fixture library")
        source = self.original_manifest.parent / "probe.c"
        source.write_bytes(b"frozen original source")
        self.original_manifest.write_text(
            json.dumps(
                {
                    "version": 1,
                    "kind": "original-bcc-llvm",
                    "compiler_prefix": str(self.root / "private/install"),
                    "kernel_source": "/kernel-source",
                    "kernel_build": "/kernel-build",
                    "files": {
                        "bcc_library": {
                            "name": library.name,
                            "sha256": matrix.digest(library.read_bytes()),
                        }
                    },
                    "sources": {str(source): matrix.digest(source.read_bytes())},
                }
            )
        )
        self.directory = self.root / "data/experiments/example"
        self.bases = {}
        for thp in matrix.THPS:
            base = {
                "benchmark_config": {
                    "generic": {
                        "benchmark": "redis",
                        "cpus": 0,
                        "skip_clear_page_cache": False,
                        "transparent_hugepages": thp,
                        "overcommit_memory": "never_check",
                    },
                    "redis": {
                        "record_count": 1000,
                        "operation_count": 1000,
                        "read_proportion": 0.20,
                        "update_proportion": 0.10,
                        "insert_proportion": 0.10,
                        "rmw_proportion": 0.10,
                        "scan_proportion": 0.10,
                        "delete_proportion": 0.30,
                        "request_distribution": "zipfian",
                    },
                },
                "collector_config": {
                    "generic": {
                        "hooks": ["mm_rss_stat", "perf"],
                        "output_dir": "data",
                        "poll_rate": 0.1,
                        "output_graphs": False,
                    }
                },
            }
            self.bases[thp] = base
            (self.root / "config" / f"redis_{thp}.yaml").write_text(json.dumps(base))
        self.active = False
        self.exit_codes = []
        self.wait_error = None
        self.bad_phase = False
        self.mpmo_overrides = {}
        self.processes = []
        self.original_validator = mock.Mock(return_value=True)
        self.original_metadata = mock.Mock(side_effect=fake_original_manifest)
        patches = {
            "ROOT": self.root,
            "registry_names": mock.Mock(return_value=HOOKS.copy()),
            "configuration": mock.Mock(side_effect=fake_configuration),
            "read_rows": mock.Mock(side_effect=self.read_rows),
        }
        for name, value in patches.items():
            patcher = mock.patch.object(matrix, name, value)
            patcher.start()
            self.addCleanup(patcher.stop)
        modules = mock.patch.dict(
            sys.modules,
            {
                "yaml": SimpleNamespace(safe_load=json.loads),
                "data_schema": SimpleNamespace(
                    get_user_group_ids=lambda: (os.getuid(), os.getgid())
                ),
                "data_collection.bpf_instrumentation.original_rss_tlb": SimpleNamespace(
                    original_manifest=self.original_metadata,
                    validate_original_status=self.original_validator,
                ),
            },
        )
        modules.start()
        self.addCleanup(modules.stop)
        for target in ("os.chown", "os.fchown", "builtins.print"):
            patcher = mock.patch(target)
            patcher.start()
            self.addCleanup(patcher.stop)
        patcher = mock.patch.object(matrix.subprocess, "Popen", side_effect=self.launch)
        self.popen = patcher.start()
        self.addCleanup(patcher.stop)

    def plan(self, **kwargs):
        return matrix.make_plan(
            self.directory,
            kwargs.get("repeats", 30),
            kwargs.get("seed", 17),
            kwargs.get(
                "manifest",
                self.original_manifest
                if kwargs.get("original_only")
                else self.mpmo_manifest
                if kwargs.get("mpmo_only")
                else self.manifest,
            ),
            mpmo_only=kwargs.get("mpmo_only", False),
            original_only=kwargs.get("original_only", False),
            selected_control_only=kwargs.get("selected_control_only", False),
        )

    def run_matrix(self, **kwargs):
        manifest = kwargs.pop(
            "manifest",
            self.original_manifest
            if kwargs.get("original_only")
            else self.mpmo_manifest
            if kwargs.get("mpmo_only")
            else self.manifest,
        )
        return matrix.run_matrix(self.directory, seed=17, manifest=manifest, **kwargs)

    def events(self):
        return [
            json.loads(line)
            for line in (self.directory / "results.jsonl").read_text().splitlines()
        ]

    def status(self):
        return json.loads((self.directory / "manifest.json").read_text())

    def read_rows(self, path, **kwargs):
        self.assertFalse(
            self.active, "no DataFrame processing while the child is active"
        )
        rows = json.loads(Path(path).read_text())
        return rows[: kwargs["n_rows"]] if "n_rows" in kwargs else rows

    def fixture(self, cell, prefix, stdout):
        cid = prefix + "-" + UUID
        directory = (
            Path(cell["output_dir"])
            / ("baseline" if cell["mode"] == "none" else "curated")
            / "redis"
            / cid
        )
        directory.mkdir(parents=True, exist_ok=True)
        Path(stdout).write_text("collector output\n" + cid + "\n")
        redis = cell["config"]["benchmark_config"]["redis"]
        positions = [("load", outer, 0) for outer in range(redis["outer_repeat"])]
        positions += [
            ("run", outer, inner)
            for outer in range(redis["outer_repeat"])
            for inner in range(redis["repeat"])
        ]
        invocations, metrics = [], []
        for identifier, (phase, outer, inner) in enumerate(positions):
            invocations.append(
                {
                    "collection_id": cid,
                    "invocation_id": identifier,
                    "benchmark_name": "redis",
                    "run_label": prefix,
                    "phase": phase,
                    "outer_iteration": outer,
                    "iteration": inner,
                    "exit_code": 0,
                    "execution_state": "completed",
                    "parse_state": "ok",
                    "argv": [
                        "/ycsb/bin/ycsb",
                        phase,
                        "redis",
                        "-p",
                        "recordcount=1000",
                    ],
                }
            )
            for section, metric, value in (
                ("OVERALL", "RunTime(ms)", 123.0),
                ("OVERALL", "Throughput(ops/sec)", 45.0),
                ("READ", "Return=ERROR", 12.0),
            ):
                metrics.append(
                    {
                        "collection_id": cid,
                        "invocation_id": identifier,
                        "section": section,
                        "metric": metric,
                        "value": value,
                        "unit": "ms"
                        if metric == "RunTime(ms)"
                        else "ops/sec"
                        if metric == "Throughput(ops/sec)"
                        else None,
                    }
                )
        if self.bad_phase:
            invocations[0]["exit_code"] = 1
        tables = {
            "system_info": [
                {
                    "collection_id": cid,
                    "benchmark_name": "redis",
                    "return_code": 0,
                    "hooks": cell["expected_hooks"],
                    "transparent_hugepages": cell["thp"],
                    "hostname": "fixture",
                    "os": "Linux",
                    "kernel_version": "fixture-kernel",
                    "architecture": "x86_64",
                    "sockets": 1,
                    "cores": 4,
                    "logical_cores": 8,
                    "ram_bytes": 1 << 30,
                }
            ],
            "ycsb_invocations": invocations,
            "ycsb_metrics": metrics,
        }
        if cell["mode"] == matrix.SELECTED_CONTROL_MODE:
            tables["mm_rss_stat"] = [
                {
                    "collection_id": cid,
                    "pid": 3,
                    "tgid": 2,
                    "ts_ns": 1000,
                    "member": "MM_ANONPAGES",
                    "count": 1 << 33,
                }
            ]
            for name in matrix.SELECTED_CONTROL_EVENTS:
                tables[name] = [
                    {
                        "collection_id": cid,
                        "cpu": 1,
                        "pid": 3,
                        "tgid": 2,
                        "ts_uptime_us": 100,
                        f"cumulative_{name}": (1 << 32) - 1,
                        "pmu_enabled_time_us": 90,
                        "pmu_running_time_us": 80,
                    }
                ]
        elif cell["mode"] == "reduced":
            tables.update(
                policy_rss=[{"tgid": 2, "ts_ns": 1, "member": 1, "count": 12}],
                policy_dtlb=[{"dtlb_missrate": None}, {"dtlb_missrate": 0.25}],
                policy_status=[
                    {
                        "rss_dropped": 0,
                        "rss_owner_misses": 0,
                        "dtlb_dropped": 0,
                        "dtlb_read_errors": 0,
                        "rss_unattributed": 0,
                    }
                ],
            )
        elif cell["mode"] == "reduced-mpmo":
            tables["policy_status"] = [{**mpmo_status(cid), **self.mpmo_overrides}]
        elif cell["mode"] in matrix.ORIGINAL_MODES:
            evidence = self.root / "compiler-evidence" / cid
            evidence.mkdir(parents=True, exist_ok=True)
            transport = "mpmo" if cell["mode"] == "original-mpmo" else "capture"
            tables["policy_status"] = [original_status(cid, transport, evidence)]
            programs = {}
            for program in (
                "raw_tracepoint__rss_stat",
                "tracepoint__kmem__rss_stat",
                "dtlb_misses_on",
                "dtlb_accesses_on",
            ):
                module = "rss" if program.endswith("rss_stat") else "perf"
                name = f"{module}.{program}.bpf.bin"
                (evidence / name).write_bytes(b"\x95" + bytes(7))
                (evidence / f"{module}.after.json").write_text(
                    json.dumps({"version": 1, "transport": transport})
                )
                (evidence / f"{module}.generated.c").write_text("/* test source */\n")
                programs[program] = {
                    "module": module,
                    "name": name,
                    "size": 8,
                    "sha256": matrix.digest((evidence / name).read_bytes()),
                    "report_sha256": matrix.digest(
                        (evidence / f"{module}.after.json").read_bytes()
                    ),
                    "generated_source_sha256": matrix.digest(
                        (evidence / f"{module}.generated.c").read_bytes()
                    ),
                }
            manifest = cell["factory_metadata"]["manifest"]
            tables["policy_status"][0]["compiled_programs_json"] = json.dumps(
                {
                    "programs": programs,
                    "sources": manifest["sources"],
                    "bcc_library_sha256": manifest["files"]["bcc_library"]["sha256"],
                }
            )
            if transport == "capture":
                tables["policy_rss"] = [
                    {
                        "collection_id": cid,
                        "tgid": 2,
                        "ts_ns": 1,
                        "member": 1,
                        "count": 12,
                    }
                ]
                tables["original_perf"] = [
                    {
                        "collection_id": cid,
                        "counter": counter,
                        "cpu": 0,
                        "ts_uptime_us": 1,
                        "count": 2,
                        "enabled_time_us": 3,
                        "running_time_us": 3,
                    }
                    for counter in ("dtlb_misses", "dtlb_accesses")
                ]
        for name, rows in tables.items():
            (directory / f"{name}.end.parquet").write_text(json.dumps(rows))
        return {
            "stdout": str(stdout),
            "prefix": prefix,
            "collection_id": cid,
        }, directory

    def launch(self, argv, **kwargs):
        self.assertFalse(self.active, "collectors must be serial")
        self.assertEqual(self.status()["status"], "running")
        self.assertEqual(self.events()[-1]["status"], "running")
        self.assertEqual(
            argv[:4],
            [sys.executable, str(self.root / "python/kernmlops"), "collect", "data"],
        )
        self.assertEqual(kwargs["stdin"], subprocess.DEVNULL)
        self.assertTrue(kwargs["start_new_session"])
        self.assertEqual(kwargs["cwd"], self.root)
        self.assertTrue(kwargs["pass_fds"], "child must retain the experiment lock")
        for source in ("stdout", "stderr"):
            self.assertTrue(
                kwargs[source].seekable(), "logs must be regular files, not PIPEs"
            )
        prefix = argv[-1]
        config = json.loads(Path(argv[argv.index("--config-file") + 1]).read_text())
        generic = config["collector_config"]["generic"]
        mode = (
            (
                "original-mpmo"
                if generic["original_rss_tlb_transport"] == "mpmo"
                else "original-reduced"
            )
            if generic.get("original_rss_tlb")
            else "reduced-mpmo"
            if generic.get("extracted_rss_tlb_mpmo")
            else "reduced"
            if generic.get("extracted_rss_tlb")
            else matrix.SELECTED_CONTROL_MODE
            if generic.get("perf_events") is not None
            else ("all" if generic["hooks"] else "none")
        )
        thp = config["benchmark_config"]["generic"]["transparent_hugepages"]
        cell = json.loads((self.directory / "plan.json").read_text())["cells"][
            f"{thp}-{mode}"
        ]
        process = mock.Mock(pid=10000 + len(self.processes), returncode=None)
        self.processes.append(process)
        self.active = True

        def wait(timeout):
            if self.wait_error is not None:
                error, self.wait_error = self.wait_error, None
                raise error
            self.fixture(cell, prefix, kwargs["stdout"].name)
            process.returncode = self.exit_codes.pop(0) if self.exit_codes else 0
            self.active = False
            return process.returncode

        process.wait.side_effect = wait
        return process

    def test_six_cells_thirty_runs_balanced_reproducible_and_unique(self):
        plan = self.plan()
        self.assertEqual(matrix.MODES, ("none", "all", "reduced"))
        self.assertEqual(plan, self.plan(mpmo_only=False))
        self.assertEqual(len(plan["cells"]), 6)
        self.assertEqual(len(plan["jobs"]), 180)
        self.assertEqual(
            Counter(job["cell"] for job in plan["jobs"]),
            {cell: 30 for cell in plan["cells"]},
        )
        self.assertEqual(plan, self.plan())
        self.assertNotEqual(plan["jobs"], self.plan(seed=18)["jobs"])
        for replicate in range(30):
            jobs = plan["jobs"][replicate * 6 : (replicate + 1) * 6]
            self.assertEqual({job["cell"] for job in jobs}, set(plan["cells"]))
            self.assertEqual({job["replicate"] for job in jobs}, {replicate})
        labels = [
            f"{plan['experiment']}-{job['key']}-a{attempt:02d}"
            for job in plan["jobs"]
            for attempt in (1, 2)
        ]
        self.assertEqual(len(set(labels)), 360)

    def test_mpmo_only_has_two_balanced_cells_and_sixty_new_jobs(self):
        plan = self.plan(mpmo_only=True)
        self.assertEqual(
            set(plan["cells"]), {f"{thp}-reduced-mpmo" for thp in matrix.THPS}
        )
        self.assertEqual(len(plan["jobs"]), 60)
        self.assertEqual(len({job["key"] for job in plan["jobs"]}), 60)
        self.assertTrue(
            {job["key"] for job in plan["jobs"]}.isdisjoint(
                job["key"] for job in self.plan()["jobs"]
            )
        )
        self.assertEqual(
            Counter(job["cell"] for job in plan["jobs"]),
            {cell: 30 for cell in plan["cells"]},
        )
        self.assertEqual(plan, self.plan(mpmo_only=True))
        self.assertNotEqual(plan["jobs"], self.plan(mpmo_only=True, seed=18)["jobs"])
        self.assertEqual(plan["all_hooks"], HOOKS)
        self.assertEqual(plan["artifact"]["path"], str(self.mpmo_manifest))
        for replicate in range(30):
            jobs = plan["jobs"][replicate * 2 : (replicate + 1) * 2]
            self.assertEqual({job["thp"] for job in jobs}, set(matrix.THPS))
            self.assertEqual({job["replicate"] for job in jobs}, {replicate})
            self.assertEqual({job["mode"] for job in jobs}, {"reduced-mpmo"})
        for cell in plan["cells"].values():
            generic = cell["config"]["collector_config"]["generic"]
            self.assertEqual(generic["hooks"], [])
            self.assertIsNone(generic["extracted_rss_tlb"])
            self.assertEqual(generic["extracted_rss_tlb_mpmo"], str(self.mpmo_manifest))
            self.assertEqual(cell["expected_hooks"], [matrix.MPMO_HOOK])
            self.assertEqual(
                cell["factory_metadata"]["declared_hooks"], [matrix.MPMO_HOOK]
            )
            expected, _ = fake_configuration(self.bases[cell["thp"]])
            self.assertEqual(
                cell["config"]["benchmark_config"], expected["benchmark_config"]
            )
            for field in ("output_dir", "poll_rate", "output_graphs"):
                self.assertEqual(
                    generic[field], expected["collector_config"]["generic"][field]
                )

    def test_selected_control_has_sixty_balanced_jobs_and_never_reads_a_manifest(self):
        old_keys = {
            job["key"]
            for options in ({}, {"mpmo_only": True}, {"original_only": True})
            for job in self.plan(**options)["jobs"]
        }
        self.original_metadata.reset_mock()
        self.manifest.unlink()
        self.mpmo_manifest.write_text("not JSON")
        self.original_manifest.unlink()
        plan = self.plan(selected_control_only=True, manifest=None)
        self.assertEqual(matrix.SELECTED_CONTROL_MODE, "selected-control")
        self.assertEqual(
            set(plan["cells"]), {"always-selected-control", "never-selected-control"}
        )
        self.assertIsNone(plan["artifact"])
        self.assertEqual(plan["seed"], 17)
        self.assertEqual(len(plan["jobs"]), 60)
        self.assertEqual(len({job["key"] for job in plan["jobs"]}), 60)
        self.assertTrue(old_keys.isdisjoint(job["key"] for job in plan["jobs"]))
        self.assertEqual(
            Counter(job["thp"] for job in plan["jobs"]), {"always": 30, "never": 30}
        )
        self.assertEqual(
            plan, self.plan(selected_control_only=True, manifest=self.mpmo_manifest)
        )
        self.assertNotEqual(
            plan["jobs"], self.plan(selected_control_only=True, seed=18)["jobs"]
        )
        for replicate in range(30):
            jobs = plan["jobs"][2 * replicate : 2 * (replicate + 1)]
            self.assertEqual({job["replicate"] for job in jobs}, {replicate})
            self.assertEqual({job["thp"] for job in jobs}, set(matrix.THPS))
            self.assertEqual(
                {job["mode"] for job in jobs}, {matrix.SELECTED_CONTROL_MODE}
            )
        for cell in plan["cells"].values():
            generic = cell["config"]["collector_config"]["generic"]
            self.assertEqual(generic["hooks"], ["perf", "mm_rss_stat"])
            self.assertEqual(generic["perf_events"], ["dtlb_misses", "dtlb_accesses"])
            self.assertEqual(cell["expected_hooks"], ["perf", "mm_rss_stat"])
            self.assertEqual(
                cell["factory_metadata"],
                {"hooks": generic["hooks"], "perf_events": generic["perf_events"]},
            )
            expected, _ = fake_configuration(self.bases[cell["thp"]])
            self.assertEqual(
                cell["config"]["benchmark_config"], expected["benchmark_config"]
            )
            for name in (
                "output_dir",
                "poll_rate",
                "output_graphs",
                "original_rss_tlb_transport",
            ):
                self.assertEqual(
                    generic[name], expected["collector_config"]["generic"][name]
                )
            for selector in (
                "original_rss_tlb",
                "extracted_rss_tlb",
                "extracted_rss_tlb_mpmo",
            ):
                self.assertIsNone(generic[selector])
            matrix.check_hooks(cell["config"], cell["mode"], HOOKS, manifest=None)
        self.assertEqual(
            self.run_matrix(selected_control_only=True, manifest=None, limit=0), 0
        )
        self.assertEqual(
            self.run_matrix(selected_control_only=True, resume=True, limit=0), 0
        )
        self.original_metadata.assert_not_called()
        self.original_validator.assert_not_called()
        self.popen.assert_not_called()

    def test_selected_control_option_is_exclusive_and_cli_forwards_it(self):
        for option in ("mpmo_only", "original_only"):
            with (
                self.subTest(option=option),
                self.assertRaisesRegex(ValueError, "incompatible"),
            ):
                self.plan(selected_control_only=True, **{option: True})
            with self.assertRaises(SystemExit) as exc:
                matrix.main(
                    [
                        "--directory",
                        str(self.directory),
                        "--limit",
                        "0",
                        "--selected-control-only",
                        "--" + option.replace("_", "-"),
                    ]
                )
            self.assertEqual(exc.exception.code, 2)
        for flags in ([], ["--selected-control-only"]):
            with mock.patch.object(matrix, "run_matrix", return_value=0) as run:
                self.assertEqual(
                    matrix.main(
                        [
                            "--directory",
                            str(self.directory),
                            "--seed",
                            "17",
                            "--limit",
                            "0",
                            *flags,
                        ]
                    ),
                    0,
                )
            self.assertEqual(run.call_args.kwargs["selected_control_only"], bool(flags))
            self.assertEqual(run.call_args.kwargs["seed"], 17)
            self.assertEqual(run.call_args.kwargs["repeats"], 30)
            self.assertFalse(run.call_args.kwargs["mpmo_only"])
            self.assertFalse(run.call_args.kwargs["original_only"])
        self.popen.assert_not_called()

    def test_selected_control_rejects_partial_or_extra_hooks_events_and_selectors(self):
        cell = self.plan(selected_control_only=True)["cells"]["always-selected-control"]
        cases = (
            [
                ("hooks", hooks)
                for hooks in (
                    [],
                    ["perf"],
                    ["mm_rss_stat"],
                    ["mm_rss_stat", "perf"],
                    ["perf", "mm_rss_stat", "cbmm"],
                    HOOKS,
                )
            ]
            + [
                ("perf_events", events)
                for events in (
                    None,
                    [],
                    ["dtlb_misses"],
                    ["dtlb_accesses"],
                    ["dtlb_misses"] * 2,
                    ["dtlb_accesses", "dtlb_misses"],
                    [*matrix.SELECTED_CONTROL_EVENTS, "itlb_misses"],
                    ["unknown"],
                )
            ]
            + [
                (selector, str(self.manifest))
                for selector in (
                    "original_rss_tlb",
                    "extracted_rss_tlb",
                    "extracted_rss_tlb_mpmo",
                )
            ]
            + [("original_rss_tlb_transport", "mpmo")]
        )
        for name, value in cases:
            config = deepcopy(cell["config"])
            config["collector_config"]["generic"][name] = value
            with self.subTest(name=name, value=value), self.assertRaises(ValueError):
                matrix.check_hooks(config, cell["mode"], HOOKS, None)
        with self.assertRaisesRegex(ValueError, "registry order"):
            matrix.check_hooks(
                cell["config"], cell["mode"], list(reversed(HOOKS)), None
            )
        for change in ("events", "metadata"):

            def wrong_configuration(raw):
                config, metadata = fake_configuration(raw)
                if change == "events":
                    config["collector_config"]["generic"]["perf_events"] = None
                else:
                    metadata["manifest"] = {}
                return config, metadata

            matrix.configuration.side_effect = wrong_configuration
            with (
                self.subTest(change=change),
                self.assertRaisesRegex(ValueError, "perf_events|metadata"),
            ):
                self.run_matrix(selected_control_only=True, limit=0)
        self.assertFalse((self.directory / "plan.json").exists())
        self.popen.assert_not_called()

    def test_other_modes_clear_event_selection_and_accept_legacy_missing_field(self):
        for thp, base in self.bases.items():
            base["collector_config"]["generic"]["perf_events"] = ["dtlb_misses"]
            (self.root / "config" / f"redis_{thp}.yaml").write_text(json.dumps(base))
        for options in ({}, {"mpmo_only": True}, {"original_only": True}):
            plan = self.plan(**options)
            for cell in plan["cells"].values():
                generic = cell["config"]["collector_config"]["generic"]
                self.assertIsNone(generic.pop("perf_events"))
                matrix.check_hooks(
                    cell["config"], cell["mode"], HOOKS, plan["artifact"]["path"]
                )
                generic["perf_events"] = ["dtlb_misses"]
                with self.assertRaisesRegex(ValueError, "perf_events"):
                    matrix.check_hooks(
                        cell["config"], cell["mode"], HOOKS, plan["artifact"]["path"]
                    )

    def test_selected_control_uses_normal_environment_without_silently_changing_flags(
        self,
    ):
        environment = {
            "PYTHONPATH": "/project:/user",
            "LD_LIBRARY_PATH": "/system/lib",
            "UNCHANGED": "yes",
        }
        with mock.patch.dict(os.environ, environment, clear=True):
            self.assertEqual(self.run_matrix(selected_control_only=True, limit=2), 0)
            self.assertEqual(dict(os.environ), environment)
            self.assertTrue(
                all(call.kwargs["env"] is None for call in self.popen.call_args_list)
            )
            cell = self.plan(selected_control_only=True)["cells"][
                "always-selected-control"
            ]
            for name in (
                "KERNMLOPS_BCC_REDUCE",
                "KERNMLOPS_BCC_EXPORT_ONLY",
                "KERNMLOPS_BCC_EVIDENCE",
            ):
                with mock.patch.dict(os.environ, {name: "user-choice"}):
                    before = dict(os.environ)
                    with self.assertRaisesRegex(ValueError, name):
                        matrix.child_environment(cell)
                    self.assertIsNone(matrix.child_environment({"mode": "all"}))
                    self.assertEqual(dict(os.environ), before)
        self.original_metadata.assert_not_called()
        self.original_validator.assert_not_called()

    def test_selected_control_freezes_original_sources_and_schema_before_each_launch(
        self,
    ):
        sources = [
            self.root / "python/kernmlops" / name
            for name in (
                "data_collection/bpf_instrumentation/bpf/mm_trace_rss_stat.bpf.c",
                "data_collection/bpf_instrumentation/bpf/perf.bpf.c",
                "data_collection/bpf_instrumentation/mm_rss_stat.py",
                "data_collection/bpf_instrumentation/perf/perf_hook.py",
                "data_schema/generic_table.py",
                "data_schema/perf/__init__.py",
                "data_schema/perf/perf_schema.py",
                "data_schema/perf/tlb_perf.py",
            )
        ]
        for path in sources:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("original source fixture\n")
        self.assertEqual(self.run_matrix(selected_control_only=True, limit=0), 0)
        frozen = (self.directory / "plan.json").read_bytes()
        self.assertEqual(
            set(json.loads(frozen)["collector_sources"]),
            {str(path.relative_to(self.root)) for path in sources},
        )
        self.assertFalse(
            any("data_schema" in name for name in self.plan()["collector_sources"])
        )
        for path in sources:
            before = path.read_bytes()
            path.write_bytes(before + b"changed\n")
            with (
                self.subTest(path=path),
                self.assertRaisesRegex(ValueError, "frozen plan"),
            ):
                self.run_matrix(selected_control_only=True, resume=True, limit=0)
            path.write_bytes(before)
            self.assertEqual((self.directory / "plan.json").read_bytes(), frozen)
        self.popen.assert_not_called()
        validate = matrix.validate_result

        def tamper_after_exit(row, cell):
            validate(row, cell)
            sources[-1].write_text("changed schema\n")

        with mock.patch.object(
            matrix, "validate_result", side_effect=tamper_after_exit
        ):
            with self.assertRaisesRegex(ValueError, "frozen plan"):
                self.run_matrix(selected_control_only=True, resume=True, limit=2)
        self.assertEqual(self.popen.call_count, 1)
        self.assertEqual(self.status()["status"], "blocked")

    def test_selected_control_requires_full_records_in_all_three_streams(self):
        cell = self.plan(selected_control_only=True)["cells"]["always-selected-control"]
        row, directory = self.fixture(cell, "control-widths", self.root / "stdout.log")
        matrix.validate_result(row, cell)
        self.assertNotIn("telemetry_status", row)
        self.assertNotIn("compiler_evidence_dir", row)
        self.assertEqual(
            set(row["tables"]),
            {
                "system_info",
                "ycsb_invocations",
                "ycsb_metrics",
                "mm_rss_stat",
                *matrix.SELECTED_CONTROL_EVENTS,
            },
        )
        for name in ("mm_rss_stat", *matrix.SELECTED_CONTROL_EVENTS):
            path = directory / f"{name}.end.parquet"
            before = path.read_text()
            samples = json.loads(before)
            self.assertEqual(row["tables"][name], [str(path)])
            path.unlink()
            with self.assertRaisesRegex(ValueError, f"no {name} samples"):
                matrix.validate_result(row, cell)
            path.write_text("[]")
            with self.assertRaisesRegex(ValueError, f"no {name} samples"):
                matrix.validate_result(row, cell)
            for field, value in samples[0].items():
                invalid = samples[0].copy()
                del invalid[field]
                path.write_text(json.dumps([invalid]))
                with (
                    self.subTest(table=name, missing=field),
                    self.assertRaises(ValueError),
                ):
                    matrix.validate_result(row, cell)
                for bad in (
                    (
                        -1,
                        True,
                        1.0,
                        None,
                        1
                        << (
                            32 if field in ("pid", "tgid", f"cumulative_{name}") else 64
                        ),
                    )
                    if type(value) is int
                    else ("wrong", 1, None)
                ):
                    path.write_text(json.dumps([{**samples[0], field: bad}]))
                    with (
                        self.subTest(table=name, field=field, value=bad),
                        self.assertRaises(ValueError),
                    ):
                        matrix.validate_result(row, cell)
            path.write_text("[]")
            chunk = directory / f"{name}.0.parquet"
            chunk.write_text(before)
            matrix.validate_result(row, cell)
            self.assertEqual(row["tables"][name], [str(chunk), str(path)])
            path.write_text(json.dumps([{**samples[0], "collection_id": "wrong"}]))
            with self.assertRaisesRegex(ValueError, "collection ID mismatch"):
                matrix.validate_result(row, cell)
            path.write_text(before)
            matrix.validate_result(row, cell)

    def test_selected_control_rejects_even_empty_extra_probe_tables(self):
        cell = self.plan(selected_control_only=True)["cells"]["never-selected-control"]
        row, directory = self.fixture(cell, "control-extra", self.root / "stdout.log")
        for name in (
            "itlb_misses",
            "tlb_flushes",
            "dtlb_walk_duration",
            "policy_rss",
            "policy_dtlb",
            "original_perf",
            "policy_status",
            "cbmm_eager",
            "memory_usage",
        ):
            for chunk in ("0", "end"):
                path = directory / f"{name}.{chunk}.parquet"
                path.write_text("[]")
                with (
                    self.subTest(path=path),
                    self.assertRaisesRegex(ValueError, f"unexpected {name}"),
                ):
                    matrix.validate_result(row, cell)
                path.unlink()
        matrix.validate_result(row, cell)

    def test_selected_control_validates_real_full_parquet_frames_and_rejects_partial(
        self,
    ):
        try:
            import polars as pl
        except ImportError:
            self.skipTest("Polars is available in the collector Docker image")
        cell = self.plan(selected_control_only=True)["cells"]["always-selected-control"]
        row, directory = self.fixture(cell, "control-parquet", self.root / "stdout.log")
        for path in directory.glob("*.parquet"):
            pl.from_dicts(json.loads(path.read_text())).write_parquet(path)
        with mock.patch.object(matrix, "read_rows", side_effect=READ_ROWS):
            matrix.validate_result(row, cell)
            for name in ("mm_rss_stat", *matrix.SELECTED_CONTROL_EVENTS):
                path = directory / f"{name}.end.parquet"
                frame = pl.read_parquet(path)
                for field in frame.columns:
                    frame.drop(field).write_parquet(path)
                    with (
                        self.subTest(table=name, missing=field),
                        self.assertRaises(ValueError),
                    ):
                        matrix.validate_result(row, cell)
                frame.write_parquet(path)

    def test_selected_control_resume_revalidates_and_never_rewrites_old_captures(self):
        self.assertEqual(self.run_matrix(selected_control_only=True, limit=1), 0)
        first = self.events()[-1]
        frozen = (self.directory / "plan.json").read_bytes()
        with self.assertRaisesRegex(ValueError, "frozen plan"):
            self.run_matrix(resume=True, limit=1)
        self.assertEqual(
            self.run_matrix(selected_control_only=True, resume=True, limit=1), 0
        )
        self.assertEqual(self.events()[1], first)
        self.assertEqual(self.status()["completed"], 2)
        self.assertEqual(self.status()["total"], 60)
        self.assertEqual(self.popen.call_count, 2)
        Path(first["tables"]["dtlb_accesses"][0]).write_text("[]")
        old_files = {
            path: path.read_bytes() for path in Path(first["collection_dir"]).iterdir()
        }
        self.assertEqual(
            self.run_matrix(selected_control_only=True, resume=True, limit=0), 0
        )
        self.assertEqual(self.events()[-1]["status"], "invalidated")
        self.assertEqual(self.status()["completed"], 1)
        self.assertEqual(
            self.run_matrix(selected_control_only=True, resume=True, limit=1), 0
        )
        retried = self.events()[-1]
        self.assertEqual(retried["key"], first["key"])
        self.assertEqual(retried["attempt"], 2)
        self.assertNotEqual(retried["collection_id"], first["collection_id"])
        self.assertEqual({path: path.read_bytes() for path in old_files}, old_files)
        self.assertEqual((self.directory / "plan.json").read_bytes(), frozen)
        self.assertEqual(self.status()["completed"], 2)
        self.assertEqual(self.popen.call_count, 3)

    def test_selected_control_never_extends_or_modifies_historical_experiments(self):
        for options in ({}, {"mpmo_only": True}, {"original_only": True}):
            self.directory = (
                self.root
                / "data/experiments"
                / (next(iter(options)) if options else "default")
            )
            self.assertEqual(self.run_matrix(repeats=1, **options), 0)
            before = {
                path: path.read_bytes()
                for path in self.root.rglob("*")
                if path.is_file()
            }
            for resume in (False, True):
                with self.assertRaises(ValueError):
                    self.run_matrix(
                        selected_control_only=True, repeats=1, resume=resume
                    )
                self.assertEqual({path: path.read_bytes() for path in before}, before)
        before = {
            path: path.read_bytes() for path in self.root.rglob("*") if path.is_file()
        }
        self.directory = self.root / "data/experiments/new-selected-control"
        self.assertEqual(self.run_matrix(selected_control_only=True, limit=2), 0)
        self.assertEqual({path: path.read_bytes() for path in before}, before)

    def test_config_workload_unchanged_and_no_cbmm(self):
        for cell in self.plan()["cells"].values():
            expected, _ = fake_configuration(self.bases[cell["thp"]])
            self.assertEqual(
                cell["config"]["benchmark_config"], expected["benchmark_config"]
            )
            generic = cell["config"]["collector_config"]["generic"]
            self.assertNotIn("cbmm", generic["hooks"])
            self.assertEqual(generic["hooks"], HOOKS if cell["mode"] == "all" else [])
            self.assertEqual(
                generic["extracted_rss_tlb"],
                str(self.manifest) if cell["mode"] == "reduced" else None,
            )
            self.assertIsNone(generic["extracted_rss_tlb_mpmo"])
            self.assertIsNone(generic["original_rss_tlb"])
            self.assertEqual(generic["original_rss_tlb_transport"], "capture")
            self.assertEqual(
                cell["factory_metadata"]["verification"], "declaration_only"
            )
        self.manifest.unlink()
        with self.assertRaises(FileNotFoundError):
            self.plan()

    def test_selectors_are_exclusive_and_match_mode_and_manifest(self):
        for options in ({}, {"mpmo_only": True}, {"original_only": True}):
            plan = self.plan(**options)
            for cell in plan["cells"].values():
                matrix.check_hooks(
                    cell["config"], cell["mode"], HOOKS, plan["artifact"]["path"]
                )
                for selector in (
                    "extracted_rss_tlb",
                    "extracted_rss_tlb_mpmo",
                    "original_rss_tlb",
                ):
                    config = deepcopy(cell["config"])
                    config["collector_config"]["generic"][selector] = (
                        "/wrong/manifest.json"
                    )
                    with self.subTest(mode=cell["mode"], selector=selector):
                        with self.assertRaisesRegex(ValueError, selector):
                            matrix.check_hooks(
                                config, cell["mode"], HOOKS, plan["artifact"]["path"]
                            )
        cell = self.plan(mpmo_only=True)["cells"]["always-reduced-mpmo"]
        for hooks in ([matrix.MPMO_HOOK], ["mm_rss_stat"], ["cbmm"]):
            config = deepcopy(cell["config"])
            config["collector_config"]["generic"]["hooks"] = hooks
            with self.subTest(hooks=hooks), self.assertRaises(ValueError):
                matrix.check_hooks(config, cell["mode"], HOOKS, self.mpmo_manifest)
        config = deepcopy(cell["config"])
        config["collector_config"]["generic"]["extracted_rss_tlb_mpmo"] = None
        with self.assertRaisesRegex(ValueError, "extracted_rss_tlb_mpmo"):
            matrix.check_hooks(config, cell["mode"], HOOKS, self.mpmo_manifest)

    def test_effective_config_cannot_reintroduce_the_other_selector(self):
        def mixed_configuration(raw):
            config, metadata = fake_configuration(raw)
            config["collector_config"]["generic"]["extracted_rss_tlb"] = str(
                self.manifest
            )
            return config, metadata

        matrix.configuration.side_effect = mixed_configuration
        with self.assertRaisesRegex(ValueError, "extracted_rss_tlb manifest mismatch"):
            self.run_matrix(mpmo_only=True, limit=0)
        self.assertFalse((self.directory / "plan.json").exists())
        self.popen.assert_not_called()

    def test_original_factory_cannot_change_transport_or_metadata(self):
        for field, value in (
            ("original_rss_tlb_transport", "ringbuf"),
            ("hooks", [matrix.EXTRACTED_HOOK]),
            ("extracted_rss_tlb", str(self.manifest)),
        ):

            def wrong_configuration(raw):
                config, metadata = fake_configuration(raw)
                config["collector_config"]["generic"][field] = value
                return config, metadata

            matrix.configuration.side_effect = wrong_configuration
            with self.subTest(field=field), self.assertRaises(ValueError):
                self.plan(original_only=True)

        def wrong_metadata(raw):
            config, metadata = fake_configuration(raw)
            metadata["manifest"]["compiler_prefix"] = "/not/frozen"
            return config, metadata

        matrix.configuration.side_effect = wrong_metadata
        with self.assertRaisesRegex(ValueError, "factory manifest mismatch"):
            self.plan(original_only=True)
        self.popen.assert_not_called()

    def test_wrong_bundle_factory_errors_prevent_freezing_or_launch(self):
        for mpmo_only, manifest in ((True, self.manifest), (False, self.mpmo_manifest)):
            with (
                self.subTest(mpmo_only=mpmo_only),
                self.assertRaisesRegex(ValueError, "factory rejected bundle"),
            ):
                self.run_matrix(mpmo_only=mpmo_only, manifest=manifest, limit=0)
            self.assertFalse((self.directory / "plan.json").exists())
        self.popen.assert_not_called()

    def test_both_plans_require_twelve_unique_hooks_without_cbmm_or_registry_additions(
        self,
    ):
        for names in (
            HOOKS + [matrix.MPMO_HOOK],
            HOOKS[:-1],
            HOOKS[:-1] + [HOOKS[0]],
            HOOKS[:-1] + ["cbmm"],
        ):
            matrix.registry_names.return_value = names
            for mpmo_only in (False, True):
                with (
                    self.subTest(names=names, mpmo_only=mpmo_only),
                    self.assertRaisesRegex(ValueError, "exactly 12"),
                ):
                    self.plan(mpmo_only=mpmo_only)

    def test_cli_forwards_mpmo_option_and_explicit_manifest_without_launch(self):
        for flags in ([], ["--mpmo-only"]):
            with mock.patch.object(matrix, "run_matrix", return_value=0) as run:
                self.assertEqual(
                    matrix.main(
                        [
                            "--directory",
                            str(self.directory),
                            "--manifest",
                            str(self.mpmo_manifest),
                            "--limit",
                            "0",
                            *flags,
                        ]
                    ),
                    0,
                )
            self.assertEqual(run.call_args.kwargs["mpmo_only"], bool(flags))
            self.assertEqual(run.call_args.kwargs["manifest"], self.mpmo_manifest)
            self.assertEqual(run.call_args.kwargs["limit"], 0)
        self.popen.assert_not_called()

    def test_registry_snapshot_never_instantiates_any_hook(self):
        factories = {
            name: mock.Mock(side_effect=AssertionError("must not instantiate hooks"))
            for name in HOOKS + ["cbmm"]
        }
        with mock.patch.dict(
            sys.modules,
            {
                "data_collection": SimpleNamespace(),
                "data_collection.bpf_instrumentation": SimpleNamespace(
                    all_hooks=factories
                ),
            },
        ):
            self.assertEqual(REGISTRY_NAMES(), HOOKS)
        for factory in factories.values():
            factory.assert_not_called()

    def test_preflight_and_resume_reject_frozen_input_changes(self):
        self.assertEqual(self.run_matrix(limit=0), 0)
        self.popen.assert_not_called()
        self.assertEqual(self.status()["status"], "paused")
        plan = json.loads((self.directory / "plan.json").read_text())
        matrix.validate_plan(plan, self.plan())
        for changed in (self.plan(seed=18), self.plan(repeats=29)):
            with self.assertRaisesRegex(ValueError, "frozen plan"):
                matrix.validate_plan(plan, changed)
        for path in (self.manifest, self.root / "config/redis_always.yaml"):
            before = path.read_bytes()
            path.write_bytes(before + b"\n")
            with self.assertRaisesRegex(ValueError, "frozen plan"):
                self.run_matrix(resume=True, limit=0)
            path.write_bytes(before)
        cell = plan["cells"]["always-all"]
        config = deepcopy(cell["config"])
        config["collector_config"]["generic"]["hooks"].append("cbmm")
        Path(cell["config_path"]).write_bytes(matrix.encoded(config))
        with self.assertRaisesRegex(ValueError, "hash mismatch"):
            self.run_matrix(resume=True, limit=0)
        with self.assertRaisesRegex(ValueError, "cbmm"):
            matrix.check_hooks(config, "all", HOOKS, self.manifest)
        self.popen.assert_not_called()

    def test_serial_fresh_processes_and_post_exit_validation(self):
        self.assertEqual(self.run_matrix(limit=6), 0)
        self.assertEqual(self.popen.call_count, 6)
        self.assertEqual(len({id(process) for process in self.processes}), 6)
        self.assertEqual(
            [row["status"] for row in self.events()], ["running", "success"] * 6
        )
        self.assertEqual(self.status()["completed"], 6)
        for process in self.processes:
            process.wait.assert_called_once_with(timeout=300)
        self.assertTrue(
            all(call.kwargs["env"] is None for call in self.popen.call_args_list)
        )

    def test_original_four_cells_120_jobs_are_balanced_and_distinct_from_archived_reduced(
        self,
    ):
        plan = self.plan(original_only=True)
        self.assertEqual(
            set(plan["cells"]),
            {f"{thp}-{mode}" for thp in matrix.THPS for mode in matrix.ORIGINAL_MODES},
        )
        self.assertEqual(len(plan["jobs"]), 120)
        self.assertEqual(
            Counter(job["cell"] for job in plan["jobs"]),
            {cell: 30 for cell in plan["cells"]},
        )
        self.assertEqual(plan, self.plan(original_only=True))
        self.assertNotEqual(
            plan["jobs"], self.plan(original_only=True, seed=18)["jobs"]
        )
        self.assertTrue(
            {job["key"] for job in plan["jobs"]}.isdisjoint(
                job["key"] for job in self.plan()["jobs"]
            )
        )
        for replicate in range(30):
            jobs = plan["jobs"][4 * replicate : 4 * (replicate + 1)]
            self.assertEqual({job["cell"] for job in jobs}, set(plan["cells"]))
            self.assertEqual({job["replicate"] for job in jobs}, {replicate})
        for cell in plan["cells"].values():
            generic = cell["config"]["collector_config"]["generic"]
            self.assertEqual(generic["hooks"], [])
            self.assertIsNone(generic["extracted_rss_tlb"])
            self.assertIsNone(generic["extracted_rss_tlb_mpmo"])
            self.assertEqual(generic["original_rss_tlb"], str(self.original_manifest))
            self.assertEqual(
                cell["expected_hooks"], [matrix.ORIGINAL_HOOKS[cell["mode"]]]
            )
            self.assertEqual(
                cell["factory_metadata"]["manifest"],
                fake_original_manifest(self.original_manifest),
            )
            self.assertEqual(
                cell["config"]["benchmark_config"],
                fake_configuration(self.bases[cell["thp"]])[0]["benchmark_config"],
            )

    def test_original_flag_excludes_draft_and_cli_defaults_remain_legacy(self):
        with self.assertRaisesRegex(ValueError, "incompatible"):
            self.plan(original_only=True, mpmo_only=True)
        with self.assertRaises(SystemExit) as exc:
            matrix.main(
                [
                    "--directory",
                    str(self.directory),
                    "--original-only",
                    "--mpmo-only",
                    "--limit",
                    "0",
                ]
            )
        self.assertEqual(exc.exception.code, 2)
        for flags in (
            [],
            ["--original-only", "--manifest", str(self.original_manifest)],
        ):
            with mock.patch.object(matrix, "run_matrix", return_value=0) as run:
                self.assertEqual(
                    matrix.main(
                        ["--directory", str(self.directory), "--limit", "0", *flags]
                    ),
                    0,
                )
            self.assertEqual(run.call_args.kwargs["original_only"], bool(flags))
            self.assertEqual(
                run.call_args.kwargs["manifest"],
                self.original_manifest if flags else matrix.MANIFEST,
            )
        self.popen.assert_not_called()

    def test_private_bcc_environment_is_child_only_and_preserves_user_variables(self):
        environment = {
            "PYTHONPATH": "/project:/user",
            "LD_LIBRARY_PATH": "/user/lib",
            "BCC_KERNEL_SOURCE": "/user/source",
            "BCC_KERNEL_BUILD": "/user/build",
            "UNCHANGED": "yes",
            "KERNMLOPS_BCC_REDUCE": "user-choice",
        }
        with mock.patch.dict(os.environ, environment, clear=True):
            self.assertEqual(self.run_matrix(original_only=True, limit=4), 0)
            self.assertEqual(dict(os.environ), environment)
        for call in self.popen.call_args_list:
            child = call.kwargs["env"]
            self.assertEqual(
                child["LD_LIBRARY_PATH"],
                str(self.root / "private/install/lib") + ":/user/lib",
            )
            self.assertEqual(
                child["PYTHONPATH"],
                str(self.root / "private/install/lib/python3/site-packages")
                + ":/project:/user",
            )
            self.assertEqual(child["BCC_KERNEL_SOURCE"], "/kernel-source")
            self.assertEqual(child["BCC_KERNEL_BUILD"], "/kernel-build")
            self.assertEqual(child["KERNMLOPS_BCC_REDUCE"], "user-choice")
            self.assertEqual(child["UNCHANGED"], "yes")
        with mock.patch.dict(os.environ, {}, clear=True):
            for cell in self.plan(original_only=True)["cells"].values():
                self.assertNotIn("KERNMLOPS_BCC_REDUCE", matrix.child_environment(cell))
            for plan in (self.plan(), self.plan(mpmo_only=True)):
                for cell in plan["cells"].values():
                    self.assertIsNone(matrix.child_environment(cell))

    def test_original_status_api_capture_streams_and_saved_evidence(self):
        self.assertEqual(self.run_matrix(original_only=True, limit=4), 0)
        self.assertEqual(self.original_validator.call_count, 4)
        for row in self.events()[1::2]:
            transport = "mpmo" if row["mode"] == "original-mpmo" else "capture"
            self.original_validator.assert_any_call(
                row["telemetry_status"], row["collection_id"], transport
            )
            self.assertEqual(
                row["compiler_evidence_dir"],
                row["telemetry_status"][0]["compiler_evidence"],
            )
            self.assertNotIn("policy_dtlb", row["tables"])
            self.assertEqual("original_perf" in row["tables"], transport == "capture")
            self.assertEqual("policy_rss" in row["tables"], transport == "capture")
        self.assertEqual(self.run_matrix(original_only=True, resume=True, limit=1), 0)
        self.assertEqual(self.status()["completed"], 5)
        self.assertEqual(self.status()["total"], 120)
        self.assertEqual(self.popen.call_count, 5)

    def test_original_status_rejection_and_final_only_status_without_fake_hard_error_counters(
        self,
    ):
        for mode in matrix.ORIGINAL_MODES:
            cell = self.plan(original_only=True)["cells"][f"always-{mode}"]
            row, directory = self.fixture(cell, mode, self.root / f"{mode}.stdout.log")
            matrix.validate_result(row, cell)
            self.assertNotIn("rss_owner_misses", row["telemetry_status"][0])
            self.assertNotIn("dtlb_read_errors", row["telemetry_status"][0])
            self.original_validator.side_effect = ValueError("bad original status")
            with self.assertRaisesRegex(ValueError, "bad original status"):
                matrix.validate_result(row, cell)
            self.original_validator.side_effect = None
            path = directory / "policy_status.0.parquet"
            path.write_text("[]")
            with self.assertRaisesRegex(ValueError, "only final original"):
                matrix.validate_result(row, cell)
            path.unlink()
            if mode == "original-mpmo":
                for name in ("policy_rss", "policy_dtlb", "original_perf"):
                    path = directory / f"{name}.end.parquet"
                    path.write_text("[]")
                    with self.assertRaisesRegex(ValueError, f"unexpected {name}"):
                        matrix.validate_result(row, cell)
                    path.unlink()

    def test_original_capture_requires_both_perf_counters_and_exact_value_widths(self):
        cell = self.plan(original_only=True)["cells"]["always-original-reduced"]
        row, directory = self.fixture(cell, "original-widths", self.root / "stdout.log")
        path = directory / "original_perf.end.parquet"
        samples = json.loads(path.read_text())
        for counter in ("dtlb_misses", "dtlb_accesses"):
            path.write_text(json.dumps([r for r in samples if r["counter"] == counter]))
            with self.assertRaisesRegex(ValueError, "every original stream"):
                matrix.validate_result(row, cell)
        for name, invalid in (
            ("count", 1 << 32),
            ("count", 2.0),
            ("cpu", -1),
            ("ts_uptime_us", 1 << 64),
        ):
            path.write_text(json.dumps([{**samples[0], name: invalid}, samples[1]]))
            with self.assertRaisesRegex(ValueError, "width"):
                matrix.validate_result(row, cell)
        path.write_text(
            json.dumps(
                [
                    {**r, "count": (1 << 32) - 1, "ts_uptime_us": (1 << 64) - 1}
                    for r in samples
                ]
            )
        )
        matrix.validate_result(row, cell)

    def test_original_bundle_sources_metadata_and_every_launch_remain_frozen(self):
        self.assertEqual(self.run_matrix(original_only=True, limit=0), 0)
        plan = (self.directory / "plan.json").read_bytes()
        for path in (
            self.original_manifest.parent / "library.so",
            self.original_manifest.parent / "probe.c",
        ):
            before = path.read_bytes()
            path.write_bytes(before + b"changed")
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                self.run_matrix(original_only=True, resume=True, limit=0)
            path.write_bytes(before)
            self.assertEqual((self.directory / "plan.json").read_bytes(), plan)
        manifest = self.original_manifest.read_bytes()
        changed = json.loads(manifest)
        changed["compiler_prefix"] = "/different/install"
        self.original_manifest.write_text(json.dumps(changed))
        with self.assertRaisesRegex(ValueError, "frozen plan"):
            self.run_matrix(original_only=True, resume=True, limit=0)
        changed["compiler_prefix"] = "relative/install"
        self.original_manifest.write_text(json.dumps(changed))
        with self.assertRaisesRegex(ValueError, "absolute path"):
            self.plan(original_only=True)
        self.original_manifest.write_bytes(manifest)
        validate = matrix.validate_result

        def tamper(row, cell):
            validate(row, cell)
            changed["compiler_prefix"] = "/different/install"
            self.original_manifest.write_text(json.dumps(changed))

        with mock.patch.object(matrix, "validate_result", side_effect=tamper):
            with self.assertRaisesRegex(ValueError, "frozen plan"):
                self.run_matrix(original_only=True, resume=True, limit=2)
        self.assertEqual(self.popen.call_count, 1)
        self.assertEqual(self.status()["status"], "blocked")

    def test_existing_default_experiment_is_not_extended_for_original_modes(self):
        self.assertEqual(self.run_matrix(repeats=1), 0)
        frozen = {
            path: path.read_bytes()
            for path in self.directory.rglob("*")
            if path.is_file()
        }
        for resume in (False, True):
            with self.assertRaises(ValueError):
                self.run_matrix(original_only=True, repeats=1, resume=resume)
            self.assertEqual({path: path.read_bytes() for path in frozen}, frozen)
        self.assertEqual(self.popen.call_count, 6)

    def test_mpmo_runs_write_status_only_telemetry_and_resume_keeps_option(self):
        self.assertEqual(self.run_matrix(mpmo_only=True, limit=1), 0)
        first = self.events()[-1]
        self.assertEqual(first["mode"], "reduced-mpmo")
        self.assertEqual(
            first["telemetry_status"], [mpmo_status(first["collection_id"])]
        )
        self.assertEqual(
            set(first["tables"]),
            {
                "system_info",
                "ycsb_invocations",
                "ycsb_metrics",
                "policy_status",
            },
        )
        self.assertIn("/curated/redis/", first["collection_dir"])
        frozen = (self.directory / "plan.json").read_bytes()
        with self.assertRaisesRegex(ValueError, "frozen plan"):
            self.run_matrix(resume=True, limit=1)
        self.assertEqual(self.popen.call_count, 1)
        self.assertEqual(self.run_matrix(mpmo_only=True, resume=True, limit=1), 0)
        self.assertEqual(self.popen.call_count, 2)
        self.assertEqual(self.status()["total"], 60)
        self.assertEqual(self.status()["completed"], 2)
        self.assertEqual(self.events()[1], first)
        self.assertEqual({row["mode"] for row in self.events()}, {"reduced-mpmo"})
        self.assertEqual({row["attempt"] for row in self.events()}, {1})
        self.assertEqual((self.directory / "plan.json").read_bytes(), frozen)

    def test_mpmo_failed_validation_records_status_and_resume_revalidates_it(self):
        self.mpmo_overrides = {"rss_slot0_reads": 0, "rss_empty": 9}
        self.assertEqual(self.run_matrix(mpmo_only=True, limit=1), 1)
        failed = self.events()[-1]
        self.assertEqual(failed["status"], "failed")
        self.assertIn("positive rss_slot0_reads", failed["error"])
        self.assertEqual(failed["telemetry_status"][0]["rss_slot0_reads"], 0)
        self.assertEqual(self.status()["completed"], 0)
        self.mpmo_overrides = {}
        self.assertEqual(self.run_matrix(mpmo_only=True, resume=True, limit=1), 0)
        success = self.events()[-1]
        self.assertEqual(success["key"], failed["key"])
        self.assertEqual(success["attempt"], 2)
        path = Path(success["tables"]["policy_status"][0])
        status = success["telemetry_status"][0]
        path.write_text(json.dumps([{**status, "reader_errors": 1}]))
        self.assertEqual(self.run_matrix(mpmo_only=True, resume=True, limit=0), 0)
        self.assertEqual(self.events()[-1]["status"], "invalidated")
        self.assertEqual(self.events()[-1]["telemetry_status"][0]["reader_errors"], 1)
        self.assertEqual(self.status()["completed"], 0)

    def test_completed_default_plan_is_never_extended_or_rewritten_for_mpmo(self):
        self.assertEqual(self.run_matrix(repeats=1), 0)
        self.assertEqual(self.status()["status"], "complete")
        paths = [
            self.directory / name
            for name in ("plan.json", "results.jsonl", "manifest.json")
        ]
        paths.extend((self.directory / "configs").glob("*.yaml"))
        frozen = {path: path.read_bytes() for path in paths}
        for resume, message in ((False, "experiment exists"), (True, "frozen plan")):
            with self.assertRaisesRegex(ValueError, message):
                self.run_matrix(mpmo_only=True, repeats=1, resume=resume)
            self.assertEqual({path: path.read_bytes() for path in paths}, frozen)
        self.assertEqual(self.popen.call_count, 6)

    def test_mpmo_frozen_manifest_and_collector_sources_remain_strict(self):
        source = self.root / "python/kernmlops/data_collection/mpmo.py"
        source.parent.mkdir(parents=True)
        source.write_text("# collector fixture\n")
        self.assertEqual(self.run_matrix(mpmo_only=True, limit=0), 0)
        frozen = (self.directory / "plan.json").read_bytes()
        for path in (self.mpmo_manifest, source):
            before = path.read_bytes()
            path.write_bytes(before + b"\n")
            with (
                self.subTest(path=path),
                self.assertRaisesRegex(ValueError, "frozen plan"),
            ):
                self.run_matrix(mpmo_only=True, resume=True, limit=0)
            path.write_bytes(before)
            self.assertEqual((self.directory / "plan.json").read_bytes(), frozen)
        other = self.mpmo_manifest.with_name("other.json")
        other.write_bytes(self.mpmo_manifest.read_bytes())
        with self.assertRaisesRegex(ValueError, "frozen plan"):
            self.run_matrix(mpmo_only=True, manifest=other, resume=True, limit=0)
        self.popen.assert_not_called()

    def test_all_frozen_configs_are_rechecked_before_every_launch(self):
        validate = matrix.validate_result

        def tamper_after_exit(row, cell):
            validate(row, cell)
            path = self.directory / "configs/always-all.yaml"
            config = json.loads(path.read_text())
            config["collector_config"]["generic"]["hooks"].append("cbmm")
            path.write_text(json.dumps(config))

        with mock.patch.object(
            matrix, "validate_result", side_effect=tamper_after_exit
        ):
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                self.run_matrix(limit=2)
        self.assertEqual(self.popen.call_count, 1)
        self.assertEqual(self.status()["status"], "blocked")

    def test_failures_stop_and_resume_preserves_attempts_and_skips_valid_successes(
        self,
    ):
        self.exit_codes = [0, 7]
        self.assertEqual(self.run_matrix(), 1)
        self.assertEqual(self.popen.call_count, 2)
        failed = self.events()[-1]
        original = Path(failed["stdout"]).read_bytes()
        self.assertEqual(self.status()["completed"], 1)
        self.assertIn("collection_dir", failed)
        self.assertEqual(self.run_matrix(resume=True, limit=1), 0)
        retried = self.events()[-1]
        self.assertEqual(retried["key"], failed["key"])
        self.assertEqual(retried["attempt"], 2)
        self.assertNotEqual(retried["prefix"], failed["prefix"])
        self.assertEqual(Path(failed["stdout"]).read_bytes(), original)
        self.assertEqual(self.status()["completed"], 2)

    def test_resume_revalidates_success_tables(self):
        self.run_matrix(limit=1)
        success = self.events()[-1]
        Path(success["tables"]["ycsb_metrics"]).unlink()
        self.assertEqual(self.run_matrix(resume=True, limit=1), 0)
        self.assertEqual(self.events()[-1]["key"], success["key"])
        self.assertEqual(self.events()[-1]["attempt"], 2)
        self.assertIn("invalidated", [row["status"] for row in self.events()])
        self.assertEqual(self.status()["completed"], 1)

    def test_unfinished_attempt_gets_a_new_label_on_resume(self):
        self.run_matrix(limit=1)
        first = self.events()[0]
        (self.directory / "results.jsonl").write_text(json.dumps(first) + "\n")
        self.assertEqual(self.run_matrix(resume=True, limit=1), 0)
        self.assertEqual(
            [row["status"] for row in self.events()],
            ["running", "incomplete", "running", "success"],
        )
        self.assertEqual(self.events()[-1]["attempt"], 2)

    def test_lock_blocks_duplicate_experiment_without_launch(self):
        self.run_matrix(limit=0)
        with (self.directory / ".lock").open("a") as lock:
            matrix.fcntl.flock(
                lock.fileno(), matrix.fcntl.LOCK_EX | matrix.fcntl.LOCK_NB
            )
            with self.assertRaises(BlockingIOError):
                self.run_matrix(resume=True, limit=1)
        self.popen.assert_not_called()

    def test_failed_ycsb_phase_is_not_a_success_even_with_zero_collector_exit(self):
        self.bad_phase = True
        self.assertEqual(self.run_matrix(), 1)
        self.assertEqual(self.popen.call_count, 1)
        self.assertEqual(self.status()["completed"], 0)
        self.assertEqual(self.events()[-1]["return_code"], 0)
        self.assertIn("invocation failed", self.events()[-1]["error"])

    def test_phase_positions_metrics_and_hooks_validation(self):
        cell = self.plan()["cells"]["always-none"]
        cell["config"]["benchmark_config"]["redis"].update(repeat=2, outer_repeat=3)
        row, directory = self.fixture(cell, "phase-test", self.root / "stdout.log")
        matrix.validate_result(
            row, cell
        )  # Includes positive operation-level Return=ERROR.
        cases = [
            ("ycsb_invocations", lambda rows: rows.pop(), "phase/repeat"),
            (
                "ycsb_invocations",
                lambda rows: rows[0].update(parse_state="partial"),
                "did not parse",
            ),
            (
                "ycsb_metrics",
                lambda rows: rows[0].update(value=float("nan")),
                "non-finite",
            ),
            ("ycsb_metrics", lambda rows: rows.pop(0), "missing OVERALL"),
            ("system_info", lambda rows: rows[0].update(return_code=1), "failed"),
            ("system_info", lambda rows: rows[0].update(hooks=["cbmm"]), "hook list"),
        ]
        for name, change, message in cases:
            with self.subTest(name=name, message=message):
                path = directory / f"{name}.end.parquet"
                before = path.read_text()
                rows = json.loads(before)
                change(rows)
                path.write_text(json.dumps(rows))
                with self.assertRaisesRegex(ValueError, message):
                    matrix.validate_result(row, cell)
                path.write_text(before)

    def test_reduced_requires_samples_and_a_finite_ratio_in_any_output_chunk(self):
        cell = self.plan()["cells"]["never-reduced"]
        row, directory = self.fixture(cell, "reduced-test", self.root / "stdout.log")
        matrix.validate_result(row, cell)
        (directory / "policy_dtlb.end.parquet").write_text('[{"dtlb_missrate": null}]')
        with self.assertRaisesRegex(ValueError, "no valid DTLB"):
            matrix.validate_result(row, cell)
        (directory / "policy_dtlb.0.parquet").write_text('[{"dtlb_missrate": 0.5}]')
        matrix.validate_result(row, cell)
        (directory / "policy_rss.end.parquet").write_text("[]")
        with self.assertRaisesRegex(ValueError, "no policy_rss samples"):
            matrix.validate_result(row, cell)

    def test_reduced_status_rejects_hard_errors_but_retains_drops(self):
        cell = self.plan()["cells"]["never-reduced"]
        row, directory = self.fixture(cell, "status-test", self.root / "stdout.log")
        path = directory / "policy_status.end.parquet"
        status = json.loads(path.read_text())[0]
        status.update(rss_dropped=3, dtlb_dropped=5, rss_unattributed=7)
        path.write_text(json.dumps([status]))
        matrix.validate_result(row, cell)
        self.assertEqual(row["telemetry_status"], [status])
        self.assertEqual(row["tables"]["policy_status"], [str(path)])
        for counter in ("rss_owner_misses", "dtlb_read_errors"):
            with self.subTest(counter=counter):
                path.write_text(json.dumps([status, {**status, counter: 1}]))
                with self.assertRaisesRegex(
                    ValueError, "attribution or counter-read failures"
                ):
                    matrix.validate_result(row, cell)

    def test_mpmo_requires_final_status_only_and_rejects_even_empty_sample_tables(self):
        cell = self.plan(mpmo_only=True)["cells"]["never-reduced-mpmo"]
        row, directory = self.fixture(cell, "mpmo-test", self.root / "stdout.log")
        matrix.validate_result(row, cell)
        status_path = directory / "policy_status.end.parquet"
        status = status_path.read_text()
        self.assertEqual(row["tables"]["policy_status"], [str(status_path)])
        self.assertEqual(row["telemetry_status"], json.loads(status))
        for name in ("policy_rss", "policy_dtlb"):
            for chunk in ("0", "end"):
                path = directory / f"{name}.{chunk}.parquet"
                path.write_text("[]")
                with (
                    self.subTest(path=path),
                    self.assertRaisesRegex(ValueError, f"unexpected {name}"),
                ):
                    matrix.validate_result(row, cell)
                path.unlink()
        for rows in ([], json.loads(status) * 2):
            status_path.write_text(json.dumps(rows))
            with (
                self.subTest(rows=rows),
                self.assertRaisesRegex(ValueError, "one MPMO"),
            ):
                matrix.validate_result(row, cell)
        status_path.unlink()
        with self.assertRaisesRegex(ValueError, "only final MPMO"):
            matrix.validate_result(row, cell)
        chunk_path = directory / "policy_status.0.parquet"
        chunk_path.write_text(status)
        with self.assertRaisesRegex(ValueError, "only final MPMO"):
            matrix.validate_result(row, cell)
        status_path.write_text(status)
        chunk_path.write_text("[]")
        with self.assertRaisesRegex(ValueError, "only final MPMO"):
            matrix.validate_result(row, cell)

    def test_mpmo_and_control_do_not_bypass_common_system_info_and_ycsb_validation(
        self,
    ):
        for options in ({"mpmo_only": True}, {"selected_control_only": True}):
            cell = next(iter(self.plan(**options)["cells"].values()))
            row, directory = self.fixture(
                cell, cell["mode"] + "-common", self.root / "stdout.log"
            )
            for name, change, message in (
                ("system_info", {"hooks": [matrix.EXTRACTED_HOOK]}, "hook list"),
                (
                    "system_info",
                    {"hooks": ["perf", "mm_rss_stat", "cbmm"]},
                    "hook list",
                ),
                ("system_info", {"return_code": 1}, "failed"),
                ("ycsb_invocations", {"parse_state": "partial"}, "did not parse"),
                ("ycsb_metrics", {"value": float("nan")}, "non-finite"),
            ):
                path = directory / f"{name}.end.parquet"
                before = path.read_text()
                rows = json.loads(before)
                rows[0].update(change)
                path.write_text(json.dumps(rows))
                with (
                    self.subTest(mode=cell["mode"], table=name, change=change),
                    self.assertRaisesRegex(ValueError, message),
                ):
                    matrix.validate_result(row, cell)
                path.write_text(before)

    def test_exact_log_cid_parser_does_not_discover_newest_directory(self):
        prefix, stdout = "experiment.+-always-none-r00-a01", self.root / "stdout.log"
        cid = prefix + "-" + UUID
        stdout.write_text("message " + cid + "\n" + cid + "\n")
        with mock.patch.object(
            Path, "glob", side_effect=AssertionError("no discovery")
        ):
            self.assertEqual(matrix.parse_cid(stdout, prefix), cid)
            for content in (
                "message " + cid + "\n",
                cid.replace(".+", "xx") + "\n",
                cid + " extra\n",
                cid + "\n" + cid + "\n",
            ):
                stdout.write_text(content)
                with self.assertRaises(ValueError):
                    matrix.parse_cid(stdout, prefix)

    def test_timeout_and_interruption_cleanup_record_only_after_child_exit(self):
        for error, result, status in (
            (subprocess.TimeoutExpired("collector", 300), 1, "failed"),
            (KeyboardInterrupt(), 130, "incomplete"),
        ):
            with self.subTest(status=status):
                self.directory = self.root / "data/experiments" / status
                self.wait_error = error
                with (
                    mock.patch.object(matrix.os, "killpg") as kill,
                    mock.patch.object(matrix.time, "sleep"),
                ):
                    self.assertEqual(self.run_matrix(), result)
                self.assertEqual(
                    [call.args[1] for call in kill.call_args_list],
                    [signal.SIGINT, signal.SIGKILL],
                )
                self.assertFalse(self.active)
                self.assertEqual(
                    [row["status"] for row in self.events()], ["running", status]
                )
                self.assertEqual(self.status()["completed"], 0)

    def test_cleanup_failure_does_not_record_exit_while_child_is_running(self):
        self.wait_error = subprocess.TimeoutExpired("collector", 300)
        with mock.patch.object(
            matrix, "stop_child", side_effect=subprocess.TimeoutExpired("cleanup", 5)
        ):
            self.assertEqual(self.run_matrix(), 1)
        self.assertEqual([row["status"] for row in self.events()], ["running"])
        self.assertEqual(self.status()["status"], "running")
        self.assertIn("cleanup failed", self.status()["last_attempt"]["error"])


if __name__ == "__main__":
    unittest.main()
