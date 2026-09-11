"""Unprivileged original collector tests: fake BCC, real userspace queue atomics.

Run with PYTHONPATH=python/kernmlops python -m unittest discover -s policy/tests
-p test_original_collector.py -v. No test constructs or loads a kernel BPF probe.
"""

import ctypes as ct
import json
import mmap
import os
import subprocess
import tempfile
import unittest
from contextlib import ExitStack
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import bcc
import polars as pl
from data_collection import CollectorConfig, GenericCollectorConfig, bpf
from data_collection.bpf_instrumentation import bpf_hook
from data_collection.bpf_instrumentation import original_rss_tlb as original
from data_schema import table_types


class OriginalCollectorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.native_work = tempfile.TemporaryDirectory(prefix="original-queue-")
        cls.addClassCleanup(cls.native_work.cleanup)
        cls.library = Path(cls.native_work.name) / "liboriginal_queue.so"
        subprocess.run(
            [
                "cc",
                "-shared",
                "-fPIC",
                "-O2",
                "-std=c11",
                "-Wall",
                "-Wextra",
                "-Werror",
                str(original.ROOT / "policy/original_queue.c"),
                "-o",
                str(cls.library),
            ],
            check=True,
            capture_output=True,
        )

    def setUp(self):
        self.work = Path(self.enterContext(tempfile.TemporaryDirectory()))
        self.bundle = self.work / "bundle"
        (self.bundle / "install/lib").mkdir(parents=True)
        self.path = self.bundle / "manifest.json"
        sources = self.work / "sources"
        for name in original.REQUIRED_SOURCES:
            target = sources / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes((original.ROOT / name).read_bytes())
        self.enterContext(patch.object(original, "ROOT", sources))
        self.enterContext(
            patch.object(
                bpf_hook,
                "BPF",
                side_effect=AssertionError("BPF construction forbidden"),
            )
        )
        self.manifest = {
            "version": 1,
            "kind": "original-bcc-llvm",
            "compiler_prefix": str(self.bundle / "install"),
            "sample_frequency": 1000,
            "queue_slots": 32,
            "reader_slot": 0,
            "kernel_source": "/kernel-source",
            "kernel_build": "/kernel-build",
            "files": {},
            "schemas": {},
            "sources": {
                name: original._sha256(sources / name)
                for name in original.REQUIRED_SOURCES
            },
        }
        for transport in ("capture", "mpmo"):
            self.manifest["schemas"][transport] = {}
            for output in original.STREAMS:
                rss = output == "rss_stat_output"
                fields = (
                    [
                        ("ts", 0, 8, False),
                        ("counter_value", 8, 8, False),
                        ("tgid", 16, 4, False),
                        ("member", 20, 4, True),
                    ]
                    if rss
                    else [
                        ("ts_uptime_us", 0, 8, False),
                        ("enabled_time_us", 8, 8, False),
                        ("running_time_us", 16, 8, False),
                        ("count", 24, 4, False),
                    ]
                )
                offset = (8 if rss else 16) if transport == "mpmo" else 0
                size = 24 if rss else 32
                self.manifest["schemas"][transport][output] = {
                    "size": size,
                    "original_size": 32 if rss else 40,
                    "byte_order": "little",
                    "source_event": "rss_stat_output" if rss else "perf_event_data",
                    "program": "tracepoint__kmem__rss_stat" if rss else output + "_on",
                    "map": "mpmo_buff_" + output if transport == "mpmo" else output,
                    "data_offset": offset,
                    "map_value_size": offset + size if transport == "mpmo" else 4,
                    "slots": 32 if transport == "mpmo" else 0,
                    "fields": [
                        dict(zip(("name", "offset", "size", "signed"), field))
                        for field in fields
                    ],
                }
        for module in ("rss", "perf"):
            outputs = {
                name: {
                    "program": schema["program"],
                    "record": schema["source_event"],
                    "fields": [f["name"] for f in schema["fields"]],
                }
                for name, schema in self.manifest["schemas"]["capture"].items()
                if (name == "rss_stat_output") == (module == "rss")
            }
            self.artifact(
                module + "_selection",
                module + ".selection.json",
                json.dumps({"version": 1, "outputs": outputs}).encode(),
            )
        self.artifact("bcc_library", "install/lib/libbcc.so.0", b"test-private-bcc")
        self.artifact(
            "queue_library", "liboriginal_queue.so", self.library.read_bytes()
        )
        self.save_manifest()
        self.probes = {}
        self.events = []
        self.report_mutation = lambda report: None
        self.table_mutation = lambda table: None

    def artifact(self, key, name, content):
        (self.bundle / name).write_bytes(content)
        self.manifest["files"][key] = {
            "name": name,
            "sha256": original._sha256(self.bundle / name),
        }

    def save_manifest(self):
        self.path.write_text(json.dumps(self.manifest))

    def fake_compile(self, text, *, cflags):
        self.assertTrue(original._COMPILE_LOCK.locked())
        self.assertNotIn("KERNMLOPS_BCC_EXPORT_ONLY", os.environ)
        module = Path(os.environ["KERNMLOPS_BCC_REDUCE"]).name.split(".")[0]
        transport = "capture" if "-DKERMIT_CAPTURE" in cflags else "mpmo"
        self.assertEqual(
            "-DKERMIT_CPU_METADATA" in cflags, transport == "mpmo" and module == "perf"
        )
        if transport == "mpmo":
            self.assertIn("-DKERMIT_QUEUE_SIZE=32", cflags)
        self.assertIn("-I/kernel-build/include", cflags)
        schemas = {
            name: deepcopy(schema)
            for name, schema in self.manifest["schemas"][transport].items()
            if (name == "rss_stat_output") == (module == "rss")
        }
        if module == "rss":
            self.assertEqual(text, original.TraceRSSStatBPFHook().bpf_text)
        else:
            self.assertEqual(
                text,
                original.PerfBPFHook(
                    events=[original.DTLBPerfTable, original.DTLBAccessPerfTable]
                ).bpf_text,
            )
        report = {
            "version": 1,
            "transport": transport,
            "export_only": False,
            "schemas": deepcopy(schemas),
        }
        if transport == "capture":
            for schema in report["schemas"].values():
                schema["slots"] = None
        self.report_mutation(report)
        with Path(os.environ["KERNMLOPS_BCC_EVIDENCE"] + ".after.json").open(
            "x"
        ) as output:
            json.dump(report, output)
        tables = {}
        probe = Mock()
        probe.funcs = (
            {
                b"raw_tracepoint__rss_stat": object(),
                b"tracepoint__kmem__rss_stat": object(),
            }
            if module == "rss"
            else {}
        )
        probe.dump_func.return_value = b"\x95" + bytes(7)
        probe.open_perf_events = {}

        def attach(**kwargs):
            probe.funcs[kwargs["fn_name"]] = object()
            if kwargs["cpu"] == -1:
                probe.open_perf_events[(kwargs["ev_type"], kwargs["ev_config"])] = {
                    2: 102,
                    7: 107,
                }

        probe.attach_perf_event.side_effect = attach
        probe.cleanup.side_effect = lambda: self.events.append("cleanup-" + module)
        for name, schema in schemas.items():
            table = SimpleNamespace(open_perf_buffer=Mock())
            if transport == "mpmo":
                fields, cursor = [], 0
                for field in sorted(schema["fields"], key=lambda f: f["offset"]):
                    if cursor < field["offset"]:
                        fields.append(
                            (f"pad{cursor}", ct.c_char * (field["offset"] - cursor))
                        )
                    member = {
                        (8, False): ct.c_uint64,
                        (4, False): ct.c_uint32,
                        (4, True): ct.c_int32,
                    }[(field["size"], field["signed"])]
                    fields.append((field["name"], member))
                    cursor = field["offset"] + field["size"]
                if cursor < schema["size"]:
                    fields.append(("padding", ct.c_char * (schema["size"] - cursor)))
                data = type(
                    "Payload", (ct.Structure,), {"_pack_": 1, "_fields_": fields}
                )
                prefix = [("valid", ct.c_uint64)]
                if schema["data_offset"] == 16:
                    prefix.append(("cpu", ct.c_uint64))
                table.Leaf = type(
                    "Slot",
                    (ct.Structure,),
                    {"_pack_": 1, "_fields_": prefix + [("data", data)]},
                )
                table.Key, table.ttype, table.max_entries, table.flags = (
                    ct.c_uint32,
                    2,
                    32,
                    1024,
                )
                file = self.enterContext((self.work / (name + ".map")).open("w+b"))
                file.truncate(mmap.PAGESIZE)
                table.get_fd = Mock(return_value=file.fileno())
                self.table_mutation(table)
            tables[schema["map"]] = table
        probe.__getitem__ = Mock(side_effect=tables.__getitem__)
        probe.tables = tables
        self.probes[module] = probe
        return probe

    def load(self, transport="capture", cid="test"):
        hook = original.OriginalRSSDTLBHook(self.path, transport)
        self.addCleanup(hook.close)
        with (
            patch.object(hook, "_verify_private_bcc"),
            patch.object(
                original, "compile_bpf", side_effect=self.fake_compile
            ) as compile_mock,
        ):
            hook.load(cid)
        self.assertEqual(compile_mock.call_count, 2)
        return hook

    def emit(self, hook, output, values, cpu=7, padding=0):
        schema = hook.schemas[output]
        payload = bytearray(schema["size"])
        for field in schema["fields"]:
            payload[field["offset"] : field["offset"] + field["size"]] = values[
                field["name"]
            ].to_bytes(field["size"], "little", signed=field["signed"])
        payload.extend(b"\xaa" * padding)
        data = ct.create_string_buffer(bytes(payload))
        probe = self.probes["rss" if output == "rss_stat_output" else "perf"]
        callback = probe.tables[schema["map"]].open_perf_buffer.call_args.args[0]
        callback(cpu, data, len(payload))

    def test_factory_is_passive_opt_in_and_registry_unchanged(self):
        registry = bpf.hook_names()
        with ExitStack() as stack:
            for hook in (*bpf.all_hooks.values(), original.OriginalRSSDTLBHook):
                stack.enter_context(
                    patch.object(
                        hook,
                        "__init__",
                        side_effect=AssertionError("hook instantiated"),
                    )
                )
            stack.enter_context(
                patch.object(
                    original.ct,
                    "CDLL",
                    side_effect=AssertionError("native library loaded"),
                )
            )
            for config in (GenericCollectorConfig(), CollectorConfig().generic):
                self.assertIsNone(config.original_rss_tlb)
                self.assertEqual(config.original_rss_tlb_transport, "capture")
                self.assertEqual(config.get_factory_metadata(), {"hooks": registry})
            for transport in ("capture", "mpmo"):
                config = GenericCollectorConfig(
                    hooks=[],
                    original_rss_tlb=str(self.path),
                    original_rss_tlb_transport=transport,
                )
                self.assertEqual(
                    config.get_factory_metadata(),
                    {
                        "hooks": ["original_rss_dtlb_" + transport],
                        "manifest": self.manifest,
                    },
                )
        for transport in ("capture", "mpmo"):
            config = GenericCollectorConfig(
                hooks=[],
                original_rss_tlb=str(self.path),
                original_rss_tlb_transport=transport,
            )
            (hook,) = config.get_hooks()
            self.assertEqual(hook.name(), "original_rss_dtlb_" + transport)
            self.assertNotIn(hook.name(), registry)
        self.assertEqual(bpf.hook_names(), registry)
        self.assertIn(original.OriginalPerfTable, table_types)

    def test_factory_rejects_mixed_modes_and_missing_manifest(self):
        for config, error in (
            (
                GenericCollectorConfig(original_rss_tlb=str(self.path)),
                "double attachment",
            ),
            (
                GenericCollectorConfig(
                    hooks=[], original_rss_tlb=str(self.path), extracted_rss_tlb="old"
                ),
                "mutually exclusive",
            ),
            (
                GenericCollectorConfig(hooks=[], original_rss_tlb_transport="mpmo"),
                "manifest",
            ),
            (
                GenericCollectorConfig(hooks=[], original_rss_tlb_transport="other"),
                "transport",
            ),
            (GenericCollectorConfig(hooks=["original_rss_dtlb_capture"]), "unknown"),
        ):
            for factory in (config.get_hooks, config.get_factory_metadata):
                with (
                    self.subTest(config=config, factory=factory),
                    self.assertRaisesRegex(ValueError, error),
                ):
                    factory()

    def test_manifest_hashes_paths_selections_and_source_provenance(self):
        self.assertEqual(original.original_manifest(self.path), self.manifest)
        mutations = [
            (("version",), True),
            (("kind",), "extracted"),
            (("sample_frequency",), 999),
            (("queue_slots",), 1),
            (("reader_slot",), 1),
            (("kernel_build",), "/other"),
            (("compiler_prefix",), "install"),
            (("compiler_prefix",), str(self.work)),
            (("sources",), {}),
            (("schemas", "capture", "rss_stat_output", "slots"), None),
        ]
        for name in self.manifest["sources"]:
            mutations.append((("sources", name), "0" * 64))
        for name in self.manifest["files"]:
            mutations.extend(
                [
                    (("files", name, "sha256"), "0" * 64),
                    (("files", name, "name"), "../escape"),
                    (("files", name, "name"), str(self.path)),
                ]
            )
        for keys, value in mutations:
            manifest = deepcopy(self.manifest)
            target = manifest
            for key in keys[:-1]:
                target = target[key]
            target[keys[-1]] = value
            self.path.write_text(json.dumps(manifest))
            with self.subTest(keys=keys, value=value), self.assertRaises(ValueError):
                original.original_manifest(self.path)
        self.save_manifest()
        outside = self.work / "outside"
        outside.write_bytes(b"test-private-bcc")
        (self.bundle / "escape").symlink_to(outside)
        self.manifest["files"]["bcc_library"]["name"] = "escape"
        self.save_manifest()
        with self.assertRaisesRegex(ValueError, "escapes"):
            original.original_manifest(self.path)
        self.manifest["files"]["bcc_library"]["name"] = "install/lib/libbcc.so.0"
        selection = json.loads((self.bundle / "perf.selection.json").read_text())
        selection["outputs"]["dtlb_misses"]["record"] = "replacement_counter"
        self.artifact(
            "perf_selection", "perf.selection.json", json.dumps(selection).encode()
        )
        self.save_manifest()
        with self.assertRaisesRegex(ValueError, "provenance"):
            original.original_manifest(self.path)

    def test_schema_validation_rejects_overlap_widths_and_old_cpu_prefix(self):
        changes = [
            ("size", 40),
            ("data_offset", 8),
            ("map_value_size", 40),
            ("slots", 1),
            ("program", "replacement"),
            ("map", "mpmo_head_dtlb_misses"),
            ("byte_order", "big"),
        ]
        for key, value in changes:
            manifest = deepcopy(self.manifest)
            manifest["schemas"]["mpmo"]["dtlb_misses"][key] = value
            self.path.write_text(json.dumps(manifest))
            with self.subTest(key=key), self.assertRaises(ValueError):
                original.original_manifest(self.path)
        for key, value in (
            ("offset", 0),
            ("offset", 9),
            ("offset", 32),
            ("offset", True),
            ("size", 4),
            ("signed", True),
            ("name", "pid"),
        ):
            manifest = deepcopy(self.manifest)
            manifest["schemas"]["capture"]["dtlb_misses"]["fields"][1][key] = value
            self.path.write_text(json.dumps(manifest))
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                original.original_manifest(self.path)

    def test_capture_uses_report_layout_original_sources_and_raw_counters(self):
        # A different valid LLVM layout must not silently use a hand-coded struct.
        for schemas in self.manifest["schemas"].values():
            fields = schemas["rss_stat_output"]["fields"]
            fields[0]["offset"], fields[1]["offset"] = 8, 0
            fields = schemas["dtlb_misses"]["fields"]
            for field, offset in zip(fields, (8, 16, 24, 0)):
                field["offset"] = offset
        self.save_manifest()
        hook = self.load()
        rss = {
            "ts": (1 << 63) + 7,
            "counter_value": (1 << 64) - 2,
            "tgid": 42,
            "member": -3,
        }
        perf = {
            "ts_uptime_us": 100,
            "count": 0xFEDCBA98,
            "enabled_time_us": (1 << 63) + 8,
            "running_time_us": 17,
        }
        self.emit(hook, "rss_stat_output", rss)
        self.emit(hook, "dtlb_misses", perf, cpu=2)
        self.emit(
            hook, "dtlb_accesses", perf | {"ts_uptime_us": 107, "count": 123}, cpu=7
        )
        self.assertEqual(
            [t.table.to_dicts() for t in hook.data()],
            [t.table.to_dicts() for t in hook.pop_data()],
        )
        self.assertEqual(hook.pop_data(), [])
        self.emit(hook, "rss_stat_output", rss)
        self.emit(hook, "dtlb_misses", perf, cpu=2)
        self.emit(
            hook, "dtlb_accesses", perf | {"ts_uptime_us": 107, "count": 123}, cpu=7
        )
        tables = {table.name(): table.table for table in hook.pop_data()}
        self.assertEqual(
            tables["policy_rss"].to_dicts(),
            [
                {
                    "ts_ns": rss["ts"],
                    "count": rss["counter_value"],
                    "tgid": 42,
                    "member": -3,
                    "collection_id": "test",
                }
            ],
        )
        frame = tables["original_perf"]
        self.assertEqual(frame["counter"].to_list(), ["dtlb_misses", "dtlb_accesses"])
        self.assertEqual(frame["count"].to_list(), [0xFEDCBA98, 123])
        self.assertEqual(frame["cpu"].to_list(), [2, 7])
        self.assertEqual(frame["ts_uptime_us"].to_list(), [100, 107])
        self.assertNotIn("dtlb_missrate", frame.columns)
        self.assertEqual(frame.schema["count"], pl.UInt32)
        for output in original.STREAMS:
            module = "rss" if output == "rss_stat_output" else "perf"
            call = self.probes[module].tables[output].open_perf_buffer.call_args
            self.assertEqual(call.kwargs["page_cnt"], 256 if module == "rss" else 64)
            call.kwargs["lost_cb"](7, 9)
        hook.poll()
        for probe in self.probes.values():
            probe.perf_buffer_poll.assert_called_once_with(
                timeout=original.POLL_TIMEOUT_MS
            )
        hook.close()
        row = hook.pop_data()[0].table.to_dicts()[0]
        original.validate_original_status([row], "test", "capture")
        self.assertEqual(
            [row[f"lost_{source}"] for source in original.STREAMS.values()], [9, 9, 9]
        )
        self.assertFalse(
            {"rss_owner_misses", "dtlb_read_errors", "rss_unattributed"} & row.keys()
        )
        self.assertEqual(hook.pop_data(), [])

    def test_capture_accepts_only_perf_raw_alignment_padding(self):
        hook = self.load()
        for output, source in original.STREAMS.items():
            values = {field["name"]: 19 for field in hook.schemas[output]["fields"]}
            self.emit(hook, output, values, padding=4)
            self.assertEqual(hook._stats[f"{source}_received"], 1)
        hook.poll()
        self.assertEqual(hook._stats["callback_errors"], 0)
        tables = {table.name(): table.table for table in hook.pop_data()}
        self.assertEqual(tables["policy_rss"]["count"].to_list(), [19])
        self.assertEqual(tables["original_perf"]["count"].to_list(), [19, 19])
        self.emit(
            hook,
            "rss_stat_output",
            {field["name"]: 19 for field in hook.schemas["rss_stat_output"]["fields"]},
            padding=12,
        )
        with self.assertRaisesRegex(RuntimeError, "callback size"):
            hook.poll()

    def test_original_group_attachment_and_loaded_bytecode_evidence(self):
        frozen = {
            path: original._sha256(path)
            for path in self.bundle.rglob("*")
            if path.is_file()
        }
        hook = self.load()
        calls = self.probes["perf"].attach_perf_event.call_args_list
        self.assertEqual(len(calls), 3)
        self.assertEqual(
            [(call.kwargs["cpu"], call.kwargs["group_fd"]) for call in calls],
            [(-1, -1), (2, 102), (7, 107)],
        )
        self.assertEqual([call.kwargs["sample_freq"] for call in calls], [1000] * 3)
        self.assertEqual(
            calls[0].kwargs["ev_config"], original.DTLBPerfTable.ev_config()
        )
        self.assertEqual(
            calls[1].kwargs["ev_config"], original.DTLBAccessPerfTable.ev_config()
        )
        self.assertEqual(
            set(hook._compiled_programs),
            {
                "raw_tracepoint__rss_stat",
                "tracepoint__kmem__rss_stat",
                "dtlb_misses_on",
                "dtlb_accesses_on",
            },
        )
        for record in hook._compiled_programs.values():
            self.assertEqual(
                original._sha256(hook.evidence / record["name"]), record["sha256"]
            )
            self.assertEqual(record["size"], 8)
        self.assertEqual(frozen, {path: original._sha256(path) for path in frozen})
        second = original.OriginalRSSDTLBHook(self.path)
        with (
            patch.object(second, "_verify_private_bcc"),
            patch.object(original, "compile_bpf") as compiler,
            self.assertRaises(FileExistsError),
        ):
            second.load("test")
        compiler.assert_not_called()
        self.assertEqual(frozen, {path: original._sha256(path) for path in frozen})

    def test_environment_restored_on_success_and_compile_failure(self):
        values = {
            "KERNMLOPS_BCC_REDUCE": "/old/selection",
            "KERNMLOPS_BCC_EVIDENCE": "/old/evidence",
            "KERNMLOPS_BCC_EXPORT_ONLY": "1",
        }
        with patch.dict(os.environ, values):
            self.load(cid="success")
            self.assertEqual({key: os.environ[key] for key in values}, values)
            hook = original.OriginalRSSDTLBHook(self.path)

            def compile_or_fail(text, **kwargs):
                if os.environ["KERNMLOPS_BCC_REDUCE"].endswith("perf.selection.json"):
                    raise RuntimeError("compiler failed")
                return self.fake_compile(text, **kwargs)

            with (
                patch.object(hook, "_verify_private_bcc"),
                patch.object(original, "compile_bpf", side_effect=compile_or_fail),
                self.assertRaisesRegex(RuntimeError, "compiler failed"),
            ):
                hook.load("failed")
            self.assertEqual({key: os.environ[key] for key in values}, values)
            self.assertTrue(hook._closed)
            self.probes["rss"].cleanup.assert_called_once()
            self.assertFalse(hook._probes)

    def test_private_bcc_must_be_selected_before_import(self):
        hook = original.OriginalRSSDTLBHook(self.path)
        with (
            patch.object(original, "compile_bpf") as compiler,
            self.assertRaisesRegex(ValueError, "private Python BCC"),
        ):
            hook.load("wrong-bcc")
        compiler.assert_not_called()
        self.assertFalse((self.bundle / "evidence").exists())
        with (
            patch.object(bcc, "__file__", str(self.bundle / "install/bcc/__init__.py")),
            self.assertRaisesRegex(ValueError, "loaded libbcc"),
        ):
            hook._verify_private_bcc()

    def test_filesystem_owner_restored_without_dropping_bpf_privileges(self):
        for environment, expected in (
            ({"SUDO_UID": "1001", "SUDO_GID": "1002"}, (1001, 1002)),
            ({"UNAME": "container-user"}, (1003, 1004)),
            ({}, (self.bundle.stat().st_uid, self.bundle.stat().st_gid)),
        ):
            ids = {"uid": 0, "gid": 0}

            def select(kind, value, ids=ids):
                previous = ids[kind]
                if value != -1:
                    ids[kind] = value
                return previous

            libc = Mock()
            libc.setfsuid.side_effect = lambda value: select("uid", value)
            libc.setfsgid.side_effect = lambda value: select("gid", value)
            libc.capget.return_value = libc.capset.return_value = 0
            with (
                self.subTest(environment=environment),
                patch.dict(os.environ, environment, clear=True),
                patch.object(original.os, "geteuid", return_value=0),
                patch.object(original.os, "seteuid") as seteuid,
                patch.object(original, "get_user_group_ids", return_value=(1003, 1004)),
                patch.object(original.ct, "CDLL", return_value=libc),
                self.assertRaisesRegex(RuntimeError, "evidence failure"),
                original._evidence_owner(self.bundle) as owner,
            ):
                self.assertEqual(owner, expected)
                self.assertEqual((ids["uid"], ids["gid"]), expected)
                raise RuntimeError("evidence failure")
            self.assertEqual(ids, {"uid": 0, "gid": 0})
            seteuid.assert_not_called()
            self.assertEqual(libc.capset.call_count, 2)

    def test_runtime_schema_and_map_descriptor_mismatches_fail_closed(self):
        self.report_mutation = lambda report: next(iter(report["schemas"].values()))[
            "fields"
        ][0].update(offset=8)
        hook = original.OriginalRSSDTLBHook(self.path)
        with (
            patch.object(hook, "_verify_private_bcc"),
            patch.object(original, "compile_bpf", side_effect=self.fake_compile),
            self.assertRaises(ValueError),
        ):
            hook.load("bad-report")
        self.probes["rss"].cleanup.assert_called_once()
        self.report_mutation = lambda report: None
        self.table_mutation = lambda table: setattr(table, "max_entries", 1)
        hook = original.OriginalRSSDTLBHook(self.path, "mpmo")
        with (
            patch.object(hook, "_verify_private_bcc"),
            patch.object(original, "compile_bpf", side_effect=self.fake_compile),
            self.assertRaisesRegex(ValueError, "queue descriptor"),
        ):
            hook.load("bad-map")
        self.assertFalse(hook._queues)
        self.assertIsNone(hook._queue_read)
        for probe in self.probes.values():
            probe.cleanup.assert_called_once()

    def test_atomic_reader_only_touches_slot_zero(self):
        lib = ct.CDLL(str(self.library))
        read = lib.original_queue_read
        read.argtypes = [
            ct.c_void_p,
            ct.c_size_t,
            ct.c_size_t,
            ct.c_void_p,
            ct.POINTER(ct.c_uint64),
        ]
        read.restype = ct.c_int
        for offset, size in ((8, 24), (16, 32)):
            width = offset + size
            memory = (ct.c_uint64 * (width * 32 // 8))()
            address = ct.addressof(memory)
            for index in range(len(memory)):
                memory[index] = index + 19
            payload = ct.create_string_buffer(size)
            cpu = ct.c_uint64(987)
            for state, result in ((0, 0), (2, 2), (3, 2), (4, -71), (1, 1)):
                memory[0] = state
                before = bytes(memory)
                self.assertEqual(
                    read(address, offset, size, payload, ct.byref(cpu)), result
                )
                self.assertEqual(bytes(memory)[8:], before[8:])
                self.assertEqual(memory[0], 0 if state == 1 else state)
                if state == 1:
                    self.assertEqual(payload.raw, before[offset : offset + size])
                    self.assertEqual(cpu.value, memory[1] if offset == 16 else 987)
            for args in (
                (None, offset, size, payload, ct.byref(cpu)),
                (address + 1, offset, size, payload, ct.byref(cpu)),
                (address, 0, size, payload, ct.byref(cpu)),
                (address, offset, 0, payload, ct.byref(cpu)),
                (address, offset, size, None, ct.byref(cpu)),
                (address, offset, size, payload, None),
            ):
                self.assertLess(read(*args), 0)

    def test_mpmo_poll_has_no_frames_stream_buffers_or_all_slot_drain(self):
        hook = self.load("mpmo")
        mappings = [queue[0] for queue in hook._queues.values()]
        for turn in range(3):
            for index, (mapping, address, offset, size, snapshot, cpu) in enumerate(
                hook._queues.values()
            ):
                mapping[:] = b"\xab" * len(mapping)
                ct.c_uint64.from_address(address).value = (turn + index) % 3
                ct.c_uint64.from_address(address + 8).value = 7
            before = [mapping[8:] for mapping in mappings]
            with patch.object(
                original.pl,
                "DataFrame",
                side_effect=AssertionError("MPMO constructed a DataFrame"),
            ):
                hook.poll()
                self.assertEqual(hook.pop_data(), [])
                self.assertEqual(hook.data(), [])
            self.assertEqual([mapping[8:] for mapping in mappings], before)
            self.assertEqual(hook._rss, [])
            self.assertEqual(hook._perf, [])
        for probe in self.probes.values():
            probe.perf_buffer_poll.assert_not_called()
            for table in probe.tables.values():
                table.open_perf_buffer.assert_not_called()
        stats = hook._stats.copy()
        hook.close()
        self.assertEqual(hook._stats, stats)
        self.assertTrue(all(mapping.closed for mapping in mappings))
        self.assertFalse(hook._queues)
        self.assertIsNone(hook._queue_read)
        tables = hook.pop_data()
        self.assertEqual([table.name() for table in tables], ["policy_status"])
        row = tables[0].table.to_dicts()[0]
        original.validate_original_status([row], "test", "mpmo")
        self.assertEqual(row["polls"], 3)
        for source in original.STREAMS.values():
            self.assertEqual(
                [row[f"{source}_{kind}"] for kind in ("received", "empty", "busy")],
                [1, 1, 1],
            )
        self.assertEqual(hook.pop_data(), [])
        hook.close()
        for probe in self.probes.values():
            probe.cleanup.assert_called_once()

    def test_close_failure_still_releases_other_producers_mappings_and_results(self):
        for transport in ("capture", "mpmo"):
            hook = self.load(transport, cid="close-" + transport)
            mappings = [queue[0] for queue in hook._queues.values()]
            if transport == "capture":
                self.emit(
                    hook,
                    "rss_stat_output",
                    {"ts": 1, "counter_value": 2, "tgid": 3, "member": 0},
                )
            self.probes["rss"].cleanup.side_effect = RuntimeError("cleanup failed")
            with self.assertRaisesRegex(RuntimeError, "cleanup failed"):
                hook.close()
            self.probes["rss"].close.assert_called_once()
            for probe in self.probes.values():
                probe.cleanup.assert_called_once()
            self.assertTrue(all(mapping.closed for mapping in mappings))
            self.assertFalse(hook._probes or hook._queues)
            self.assertIsNone(hook._queue_read)
            tables = {table.name(): table.table for table in hook.pop_data()}
            self.assertEqual(
                tables["policy_status"]["collector_error"].item(), "cleanup failed"
            )
            if transport == "capture":
                self.assertEqual(tables["policy_rss"].height, 1)
            self.assertEqual(hook.pop_data(), [])

    def test_callback_errors_and_bad_queue_states_remain_visible(self):
        hook = self.load()
        callback = (
            self.probes["rss"]
            .tables["rss_stat_output"]
            .open_perf_buffer.call_args.args[0]
        )
        callback(0, None, 24)
        with self.assertRaisesRegex(RuntimeError, "callback payload"):
            hook.poll()
        hook.close()
        self.assertEqual(hook.pop_data()[0].table["callback_errors"].item(), 1)
        hook = self.load("mpmo", cid="bad-state")
        for _, address, *_ in hook._queues.values():
            ct.c_uint64.from_address(address).value = 4
        with self.assertRaisesRegex(RuntimeError, "queue state"):
            hook.poll()
        hook.close()
        row = hook.pop_data()[0].table.to_dicts()[0]
        self.assertEqual(row["callback_errors"], 3)
        with self.assertRaises(ValueError):
            original.validate_original_status([row], "bad-state", "mpmo")

    def test_status_validator_fixed_fields_uint64_and_poll_consistency(self):
        for transport in ("capture", "mpmo"):
            row = {
                "collection_id": "cid",
                "origin": "original-bcc-llvm",
                "transport": transport,
                "counter_scope": original.COUNTER_SCOPE,
                "rss_schema_bytes": 24,
                "dtlb_schema_bytes": 32,
                "callback_errors": 0,
                "compiler_evidence": "/bundle/evidence/collections/cid",
            }
            row.update(
                {f"{source}_received": 1 for source in original.STREAMS.values()}
            )
            if transport == "mpmo":
                row.update(data_slots=32, read_slot=0, polls=3)
                row.update(
                    {
                        f"{source}_{kind}": 1
                        for source in original.STREAMS.values()
                        for kind in ("empty", "busy")
                    }
                )
            else:
                row.update(
                    {f"lost_{source}": 99 for source in original.STREAMS.values()}
                )
            original.validate_original_status([row], "cid", transport)
            for rows in ([], [row, row], [None]):
                with self.assertRaises(ValueError):
                    original.validate_original_status(rows, "cid", transport)
            for key, current in row.items():
                changes = (
                    (-1, 1 << 64, True, None)
                    if type(current) is int
                    else ("wrong", None)
                )
                if key.endswith("received") or key == "polls":
                    changes += (0,)
                if key == "callback_errors":
                    changes += (1,)
                if key == "compiler_evidence":
                    changes = ("", None)
                for value in changes:
                    with (
                        self.subTest(transport=transport, key=key, value=value),
                        self.assertRaises(ValueError),
                    ):
                        original.validate_original_status(
                            [row | {key: value}], "cid", transport
                        )
            if transport == "mpmo":
                with self.assertRaisesRegex(ValueError, "inconsistent"):
                    original.validate_original_status(
                        [row | {"dtlb_misses_empty": 2}], "cid", transport
                    )


if __name__ == "__main__":
    unittest.main()
