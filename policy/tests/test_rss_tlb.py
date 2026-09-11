"""RSS/DTLB contract tests using temporary artifacts, without loading kernel BPF."""

import ctypes as ct
import hashlib
import json
import struct
import tempfile
import unittest
from contextlib import ExitStack
from copy import deepcopy
from pathlib import Path
from unittest.mock import Mock, patch

from data_collection import CollectorConfig, GenericCollectorConfig, bpf
from data_collection.bpf_instrumentation import rss_tlb_policy as policy

BASE = 1 << 60  # Small deltas disappear if snapshots are converted to floats first.
FIRST = (10, BASE, BASE, BASE, BASE, BASE, BASE, 2)
SECOND = (20, BASE + 3, BASE + 20, BASE + 20, BASE + 5, BASE + 40, BASE + 20, 2)


class RSSDTLBTests(unittest.TestCase):
    def setUp(self):
        self.path = (
            Path(self.enterContext(tempfile.TemporaryDirectory())) / "manifest.json"
        )
        self.manifest = {
            "version": 1,
            "sample_frequency": 1000,
            "files": {},
            "schemas": {},
        }
        for name, filename in (
            ("rss", "rss.bpf.o"),
            ("dtlb", "dtlb.bpf.o"),
            ("library", "librss_tlb.so"),
        ):
            payload = name.encode()
            (self.path.parent / filename).write_bytes(payload)
            self.manifest["files"][name] = {
                "name": filename,
                "sha256": hashlib.sha256(payload).hexdigest(),
            }
        for name, fields, widths, size, original in (
            ("rss", "timestamp_ns count_pages tgid member", (8, 8, 4, 4), 24, 32),
            (
                "dtlb",
                "timestamp_ns misses accesses miss_enabled_ns miss_running_ns "
                "access_enabled_ns access_running_ns cpu",
                (8,) * 7 + (4,),
                64,
                72,
            ),
        ):
            self.manifest["schemas"][name] = {
                "size": size,
                "original_size": original,
                "byte_order": "little",
                "fields": [
                    dict(
                        name=field,
                        offset=sum(widths[:i]),
                        size=widths[i],
                        signed=field == "member",
                    )
                    for i, field in enumerate(fields.split())
                ],
            }
        self.path.write_text(json.dumps(self.manifest))
        self.lib = Mock()
        for name, result in (("open", 1), ("poll", 0), ("stop", 0), ("errors", 0)):
            getattr(self.lib, f"rss_tlb_{name}").return_value = result
        with patch.object(policy.ct, "CDLL", return_value=self.lib):
            self.hook = policy.ExtractedRSSDTLBHook(str(self.path))
        self.hook.load("test")

    def emit(self, stream, values):
        payload = struct.pack(("<QQIi", "<7QI4x")[stream], *values)
        buffer = ct.create_string_buffer(payload)
        self.assertEqual(self.hook._callback(None, stream, buffer, len(payload)), 0)

    def test_exact_integer_deltas_scaling_and_cpu_separation(self):
        other = (10, 90, 200, 100, 50, 100, 50, 3)
        other_next = (20, 95, 250, 120, 60, 110, 60, 3)
        previous = {}
        rows = policy.dtlb_rows([SECOND, other_next, FIRST, other], previous)
        self.assertEqual([row["cpu"] for row in rows], [2, 2, 3, 3])
        self.assertEqual([row["dtlb_missrate"] for row in rows], [None, 0.3, None, 0.2])
        self.assertEqual(previous, {})

    def test_zero_reset_and_unscheduled_intervals(self):
        for index in range(7):
            for delta in (-1, 0):
                sample = list(SECOND)
                sample[index] = FIRST[index] + delta
                with self.subTest(field=index, delta=delta):
                    row = policy.dtlb_rows([tuple(sample)], {2: FIRST})[0]
                    self.assertEqual(
                        row["dtlb_missrate"], 0.0 if (index, delta) == (1, 0) else None
                    )

    def test_manifest_hashes_and_layout_validation(self):
        self.assertEqual(policy.policy_manifest(self.path), self.manifest)
        changes = [(("version",), 2), (("sample_frequency",), 999)]
        for name, item in self.manifest["files"].items():
            changes.append((("files", name, "sha256"), "0" * 64))
            for filename in (
                f"../{self.path.parent.name}/{item['name']}",
                str(self.path.parent / item["name"]),
            ):
                changes.append((("files", name, "name"), filename))
        for name, schema in self.manifest["schemas"].items():
            for key, value in (
                ("size", schema["original_size"]),
                ("byte_order", "big"),
                ("fields", schema["fields"][::-1]),
            ):
                changes.append((("schemas", name, key), value))
            for i, field in enumerate(schema["fields"]):
                for key, value in (
                    ("name", "wrong"),
                    ("offset", field["offset"] + 1),
                    ("size", field["size"] + 1),
                    ("signed", not field["signed"]),
                ):
                    changes.append((("schemas", name, "fields", i, key), value))
        for keys, value in changes:
            with self.subTest(path=keys, value=value):
                manifest = deepcopy(self.manifest)
                target = manifest
                for key in keys[:-1]:
                    target = target[key]
                target[keys[-1]] = value
                self.path.write_text(json.dumps(manifest))
                with self.assertRaises(ValueError):
                    policy.policy_manifest(self.path)

    def test_callback_tables_preserve_counters_and_carry_across_pops(self):
        self.emit(0, (BASE, BASE + 9, 42, 3))
        self.emit(1, SECOND)
        self.emit(1, FIRST)  # Arrival order must not replace the newest baseline.
        preview = self.hook.data()
        tables = self.hook.pop_data()
        self.assertEqual(
            [(type(t), t.name()) for t in tables],
            [
                (policy.PolicyRSSTable, "policy_rss"),
                (policy.PolicyDTLBTable, "policy_dtlb"),
            ],
        )
        self.assertEqual(
            [t.table.to_dicts() for t in preview], [t.table.to_dicts() for t in tables]
        )
        rss, dtlb = (t.table for t in tables)
        self.assertEqual(
            rss.to_dicts(),
            [
                {
                    "ts_ns": BASE,
                    "count": BASE + 9,
                    "tgid": 42,
                    "member": 3,
                    "collection_id": "test",
                }
            ],
        )
        self.assertEqual(dtlb["misses"].to_list(), [BASE, BASE + 3])
        self.assertEqual(dtlb["accesses"].to_list(), [BASE, BASE + 20])
        self.assertFalse({"pid", "tgid"} & set(dtlb.columns))
        self.assertEqual(dtlb["dtlb_missrate"].to_list(), [None, 0.3])
        self.emit(
            1, (30, BASE + 4, BASE + 30, BASE + 40, BASE + 15, BASE + 50, BASE + 30, 2)
        )
        self.assertEqual(
            self.hook.pop_data()[1].table["dtlb_missrate"].to_list(), [0.2]
        )
        self.assertTrue(all(t.table.is_empty() for t in self.hook.pop_data()))

    def test_close_errors_still_free_and_record_status(self):
        self.emit(0, (10, 20, 42, 0))
        self.emit(1, FIRST)
        for errors in ([7, 0, 9, 0, 11], [7, 2, 9, 0, 11], [7, 0, 9, 3, 11]):
            with self.subTest(errors=errors):
                self.hook._handle = 1
                self.lib.rss_tlb_close.reset_mock()

                def read_errors(handle, output):
                    output[:] = errors
                    return 0

                self.lib.rss_tlb_errors.side_effect = read_errors
                if errors[1] or errors[3]:
                    with self.assertRaises(RuntimeError):
                        self.hook.close()
                else:
                    self.hook.close()
                self.assertIsNone(self.hook._handle)
                self.hook.close()
                self.lib.rss_tlb_close.assert_called_once_with(1)
                status = self.hook.pop_data()[2]
                self.assertIsInstance(status, policy.PolicyStatusTable)
                self.assertEqual(status.name(), "policy_status")
                self.assertEqual(
                    status.table.select(
                        "rss_dropped",
                        "rss_owner_misses",
                        "dtlb_dropped",
                        "dtlb_read_errors",
                        "rss_unattributed",
                    ).row(0),
                    tuple(errors),
                )
                self.assertEqual(
                    status.table.select(
                        "rss_received",
                        "dtlb_received",
                        "rss_schema_bytes",
                        "dtlb_schema_bytes",
                    ).row(0),
                    (1, 1, 24, 64),
                )

    def test_native_calls_share_lock_without_blocking_callbacks(self):
        def native(*args):
            self.assertTrue(self.hook._native_lock.locked())
            self.assertFalse(self.hook._lock.locked())
            self.emit(1, FIRST)
            return 0

        for name in ("poll", "stop", "errors", "close"):
            getattr(self.lib, f"rss_tlb_{name}").side_effect = native
        self.hook.poll()
        self.hook.close()

    def test_factory_validation_and_metadata_do_not_instantiate_probes(self):
        with ExitStack() as stack:
            for hook in (*bpf.all_hooks.values(), policy.ExtractedRSSDTLBHook):
                stack.enter_context(
                    patch.object(
                        hook,
                        "__init__",
                        side_effect=AssertionError("probe instantiated"),
                    )
                )
            for config in (GenericCollectorConfig(), CollectorConfig().generic):
                self.assertEqual(
                    config.get_factory_metadata(), {"hooks": bpf.hook_names()}
                )
                self.assertIn(bpf.CBMMBPFHook.name(), config.hooks)
            config = GenericCollectorConfig(hooks=[], extracted_rss_tlb=str(self.path))
            self.assertEqual(
                config.get_factory_metadata(),
                {"hooks": ["extracted_rss_dtlb"], "manifest": self.manifest},
            )
            for config, message in (
                (GenericCollectorConfig(hooks=["not_a_hook"]), "unknown"),
                (
                    GenericCollectorConfig(extracted_rss_tlb=str(self.path)),
                    "double attachment",
                ),
            ):
                for factory in (config.get_hooks, config.get_factory_metadata):
                    with (
                        self.subTest(error=message, factory=factory.__name__),
                        self.assertRaisesRegex(ValueError, message),
                    ):
                        factory()


if __name__ == "__main__":
    unittest.main()
