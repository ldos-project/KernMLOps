"""Collector for compiler-extracted RSS components and paired DTLB counters."""

import ctypes as ct
import hashlib
import json
import math
import struct
from pathlib import Path
from threading import Lock

import polars as pl
from data_schema.generic_table import PolicyDTLBTable, PolicyRSSTable, PolicyStatusTable
from data_schema.schema import CollectionTable

RSS_FIELDS = ("timestamp_ns", "count_pages", "tgid", "member")
DTLB_FIELDS = (
    "timestamp_ns",
    "misses",
    "accesses",
    "miss_enabled_ns",
    "miss_running_ns",
    "access_enabled_ns",
    "access_running_ns",
    "cpu",
)
RSS = struct.Struct("<QQIi")
DTLB = struct.Struct("<7QI4x")
CALLBACK = ct.CFUNCTYPE(ct.c_int, ct.c_void_p, ct.c_int, ct.c_void_p, ct.c_size_t)


def policy_manifest(path: str | Path) -> dict:
    path = Path(path).resolve()
    manifest = json.loads(path.read_text())
    if manifest.get("version") != 1 or manifest.get("sample_frequency") != 1000:
        raise ValueError("unsupported RSS/DTLB policy manifest")
    for name in ("rss", "dtlb", "library"):
        item = manifest["files"][name]
        if Path(item["name"]).name != item["name"]:
            raise ValueError("policy artifact names must be local to the manifest")
        artifact = path.parent / item["name"]
        if hashlib.sha256(artifact.read_bytes()).hexdigest() != item["sha256"]:
            raise ValueError(f"RSS/DTLB artifact hash mismatch: {name}")
    for name, fields, size in (
        ("rss", RSS_FIELDS, RSS.size),
        ("dtlb", DTLB_FIELDS, DTLB.size),
    ):
        schema = manifest["schemas"][name]
        if (
            schema["size"] != size
            or schema["byte_order"] != "little"
            or tuple(field["name"] for field in schema["fields"]) != fields
        ):
            raise ValueError(f"incompatible extracted {name} layout")
        offsets = (0, 8, 16, 20) if name == "rss" else (0, 8, 16, 24, 32, 40, 48, 56)
        widths = (8, 8, 4, 4) if name == "rss" else (8, 8, 8, 8, 8, 8, 8, 4)
        for index, field in enumerate(schema["fields"]):
            if (
                field["offset"] != offsets[index]
                or field["size"] != widths[index]
                or field["signed"] != (name == "rss" and index == 3)
            ):
                raise ValueError(f"incompatible extracted {name} member layout")
    return manifest


def dtlb_rows(samples: list[tuple], previous: dict[int, tuple]) -> list[dict]:
    """Difference exact integer snapshots before scaling, keeping reset/zero spans null."""
    last = previous.copy()
    rows = []
    for sample in sorted(samples, key=lambda value: (value[7], value[0])):
        ts, misses, accesses, me, mr, ae, ar, cpu = sample
        ratio = None
        prior = last.get(cpu)
        if prior is not None:
            delta = [sample[i] - prior[i] for i in range(7)]
            if (
                delta[0] > 0
                and delta[1] >= 0
                and delta[2] > 0
                and all(value > 0 for value in delta[3:])
            ):
                scaled_misses = delta[1] * delta[3] / delta[4]
                scaled_accesses = delta[2] * delta[5] / delta[6]
                value = scaled_misses / scaled_accesses
                if math.isfinite(value):
                    ratio = value
        rows.append(
            dict(zip(DTLB_FIELDS, sample, strict=True)) | {"dtlb_missrate": ratio}
        )
        last[cpu] = sample
    return rows


class ExtractedRSSDTLBHook:
    @classmethod
    def name(cls) -> str:
        return "extracted_rss_dtlb"

    def __init__(self, manifest_path: str):
        self.path = Path(manifest_path).resolve()
        self.manifest = policy_manifest(self.path)
        self._rss: list[tuple] = []
        self._dtlb: list[tuple] = []
        self._previous: dict[int, tuple] = {}
        self._lock = Lock()
        self._native_lock = Lock()
        self._error: str | None = None
        self._errors = [0, 0, 0, 0, 0]
        self._received = [0, 0]
        self._handle = None
        self._closed = False
        self._callback = CALLBACK(self._sample)
        self._lib = ct.CDLL(
            str(self.path.parent / self.manifest["files"]["library"]["name"]),
            use_errno=True,
        )
        self._lib.rss_tlb_open.argtypes = [
            ct.c_char_p,
            ct.c_char_p,
            CALLBACK,
            ct.c_void_p,
            ct.c_void_p,
            ct.c_size_t,
        ]
        self._lib.rss_tlb_open.restype = ct.c_void_p
        self._lib.rss_tlb_poll.argtypes = [ct.c_void_p, ct.c_int]
        self._lib.rss_tlb_poll.restype = ct.c_int
        self._lib.rss_tlb_stop.argtypes = [ct.c_void_p]
        self._lib.rss_tlb_stop.restype = ct.c_int
        self._lib.rss_tlb_errors.argtypes = [ct.c_void_p, ct.POINTER(ct.c_uint64)]
        self._lib.rss_tlb_errors.restype = ct.c_int
        self._lib.rss_tlb_close.argtypes = [ct.c_void_p]
        self._lib.rss_tlb_close.restype = None

    def _sample(self, context, stream, data, size):
        if self._error:
            return 0
        try:
            layout = RSS if stream == 0 else DTLB
            if stream not in (0, 1) or size != layout.size:
                raise ValueError("unexpected extracted event layout")
            values = layout.unpack(ct.string_at(data, size))
            if stream == 0 and values[3] not in range(4):
                raise ValueError("unexpected RSS member")
            with self._lock:
                (self._rss if stream == 0 else self._dtlb).append(values)
                self._received[stream] += 1
            return 0
        except Exception as exc:
            self._error = str(exc)
            return -1

    def load(self, collection_id: str):
        with self._native_lock:
            if self._handle or self._closed:
                raise RuntimeError(
                    "extracted RSS/DTLB collector cannot be loaded twice"
                )
            self.collection_id = collection_id
            error = ct.create_string_buffer(4096)
            files = self.manifest["files"]
            self._handle = self._lib.rss_tlb_open(
                str(self.path.parent / files["rss"]["name"]).encode(),
                str(self.path.parent / files["dtlb"]["name"]).encode(),
                self._callback,
                None,
                error,
                len(error),
            )
            if not self._handle:
                raise RuntimeError(error.value.decode(errors="replace"))

    def poll(self):
        with self._native_lock:
            if not self._handle:
                raise RuntimeError("extracted RSS/DTLB collector is not loaded")
            result = self._lib.rss_tlb_poll(self._handle, 5)
        if self._error or result < 0:
            raise RuntimeError(self._error or f"RSS/DTLB poll failed: {result}")

    def close(self):
        with self._native_lock:
            if not self._handle:
                return
            errors = (ct.c_uint64 * 5)()
            try:
                stopped = self._lib.rss_tlb_stop(self._handle)
                status = self._lib.rss_tlb_errors(self._handle, errors)
                self._errors = list(errors)
                if stopped < 0 or status < 0 or self._error or errors[1] or errors[3]:
                    raise RuntimeError(
                        self._error
                        or f"RSS/DTLB finalization failed: {stopped}/{status}; counters={list(errors)}"
                    )
            finally:
                self._lib.rss_tlb_close(self._handle)
                self._handle = None
                self._closed = True

    def _tables(self, rss, dtlb, previous) -> list[CollectionTable]:
        rss_schema = {
            "ts_ns": pl.UInt64,
            "count": pl.UInt64,
            "tgid": pl.UInt32,
            "member": pl.Int32,
        }
        rss_frame = pl.DataFrame(rss, schema=rss_schema, orient="row")
        dtlb_schema = dict.fromkeys(DTLB_FIELDS[:-1], pl.UInt64) | {
            "cpu": pl.UInt32,
            "dtlb_missrate": pl.Float64,
        }
        dtlb_frame = pl.DataFrame(dtlb_rows(dtlb, previous), schema=dtlb_schema).rename(
            {"timestamp_ns": "ts_ns"}
        )
        tables = [
            PolicyRSSTable.from_df_id(rss_frame, self.collection_id),
            PolicyDTLBTable.from_df_id(dtlb_frame, self.collection_id),
        ]
        if self._closed:
            tables.append(
                PolicyStatusTable.from_df_id(
                    pl.DataFrame(
                        {
                            "rss_dropped": [self._errors[0]],
                            "rss_owner_misses": [self._errors[1]],
                            "dtlb_dropped": [self._errors[2]],
                            "dtlb_read_errors": [self._errors[3]],
                            "rss_unattributed": [self._errors[4]],
                            "rss_received": [self._received[0]],
                            "dtlb_received": [self._received[1]],
                            "rss_schema_bytes": [RSS.size],
                            "dtlb_schema_bytes": [DTLB.size],
                            "counter_scope": [
                                "system-wide per CPU; DTLB READ MISS / READ ACCESS"
                            ],
                        }
                    ),
                    self.collection_id,
                )
            )
        return tables

    def data(self):
        with self._lock:
            rss, dtlb, previous = (
                self._rss.copy(),
                self._dtlb.copy(),
                self._previous.copy(),
            )
        return self._tables(rss, dtlb, previous)

    def pop_data(self):
        with self._lock:
            rss, dtlb, previous = self._rss, self._dtlb, self._previous.copy()
            self._rss, self._dtlb = [], []
            for sample in dtlb:
                if (
                    sample[7] not in self._previous
                    or sample[0] > self._previous[sample[7]][0]
                ):
                    self._previous[sample[7]] = sample
        return self._tables(rss, dtlb, previous)

    def clear(self):
        with self._lock:
            self._rss.clear()
            self._dtlb.clear()
            self._previous.clear()
