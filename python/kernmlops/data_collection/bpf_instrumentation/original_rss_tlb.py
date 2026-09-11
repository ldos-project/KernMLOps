"""Opt-in runtime compaction of the original RSS and DTLB BCC sources.

The runner must select the private Python BCC and libbcc before importing the
collector. MPMO deliberately consumes only slot zero and retains no event stream.
"""

import ctypes as ct
import hashlib
import json
import mmap
import os
import re
from contextlib import contextmanager
from pathlib import Path
from threading import Lock

import polars as pl
from data_collection.bpf_instrumentation.bpf_hook import POLL_TIMEOUT_MS, compile_bpf
from data_collection.bpf_instrumentation.mm_rss_stat import TraceRSSStatBPFHook
from data_collection.bpf_instrumentation.perf.perf_hook import PerfBPFHook
from data_schema import get_user_group_ids
from data_schema.generic_table import (
    OriginalPerfTable,
    PolicyRSSTable,
    PolicyStatusTable,
)
from data_schema.perf.tlb_perf import DTLBAccessPerfTable, DTLBPerfTable

ROOT = Path(__file__).resolve().parents[4]
SOURCE_DIR = "python/kernmlops/data_collection/bpf_instrumentation/"
REQUIRED_SOURCES = {
    SOURCE_DIR + "bpf/mm_trace_rss_stat.bpf.c",
    SOURCE_DIR + "bpf/perf.bpf.c",
    SOURCE_DIR + "bpf/kermit.h",
    SOURCE_DIR + "perf/perf_hook.py",
}
STREAMS = {
    "rss_stat_output": "rss",
    "dtlb_misses": "dtlb_misses",
    "dtlb_accesses": "dtlb_accesses",
}
COUNTER_SCOPE = (
    "system-wide per CPU; independently sampled cumulative DTLB READ MISS + READ ACCESS; "
    "original u32 counts; no inference, aggregation, or model"
)
_COMPILE_LOCK = Lock()


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _uint64(value):
    return type(value) is int and 0 <= value < 1 << 64


def _sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _local_path(root, name):
    _require(isinstance(name, str) and bool(name), "missing artifact/source path")
    relative = Path(name)
    _require(
        not relative.is_absolute() and ".." not in relative.parts,
        "path must be relative and contained",
    )
    path = (root / relative).resolve()
    _require(
        path.is_relative_to(root) and path != root,
        "path escapes its bundle/source root",
    )
    return path


def _validate_schema(schema, output, transport):
    rss = output == "rss_stat_output"
    fields = (
        {
            "ts": (8, False),
            "counter_value": (8, False),
            "tgid": (4, False),
            "member": (4, True),
        }
        if rss
        else {
            "ts_uptime_us": (8, False),
            "count": (4, False),
            "enabled_time_us": (8, False),
            "running_time_us": (8, False),
        }
    )
    size = 24 if rss else 32
    offset = (8 if rss else 16) if transport == "mpmo" else 0
    expected = {
        "size": size,
        "byte_order": "little",
        "source_event": "rss_stat_output" if rss else "perf_event_data",
        "program": "tracepoint__kmem__rss_stat" if rss else output + "_on",
        "map": "mpmo_buff_" + output if transport == "mpmo" else output,
        "data_offset": offset,
        "map_value_size": offset + size if transport == "mpmo" else 4,
        "slots": 32 if transport == "mpmo" else 0,
    }
    for key, value in expected.items():
        _require(
            type(schema.get(key)) is type(value) and schema[key] == value,
            f"incompatible {output}/{transport} {key}",
        )
    if "original_size" in schema:
        _require(
            type(schema["original_size"]) is int
            and schema["original_size"] == (32 if rss else 40),
            "incompatible original record size",
        )
    members = schema["fields"]
    _require(
        len(members) == len(fields) and {f["name"] for f in members} == set(fields),
        f"incompatible {output} fields",
    )
    occupied = set()
    for field in members:
        width, signed = fields[field["name"]]
        start = field["offset"]
        _require(
            _uint64(start) and start % width == 0 and start + width <= size,
            "invalid field offset",
        )
        _require(
            type(field["size"]) is int
            and field["size"] == width
            and field["signed"] is signed,
            "invalid field type",
        )
        span = set(range(start, start + width))
        _require(not occupied & span, "overlapping schema fields")
        occupied.update(span)


def original_manifest(path: str | Path) -> dict:
    """Validate frozen artifacts and original-source provenance without loading BPF."""
    path = Path(path).resolve()
    try:
        manifest = json.loads(path.read_text())
        for key, value in {
            "version": 1,
            "kind": "original-bcc-llvm",
            "sample_frequency": 1000,
            "queue_slots": 32,
            "reader_slot": 0,
            "kernel_source": "/kernel-source",
            "kernel_build": "/kernel-build",
        }.items():
            _require(
                type(manifest.get(key)) is type(value) and manifest[key] == value,
                f"unsupported original manifest {key}",
            )
        prefix = Path(manifest["compiler_prefix"])
        _require(
            prefix.is_absolute() and prefix.resolve() == path.parent / "install",
            "compiler_prefix must be the bundle's absolute install path",
        )
        files = manifest["files"]
        _require(
            {"bcc_library", "queue_library", "rss_selection", "perf_selection"}
            <= files.keys(),
            "missing original artifacts",
        )
        artifacts = {}
        for name, item in files.items():
            artifact = _local_path(path.parent, item["name"])
            _require(
                _sha256(artifact) == item["sha256"],
                f"original artifact hash mismatch: {name}",
            )
            artifacts[name] = artifact
        _require(
            artifacts["bcc_library"].is_relative_to(prefix.resolve() / "lib"),
            "BCC library is outside compiler_prefix/lib",
        )
        sources = manifest["sources"]
        _require(
            REQUIRED_SOURCES <= sources.keys(), "missing original-source provenance"
        )
        for name, digest in sources.items():
            _require(
                _sha256(_local_path(ROOT, name)) == digest,
                f"original source hash mismatch: {name}",
            )
        _require(
            set(manifest["schemas"]) == {"capture", "mpmo"},
            "missing original transport schemas",
        )
        for transport, schemas in manifest["schemas"].items():
            _require(set(schemas) == set(STREAMS), "missing original output schemas")
            for output, schema in schemas.items():
                _validate_schema(schema, output, transport)
        for module, outputs in (
            ("rss", {"rss_stat_output"}),
            ("perf", {"dtlb_misses", "dtlb_accesses"}),
        ):
            selection = json.loads(artifacts[module + "_selection"].read_text())
            _require(
                type(selection["version"]) is int
                and selection["version"] == 1
                and set(selection["outputs"]) == outputs,
                "invalid original selection",
            )
            for output, request in selection["outputs"].items():
                schema = manifest["schemas"]["capture"][output]
                _require(
                    request["program"] == schema["program"]
                    and request["record"] == schema["source_event"],
                    "selection source provenance mismatch",
                )
                _require(
                    len(request["fields"]) == len(schema["fields"])
                    and set(request["fields"]) == {f["name"] for f in schema["fields"]},
                    "selection fields mismatch",
                )
        return manifest
    except (KeyError, TypeError, AttributeError, OSError) as exc:
        raise ValueError(f"invalid original RSS/DTLB manifest: {exc}") from exc


def decode_original_event(schema: dict, payload: bytes) -> dict[str, int]:
    """Decode only report-selected members, including their actual padding/widths."""
    _require(len(payload) == schema["size"], "unexpected original event size")
    return {
        field["name"]: int.from_bytes(
            payload[field["offset"] : field["offset"] + field["size"]],
            schema["byte_order"],
            signed=field["signed"],
        )
        for field in schema["fields"]
    }


def validate_original_status(rows, cid: str, transport: str) -> None:
    """Pure runner-side validation; capture stream-file checks belong to the runner."""
    _require(transport in ("capture", "mpmo"), "invalid original transport")
    rows = list(rows)
    _require(len(rows) == 1, "expected exactly one final original status row")
    row = rows[0]
    _require(isinstance(row, dict), "invalid original status row")
    for key, value in {
        "collection_id": cid,
        "origin": "original-bcc-llvm",
        "transport": transport,
        "counter_scope": COUNTER_SCOPE,
        "rss_schema_bytes": 24,
        "dtlb_schema_bytes": 32,
        "callback_errors": 0,
    }.items():
        _require(
            type(row.get(key)) is type(value) and row[key] == value,
            f"invalid original status {key}",
        )
    counters = [f"{source}_received" for source in STREAMS.values()]
    if transport == "mpmo":
        _require(
            type(row.get("data_slots")) is int
            and row["data_slots"] == 32
            and type(row.get("read_slot")) is int
            and row["read_slot"] == 0,
            "invalid original reader geometry",
        )
        counters += ["polls"] + [
            f"{source}_{kind}"
            for source in STREAMS.values()
            for kind in ("empty", "busy")
        ]
    else:
        counters += [f"lost_{source}" for source in STREAMS.values()]
    for key in counters:
        _require(_uint64(row.get(key)), f"invalid original counter {key}")
    for source in STREAMS.values():
        _require(row[f"{source}_received"] > 0, f"no original {source} records")
        if transport == "mpmo":
            _require(
                sum(row[f"{source}_{kind}"] for kind in ("received", "empty", "busy"))
                == row["polls"],
                f"inconsistent original {source} polls",
            )
    _require(
        isinstance(row.get("compiler_evidence"), str)
        and bool(row["compiler_evidence"]),
        "missing compiler evidence",
    )
    _require(not row.get("collector_error"), "original collector reported an error")


@contextmanager
def _evidence_owner(bundle):
    if os.geteuid() != 0:
        yield os.getuid(), os.getgid()
        return
    if "SUDO_UID" in os.environ:
        owner = (
            int(os.environ["SUDO_UID"]),
            int(os.environ.get("SUDO_GID", os.environ["SUDO_UID"])),
        )
    elif "UNAME" in os.environ:
        owner = get_user_group_ids()
    else:
        stat = bundle.stat()
        owner = stat.st_uid, stat.st_gid
    # Root-squashed NFS needs the workspace owner's filesystem credentials even
    # for LLVM's evidence writes. Keep effective BPF/perf privileges unchanged.
    libc = ct.CDLL(None)
    for function in (libc.setfsuid, libc.setfsgid):
        function.argtypes, function.restype = [ct.c_uint], ct.c_uint
    for function in (libc.capget, libc.capset):
        function.argtypes, function.restype = [ct.c_void_p, ct.c_void_p], ct.c_int
    header = (ct.c_uint32 * 2)(
        0x20080522, 0
    )  # Linux capability ABI v3, current thread.
    capabilities = (ct.c_uint32 * 6)()
    _require(
        libc.capget(header, capabilities) == 0, "cannot read evidence capabilities"
    )
    old_gid = libc.setfsgid(owner[1])
    old_uid = libc.setfsuid(owner[0])
    try:
        _require(
            libc.setfsuid(-1) == owner[0] and libc.setfsgid(-1) == owner[1],
            "cannot select evidence filesystem owner",
        )
        # setfsuid clears filesystem capabilities; retain the pre-existing ones
        # so BCC can still read root-only tracefs paths during auto-attachment.
        _require(libc.capset(header, capabilities) == 0, "cannot retain tracefs access")
        yield owner
    finally:
        libc.setfsuid(old_uid)
        libc.setfsgid(old_gid)
        _require(
            libc.capset(header, capabilities) == 0,
            "cannot restore evidence capabilities",
        )
        _require(
            libc.setfsuid(-1) == old_uid and libc.setfsgid(-1) == old_gid,
            "cannot restore evidence filesystem owner",
        )


class OriginalRSSDTLBHook:
    def __init__(self, manifest_path: str | Path, transport: str = "capture"):
        _require(transport in ("capture", "mpmo"), "invalid original transport")
        self.path = Path(manifest_path).resolve()
        self.manifest = original_manifest(self.path)
        self.transport = transport
        self.schemas = self.manifest["schemas"][transport]
        self._lock, self._native_lock = Lock(), Lock()
        self._probes, self._queues, self._compiled_programs = {}, {}, {}
        self._rss, self._perf = [], []
        self._stats = {f"{source}_received": 0 for source in STREAMS.values()}
        self._stats["callback_errors"] = 0
        if transport == "mpmo":
            self._stats.update(polls=0, data_slots=32, read_slot=0)
            self._stats.update(
                {
                    f"{source}_{kind}": 0
                    for source in STREAMS.values()
                    for kind in ("empty", "busy")
                }
            )
        else:
            self._stats.update({f"lost_{source}": 0 for source in STREAMS.values()})
        self._started = self._closed = self._status_emitted = False
        self._error = ""
        self._queue_read = self._queue_lib = None
        self.evidence = None

    def name(self) -> str:
        return f"original_rss_dtlb_{self.transport}"

    def _verify_private_bcc(self):
        import bcc
        from bcc.libbcc import lib

        prefix = Path(self.manifest["compiler_prefix"]).resolve()
        _require(
            Path(bcc.__file__).resolve().is_relative_to(prefix),
            "select private Python BCC before importing the collector",
        )

        class DlInfo(ct.Structure):
            _fields_ = [
                ("filename", ct.c_char_p),
                ("base", ct.c_void_p),
                ("symbol", ct.c_char_p),
                ("address", ct.c_void_p),
            ]

        dladdr = ct.CDLL(None).dladdr
        dladdr.argtypes, dladdr.restype = [ct.c_void_p, ct.POINTER(DlInfo)], ct.c_int
        info = DlInfo()
        _require(
            dladdr(
                ct.cast(lib.bpf_module_create_c_from_string, ct.c_void_p),
                ct.byref(info),
            )
            != 0
            and bool(info.filename),
            "cannot identify loaded libbcc",
        )
        expected = self.path.parent / self.manifest["files"]["bcc_library"]["name"]
        _require(
            Path(os.fsdecode(info.filename)).resolve() == expected.resolve(),
            "loaded libbcc is not the manifest's private library; set LD_LIBRARY_PATH before Python starts",
        )
        for key in ("source", "build"):
            _require(
                os.environ.get("BCC_KERNEL_" + key.upper())
                == self.manifest["kernel_" + key],
                f"BCC_KERNEL_{key.upper()} must match the manifest",
            )

    def _compile(self, module, text):
        prefix = self.evidence / module
        with Path(f"{prefix}.generated.c").open("x") as output:
            output.write(text)
        flags = (
            ("-DKERMIT_CAPTURE",)
            if self.transport == "capture"
            else ("-DKERMIT_QUEUE_SIZE=32",)
        )
        if self.transport == "mpmo" and module == "perf":
            flags += ("-DKERMIT_CPU_METADATA",)
        flags += tuple(
            f"-I{self.manifest['kernel_build']}/{suffix}"
            for suffix in (
                "include",
                "arch/x86/include/generated",
                "include/generated/uapi",
                "arch/x86/include/generated/uapi",
            )
        )
        changes = {
            "KERNMLOPS_BCC_REDUCE": str(
                self.path.parent / self.manifest["files"][module + "_selection"]["name"]
            ),
            "KERNMLOPS_BCC_EVIDENCE": str(prefix),
            "KERNMLOPS_BCC_EXPORT_ONLY": None,
        }
        previous = {key: os.environ.get(key) for key in changes}
        try:
            for key, value in changes.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value
            probe = compile_bpf(text, cflags=flags)
            self._probes[module] = probe
        finally:
            for key, value in previous.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value
        report = json.loads(Path(f"{prefix}.after.json").read_text())
        wanted = {
            name: schema
            for name, schema in self.schemas.items()
            if (name == "rss_stat_output") == (module == "rss")
        }
        _require(
            report.get("version") == 1
            and report.get("transport") == self.transport
            and report.get("export_only") is False,
            "invalid runtime compiler report",
        )
        actual = report["schemas"]
        _require(
            set(actual) == set(wanted), "runtime compiler outputs differ from manifest"
        )
        for name, schema in actual.items():
            # The LLVM report uses null for a capture map's inapplicable slot count.
            if self.transport == "capture" and schema["slots"] is None:
                schema["slots"] = 0
            _validate_schema(schema, name, self.transport)
            _require(
                all(schema.get(key) == value for key, value in wanted[name].items()),
                f"runtime {name} schema differs from manifest",
            )
        return probe

    def _map_queue(self, probe, output):
        schema = self.schemas[output]
        table = probe[schema["map"]]
        _require(
            table.ttype == 2
            and table.max_entries == 32
            and table.flags == 1024
            and ct.sizeof(table.Key) == 4
            and ct.sizeof(table.Leaf) == schema["map_value_size"],
            f"invalid runtime queue descriptor: {output}",
        )
        leaf_fields = dict(table.Leaf._fields_)
        _require(
            table.Leaf.valid.offset == 0
            and leaf_fields["valid"] is ct.c_uint64
            and table.Leaf.data.offset == schema["data_offset"],
            "invalid runtime queue prefix",
        )
        if schema["data_offset"] == 16:
            _require(
                table.Leaf.cpu.offset == 8 and leaf_fields["cpu"] is ct.c_uint64,
                "missing runtime queue CPU metadata",
            )
        data_type = leaf_fields["data"]
        _require(
            ct.sizeof(data_type) == schema["size"], "invalid runtime queue payload size"
        )
        members = dict(data_type._fields_)
        for field in schema["fields"]:
            member = members[field["name"]]
            _require(
                getattr(data_type, field["name"]).offset == field["offset"]
                and ct.sizeof(member) == field["size"]
                and (member(-1).value < 0) == field["signed"],
                "runtime queue field differs from schema",
            )
        length = (
            (schema["map_value_size"] * 32 + mmap.PAGESIZE - 1) // mmap.PAGESIZE
        ) * mmap.PAGESIZE
        mapping = mmap.mmap(
            table.get_fd(),
            length,
            flags=mmap.MAP_SHARED,
            prot=mmap.PROT_READ | mmap.PROT_WRITE,
        )
        try:
            address = ct.addressof(ct.c_char.from_buffer(mapping))
            snapshot, cpu = ct.create_string_buffer(schema["size"]), ct.c_uint64()
            self._queues[output] = (
                mapping,
                address,
                schema["data_offset"],
                schema["size"],
                snapshot,
                cpu,
            )
        except BaseException:
            mapping.close()
            raise

    def _callbacks(self, output):
        source = STREAMS[output]

        def sample(cpu, data, size):
            try:
                payload_size = self.schemas[output]["size"]
                # PERF_SAMPLE_RAW aligns its u32 length plus data to eight bytes.
                padded_size = ((payload_size + 4 + 7) // 8) * 8 - 4
                _require(bool(data), f"missing {output} callback payload")
                _require(
                    size in (payload_size, padded_size),
                    f"unexpected {output} callback size {size}; expected {payload_size} or {padded_size}",
                )
                event = decode_original_event(
                    self.schemas[output], ct.string_at(data, payload_size)
                )
                _require(_uint64(cpu), "invalid callback CPU")
                with self._lock:
                    if source == "rss":
                        self._rss.append(
                            (
                                event["ts"],
                                event["counter_value"],
                                event["tgid"],
                                event["member"],
                            )
                        )
                    else:
                        self._perf.append(
                            (
                                output,
                                cpu,
                                event["ts_uptime_us"],
                                event["count"],
                                event["enabled_time_us"],
                                event["running_time_us"],
                            )
                        )
                    self._stats[f"{source}_received"] += 1
            except Exception as exc:  # noqa: BLE001 - report callback errors from poll().
                with self._lock:
                    self._stats["callback_errors"] += 1
                    self._error = str(exc)

        def lost(cpu, count):
            with self._lock:
                if _uint64(count):
                    self._stats[f"lost_{source}"] += count
                else:
                    self._stats["callback_errors"] += 1
                    self._error = "invalid original lost callback"

        return sample, lost

    def load(self, collection_id: str):
        with self._native_lock, _COMPILE_LOCK:
            _require(
                not self._started and not self._closed,
                "original collector cannot be loaded twice",
            )
            self._started = True
            self.collection_id = collection_id
            _require(
                isinstance(collection_id, str)
                and re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", collection_id),
                "unsafe collection ID",
            )
            _require(
                original_manifest(self.path) == self.manifest,
                "original manifest changed before load",
            )
            self._verify_private_bcc()
            self.evidence = (
                self.path.parent / "evidence" / "collections" / collection_id
            )
            _require(
                self.evidence.resolve() == self.evidence,
                "evidence directory must not traverse symlinks",
            )
            try:
                with _evidence_owner(self.path.parent) as owner:
                    os.makedirs(self.evidence, exist_ok=False)
                    os.chown(self.evidence, *owner)
                    if self.transport == "mpmo":
                        self._queue_lib = ct.CDLL(
                            str(
                                self.path.parent
                                / self.manifest["files"]["queue_library"]["name"]
                            )
                        )
                        self._queue_read = self._queue_lib.original_queue_read
                        self._queue_read.argtypes = [
                            ct.c_void_p,
                            ct.c_size_t,
                            ct.c_size_t,
                            ct.c_void_p,
                            ct.POINTER(ct.c_uint64),
                        ]
                        self._queue_read.restype = ct.c_int
                    rss = self._compile("rss", TraceRSSStatBPFHook().bpf_text)
                    perf = PerfBPFHook(events=[DTLBPerfTable, DTLBAccessPerfTable])
                    perf.bpf = self._compile("perf", perf.bpf_text)
                    for output in STREAMS:
                        probe = rss if output == "rss_stat_output" else perf.bpf
                        if self.transport == "mpmo":
                            self._map_queue(probe, output)
                        else:
                            callback, lost = self._callbacks(output)
                            probe[self.schemas[output]["map"]].open_perf_buffer(
                                callback,
                                page_cnt=256 if output == "rss_stat_output" else 64,
                                lost_cb=lost,
                            )
                    for event, config in perf.loaded_hw_event_configs.items():
                        perf._attach_perf_event(
                            event.ev_type(), config, f"{event.name()}_on".encode(), 1000
                        )
                    for module, probe in self._probes.items():
                        expected = {
                            s["program"].encode()
                            for name, s in self.schemas.items()
                            if (name == "rss_stat_output") == (module == "rss")
                        }
                        if module == "rss":
                            expected.add(b"raw_tracepoint__rss_stat")
                        _require(
                            set(probe.funcs) == expected,
                            "unexpected loaded original programs",
                        )
                        for name in sorted(probe.funcs):
                            bytecode = probe.dump_func(name)
                            _require(
                                bool(bytecode) and len(bytecode) % 8 == 0,
                                "invalid original BPF bytecode",
                            )
                            filename = f"{module}.{name.decode()}.bpf.bin"
                            with (self.evidence / filename).open("xb") as output:
                                output.write(bytecode)
                            self._compiled_programs[name.decode()] = {
                                "module": module,
                                "name": filename,
                                "size": len(bytecode),
                                "sha256": hashlib.sha256(bytecode).hexdigest(),
                                "report_sha256": _sha256(
                                    self.evidence / f"{module}.after.json"
                                ),
                                "generated_source_sha256": _sha256(
                                    self.evidence / f"{module}.generated.c"
                                ),
                            }
            except BaseException as exc:
                self._error = str(exc)
                self._cleanup()
                raise

    def poll(self):
        with self._native_lock:
            _require(
                bool(self._probes) and not self._closed,
                "original collector is not loaded",
            )
            if self.transport == "capture":
                for probe in self._probes.values():
                    probe.perf_buffer_poll(timeout=POLL_TIMEOUT_MS)
            else:
                self._stats["polls"] += 1
                for output, (
                    _,
                    address,
                    offset,
                    size,
                    snapshot,
                    cpu,
                ) in self._queues.items():
                    result = self._queue_read(
                        address, offset, size, snapshot, ct.byref(cpu)
                    )
                    if result in (0, 1, 2):
                        self._stats[
                            f"{STREAMS[output]}_{('empty', 'received', 'busy')[result]}"
                        ] += 1
                    else:
                        self._stats["callback_errors"] += 1
                        self._error = (
                            f"invalid original queue state: {output}: {result}"
                        )
            if self._error:
                raise RuntimeError(self._error)

    def _cleanup(self):
        # The outer collector has joined its workers. Stop producers before unmap,
        # and attempt every release even when another BCC cleanup fails.
        errors = []
        for probe in self._probes.values():
            try:
                probe.cleanup()
            except BaseException as exc:  # noqa: BLE001 - re-raise after all releases.
                errors.append(exc)
                try:
                    probe.close()
                except BaseException as exc:  # noqa: BLE001 - retain the detach error too.
                    errors.append(exc)
        self._probes.clear()
        while self._queues:
            queue = self._queues.popitem()[1]
            try:
                queue[0].close()
            except BaseException as exc:  # noqa: BLE001 - re-raise after all releases.
                errors.append(exc)
        self._queue_read = self._queue_lib = None
        self._closed = True
        if errors:
            self._error = self._error or str(errors[0])
        return errors

    def close(self):
        with self._native_lock:
            if self._closed:
                return
            errors = self._cleanup()
            if errors:
                raise errors[0]

    def _tables(self, rss, perf, final=False):
        tables = []
        if self.transport == "capture":
            if rss:
                frame = pl.DataFrame(
                    rss,
                    schema={
                        "ts_ns": pl.UInt64,
                        "count": pl.UInt64,
                        "tgid": pl.UInt32,
                        "member": pl.Int32,
                    },
                    orient="row",
                )
                tables.append(PolicyRSSTable.from_df_id(frame, self.collection_id))
            if perf:
                frame = pl.DataFrame(
                    perf,
                    schema={
                        "counter": pl.String,
                        "cpu": pl.UInt64,
                        "ts_uptime_us": pl.UInt64,
                        "count": pl.UInt32,
                        "enabled_time_us": pl.UInt64,
                        "running_time_us": pl.UInt64,
                    },
                    orient="row",
                )
                tables.append(OriginalPerfTable.from_df_id(frame, self.collection_id))
        if final:
            row = self._stats | {
                "origin": "original-bcc-llvm",
                "transport": self.transport,
                "counter_scope": COUNTER_SCOPE,
                "rss_schema_bytes": 24,
                "dtlb_schema_bytes": 32,
                "compiler_evidence": str(self.evidence),
                "collector_error": self._error,
                "compiled_programs_json": json.dumps(
                    {
                        "programs": self._compiled_programs,
                        "sources": self.manifest["sources"],
                        "bcc_library_sha256": self.manifest["files"]["bcc_library"][
                            "sha256"
                        ],
                    },
                    sort_keys=True,
                ),
            }
            frame = pl.DataFrame(
                [row],
                schema_overrides={
                    key: pl.UInt64 for key, value in row.items() if type(value) is int
                },
            )
            tables.append(PolicyStatusTable.from_df_id(frame, self.collection_id))
        return tables

    def data(self):
        with self._lock:
            rss, perf = self._rss.copy(), self._perf.copy()
        return self._tables(rss, perf)

    def pop_data(self):
        with self._lock:
            rss, perf = self._rss, self._perf
            self._rss, self._perf = [], []
            final = self._closed and not self._status_emitted
            self._status_emitted |= final
        return self._tables(rss, perf, final)

    def clear(self):
        with self._lock:
            self._rss.clear()
            self._perf.clear()
