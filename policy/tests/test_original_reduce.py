"""Offline regression tests using the frozen, actual original BCC frontend IR.

No frontend is invoked, no replacement probes are used, and no BPF is loaded.
Native replay replaces only helper addresses/pseudo-fds and injects a CAS race.
"""
import argparse
import copy
import ctypes
import hashlib
import json
import re
import shlex
import struct
import subprocess
import tempfile
import unittest
from pathlib import Path


POLICY = Path(__file__).resolve().parents[1]
PARSER = argparse.ArgumentParser(add_help=False)
PARSER.add_argument("--build", type=Path, default=POLICY / "build/container")
PARSER.add_argument("--fixtures", type=Path)
PARSER.add_argument("--llvm-bindir", type=Path, default=Path("/usr/lib/llvm-18/bin"))
OPTIONS, TEST_ARGS = PARSER.parse_known_args()
BUILD = OPTIONS.build.resolve()
FIXTURES = OPTIONS.fixtures or BUILD / "original_bcc/evidence/frontend-v4"
BIN = OPTIONS.llvm_bindir

RSS = {"rss_stat_output": {"program": "tracepoint__kmem__rss_stat", "record": "rss_stat_output",
                            "fields": ["tgid", "ts", "member", "counter_value"]}}
PERF = {name: {"program": name + "_on", "record": "perf_event_data",
               "fields": ["ts_uptime_us", "count", "enabled_time_us", "running_time_us"]}
        for name in ("dtlb_misses", "dtlb_accesses")}
FULL = {
    "rss": {"pid": (0, 4, False), "tgid": (4, 4, False), "ts": (8, 8, False),
            "member": (16, 4, True), "counter_value": (24, 8, False)},
    "perf": {"pid": (0, 4, False), "tgid": (4, 4, False), "ts_uptime_us": (8, 8, False),
             "count": (16, 4, False), "enabled_time_us": (24, 8, False), "running_time_us": (32, 8, False)},
}
HELPERS = {
    1: ("replay_lookup", "ptr", "ptr, ptr"),
    2: ("replay_update", "i32", "ptr, ptr, ptr, i64"),
    3: ("replay_delete", "i32", "ptr, ptr"),
    4: ("replay_read", "i32", "ptr, i64, ptr"),
    5: ("replay_time", "i64", ""), 8: ("replay_cpu", "i64", ""),
    14: ("replay_pid", "i64", ""), 15: ("replay_uid", "i64", ""),
    25: ("replay_output", "i32", "ptr, ptr, i64, ptr, i32"),
    56: ("replay_perf", "i32", "ptr, ptr, i32"),
}


def run(*args, ok=True):
    result = subprocess.run(list(map(str, args)), capture_output=True, text=True, timeout=120)
    if ok and result.returncode:
        raise AssertionError(f"{' '.join(map(str, args))}\n{result.stdout}\n{result.stderr}")
    return result


def body(text, program):
    return re.search(r"^define [^\n]*@" + re.escape(program) + r"\([^\n]*\n(.*?)^}",
                     text, re.M | re.S)[1]


def values(payload, fields):
    return {name: int.from_bytes(payload[offset:offset + size], "little", signed=signed)
            for name, (offset, size, signed) in fields.items()}


def btf_types(path):
    """Read emitted BTF, including VAR -> BCC wrapper -> key/value references."""
    data = path.read_bytes()
    magic, version, flags, hdr, off, length, stroff, strlen = struct.unpack_from("<HBBIIIII", data)
    assert (magic, version) == (0xEB9F, 1)
    strings = data[hdr + stroff:hdr + stroff + strlen]
    name = lambda n: strings[n:strings.index(b"\0", n)].decode()
    result = [None]
    cursor = hdr + off
    sizes = {1: 4, 2: 0, 3: 12, 4: 12, 5: 12, 6: 8, 7: 0, 8: 0, 9: 0,
             10: 0, 11: 0, 12: 0, 13: 8, 14: 4, 15: 12, 16: 0, 17: 4, 18: 0, 19: 12}
    while cursor < hdr + off + length:
        n, info, size = struct.unpack_from("<III", data, cursor)
        cursor += 12
        kind, count = (info >> 24) & 31, info & 65535
        item = {"name": name(n), "kind": kind, "size": size, "members": []}
        if kind in (4, 5):
            for i in range(count):
                member, tid, offset = struct.unpack_from("<III", data, cursor + 12 * i)
                item["members"].append((name(member), tid, offset))
        result.append(item)
        cursor += sizes[kind] * (count if kind in (4, 5, 6, 13, 15, 19) else 1)
    return result


class Replay:
    def __init__(self, ir, metadata, work, name):
        text = ir.read_text()
        used = set()

        def helper(match):
            number = int(match[1])
            used.add(number)
            return "@" + HELPERS[number][0]

        text = re.sub(r"inttoptr \(i64 (\d+) to ptr\)", helper, text)
        text = text.replace("@llvm.bpf.pseudo", "@replay_pseudo")
        text = re.sub(r"(?m)^(\s*%[^\n=]+ = cmpxchg ptr ([^,]+),)",
                      r"  call void @replay_before_cas(ptr \2)\n\1", text)
        text += "\ndeclare void @replay_before_cas(ptr)\n"
        for number in sorted(used):
            function, ret, args = HELPERS[number]
            text += f"declare {ret} @{function}({args})\n"
        native = work / f"{name}.native.ll"
        native.write_text(text)
        library = work / f"{name}.so"
        run(BIN / "clang", "-shared", "-fPIC", "-O2", "-Wno-override-module",
            "-target", "x86_64-linux-gnu", native, POLICY / "tests/original_reduce_replay.c", "-o", library)
        self.lib = ctypes.CDLL(str(library))
        self.lib.replay_bytes.restype = ctypes.c_void_p
        self.metadata = metadata
        self.maps = {m["name"]: m for m in metadata["maps"]}
        for m in metadata["maps"]:
            kind = (1 if m["name"] == "rss_stat_hash" else
                    2 if m["name"].startswith("mpmo_head_") else
                    3 if m["name"].startswith("mpmo_buff_") else 4)
            self.lib.replay_config(m["fake_fd"], kind, m["value_size"], m["max_entries"])

    def integer(self, name, value=None):
        item = ctypes.c_int.in_dll(self.lib, "replay_" + name)
        if value is not None:
            item.value = value
        return item.value

    def address(self, name):
        return self.lib.replay_bytes(self.maps[name]["fake_fd"])

    def invoke(self, stream, output, state=0, fail=0, head=0, rss_size=12345 * 4096, insert_twice=False):
        self.lib.replay_reset()
        self.integer("fail", fail)
        request = (RSS if stream == "rss" else PERF)[output]
        mpmo = "mpmo_buff_" + output in self.maps
        if mpmo:
            buffer_map = self.maps["mpmo_buff_" + output]
            address = self.address(buffer_map["name"])
            for i in range(32):
                ctypes.c_uint64.from_address(address + i * buffer_map["value_size"]).value = state
            ctypes.c_uint64.from_address(self.address("mpmo_head_" + output)).value = head
        ctx = ctypes.create_string_buffer(256)
        if stream == "rss":
            raw = (ctypes.c_uint64 * 1)(0x200000)
            self.lib.raw_tracepoint__rss_stat(raw)
            if insert_twice:
                self.lib.raw_tracepoint__rss_stat(raw)
            bridge = ctypes.string_at(self.address("rss_stat_hash"), 32)
            struct.pack_into("<i", ctx, 16, -3)
            struct.pack_into("<q", ctx, 24, rss_size)
        else:
            bridge = None
        getattr(self.lib, request["program"])(ctx)
        log = list((ctypes.c_int * self.integer("log_size")).in_dll(self.lib, "replay_log"))
        if mpmo:
            map_size = buffer_map["value_size"]
            schema = self.metadata.get("schemas", {}).get(output)
            size = schema["size"] if schema else (32 if stream == "rss" else 40)
            offset = schema["data_offset"] if schema else map_size - size
            buffer = ctypes.string_at(address, 32 * map_size)
            valid = [struct.unpack_from("<Q", buffer, i * map_size)[0] for i in range(32)]
            slot = head % 32
            payload = buffer[slot * map_size + offset:slot * map_size + offset + size]
            final_head = ctypes.c_uint64.from_address(self.address("mpmo_head_" + output)).value
            emitted = valid[slot] == 1 and log.count(5) != 0
        else:
            valid, final_head = [], None
            emitted = self.integer("emits") != 0
            payload = bytes((ctypes.c_ubyte * self.integer("payload_size")).in_dll(self.lib, "replay_payload"))
        present = (self.lib.replay_present(self.maps["rss_stat_hash"]["fake_fd"])
                   if stream == "rss" else None)
        return {"payload": payload, "emitted": emitted, "valid": valid, "head": final_head,
                "prefix": buffer[slot * map_size:slot * map_size + offset] if mpmo else b"",
                "bridge": bridge, "present": present, "log": log}


class OriginalReduceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        root = BUILD / "original_reduce"
        root.mkdir(exist_ok=True)
        cls.work = Path(tempfile.mkdtemp(prefix="run-", dir=root))
        cls.hashes = {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in FIXTURES.iterdir()}
        cls.cases = {}
        for stream, outputs in (("rss", RSS), ("perf", PERF)):
            selection = cls.work / f"{stream}.selection.json"
            selection.write_text(json.dumps({"version": 1, "outputs": outputs}))
            for transport in ("capture", "mpmo"):
                name = f"{stream}-{transport}"
                source = FIXTURES / f"{name}.before.bc"
                before = json.loads((FIXTURES / f"{name}.before.json").read_text())
                prefix = cls.work / name
                run(BUILD / "original-reduce", "--input", source,
                    "--metadata", FIXTURES / f"{name}.before.json", "--selection", selection,
                    "--output", prefix)
                after = json.loads(prefix.with_suffix(".after.json").read_text())
                after_ir = prefix.with_suffix(".after.ll")
                full_ir = cls.work / f"{name}.full.ll"
                run(BIN / "opt", "-S",
                    "-passes=globalopt,always-inline,function(sroa,mem2reg,instcombine,simplifycfg,dse,adce)",
                    source, "-o", full_ir)
                cls.cases[name] = {
                    "before": before, "after": after, "ir": after_ir.read_text(), "full_ir": full_ir.read_text(),
                    "prefix": prefix,
                    "full": Replay(full_ir, before, cls.work, name + ".full"),
                    "selected": Replay(after_ir, after, cls.work, name + ".selected"),
                }
        print(f"Original reduction evidence: {cls.work}")

    @classmethod
    def tearDownClass(cls):
        for path, digest in cls.hashes.items():
            assert hashlib.sha256(path.read_bytes()).hexdigest() == digest, path

    def test_compact_schemas_and_map_descriptors(self):
        for name, case in self.cases.items():
            stream, transport = name.split("-")
            before, after = case["before"], case["after"]
            self.assertEqual(after["transport"], transport)
            originals = {m["name"]: m for m in before["maps"]}
            for output, schema in after["schemas"].items():
                self.assertEqual(schema["size"], 24 if stream == "rss" else 32)
                self.assertEqual(schema["original_size"], 32 if stream == "rss" else 40)
                self.assertEqual(schema["slots"], 32 if transport == "mpmo" else None)
                self.assertEqual(schema["data_offset"], 8 if transport == "mpmo" else 0)
                self.assertEqual([f["name"] for f in schema["fields"]],
                                 ["ts", "counter_value", "tgid", "member"] if stream == "rss" else
                                 ["ts_uptime_us", "enabled_time_us", "running_time_us", "count"])
                for f in schema["fields"]:
                    self.assertEqual((f["size"], f["signed"]), FULL[stream][f["name"]][1:])
                if transport == "capture":
                    self.assertEqual([f.split("#")[0] for f in after["perf_events"][output]],
                                     [f["name"] for f in schema["fields"]])
            for m in after["maps"]:
                old = originals[m["name"]]
                if not m["name"].startswith("mpmo_buff_"):
                    self.assertEqual(m, old)
                    continue
                self.assertEqual({k: v for k, v in m.items() if k not in ("value_size", "value_desc")},
                                 {k: v for k, v in old.items() if k not in ("value_size", "value_desc")})
                self.assertEqual(m["value_size"], old["value_size"] - 8)
                desc = json.loads(m["value_desc"])
                self.assertEqual(desc[1][0], ["valid", "unsigned long long"])
                fields = desc[1][-1][1][1]
                self.assertNotIn("pid", [f[0] for f in fields])

    def test_native_selected_values_and_failure_paths(self):
        scenarios = [(0, 0), (1, 0), (2, 0), (3, 0)]
        scenarios += [(0, bit) for bit in (1, 2, 4, 8, 16, 32, 64, 128)]
        for name, case in self.cases.items():
            stream, transport = name.split("-")
            for output, schema in case["after"]["schemas"].items():
                fields = {f["name"]: (f["offset"], f["size"], f["signed"]) for f in schema["fields"]}
                for state, fail in scenarios:
                    for head in (0, 31, 32, (1 << 64) - 1):
                        with self.subTest(name=name, output=output, state=state, fail=fail, head=head):
                            full = case["full"].invoke(stream, output, state, fail, head)
                            selected = case["selected"].invoke(stream, output, state, fail, head)
                            for key in ("emitted", "valid", "head", "bridge", "present"):
                                self.assertEqual(selected[key], full[key], key)
                            # Only pure measurements for dropped fields may disappear.
                            self.assertEqual([v for v in selected["log"] if v != 14],
                                             [v for v in full["log"] if v != 14])
                            if full["emitted"]:
                                old = values(full["payload"], FULL[stream])
                                new = values(selected["payload"], fields)
                                self.assertEqual(new, {k: old[k] for k in new})
                                if stream == "rss":
                                    self.assertEqual(new["member"], -3)
                                    self.assertEqual(new["counter_value"], 12345)
                                    self.assertEqual(new["ts"], 9876543210123)
                                else:
                                    self.assertEqual(new["count"], 0xFEDCBA98)
                                    self.assertEqual(new["ts_uptime_us"], 9876543210)
                                    self.assertEqual(new["enabled_time_us"], 123456)
                                    self.assertEqual(new["running_time_us"], 98765)

    def test_rss_bridge_insert_noexist_and_negative_pages(self):
        for transport in ("capture", "mpmo"):
            case = self.cases["rss-" + transport]
            a = case["full"].invoke("rss", "rss_stat_output", rss_size=-4097, insert_twice=True)
            b = case["selected"].invoke("rss", "rss_stat_output", rss_size=-4097, insert_twice=True)
            self.assertEqual(a["bridge"], b["bridge"])
            self.assertEqual(struct.unpack_from("<II", a["bridge"]), (101, 202))
            self.assertEqual(a["bridge"][8:], bytes(24))
            self.assertEqual(a["present"], 0)
            self.assertEqual(b["present"], 0)
            schema = case["after"]["schemas"]["rss_stat_output"]
            field = next(f for f in schema["fields"] if f["name"] == "counter_value")
            self.assertEqual(int.from_bytes(b["payload"][field["offset"]:field["offset"] + 8], "little"),
                             (1 << 64) - 2)

    def test_llvm_verification_helper_pruning_and_queue_protocol(self):
        for name, case in self.cases.items():
            run(BIN / "opt", "-passes=verify", "-disable-output", case["prefix"].with_suffix(".after.bc"))
            stream, transport = name.split("-")
            for output, schema in case["after"]["schemas"].items():
                selected = body(case["ir"], schema["program"])
                if stream == "perf":
                    self.assertNotIn("inttoptr (i64 14 to ptr)", selected)
                    self.assertEqual(selected.count("udiv i64"), 3)
                    self.assertRegex(selected, r"trunc i64 %\S+ to i32")
                    self.assertIn("inttoptr (i64 56 to ptr)", selected)
                if transport == "mpmo":
                    self.assertRegex(selected, r"atomicrmw add ptr .*i64 1 monotonic")
                    self.assertRegex(selected, r"(and i(?:32|64) .*31|urem i64 .*32)")
                    self.assertRegex(selected, r"cmpxchg ptr .*i64 3 acquire monotonic")
                    self.assertRegex(selected, r"atomicrmw xchg ptr .*i64 1 release")
                    self.assertIn("load volatile i64", selected)
                    self.assertLess(selected.index("cmpxchg"), selected.index("inttoptr (i64 5 to ptr)"))
                else:
                    self.assertRegex(selected, rf"inttoptr \(i64 25 to ptr\).*i32 noundef {schema['size']}\)")
            if stream == "rss":
                raw = body(case["ir"], "raw_tracepoint__rss_stat")
                self.assertEqual(raw.count("inttoptr (i64 4 to ptr)"), 4)
                self.assertRegex(raw, r"inttoptr \(i64 2 to ptr\).*i64 noundef 1\)")
                self.assertIn("inttoptr (i64 3 to ptr)", case["ir"])

    def test_all_32_slots_and_reserve_order(self):
        for stream in ("rss", "perf"):
            case = self.cases[stream + "-mpmo"]
            for output in case["after"]["schemas"]:
                for head in range(64):
                    result = case["selected"].invoke(stream, output, head=head)
                    self.assertEqual(result["head"], head + 1)
                    self.assertEqual(result["valid"], [int(i == head % 32) for i in range(32)])
                    self.assertLess(result["log"].index(102), result["log"].index(103))
                    self.assertLess(result["log"].index(103), result["log"].index(5))
                    reference = case["full"].invoke(stream, output, head=head)
                    self.assertEqual([h for h in result["log"] if h != 14],
                                     [h for h in reference["log"] if h != 14])

    def test_additional_selections_remove_dead_measurements(self):
        for stream, outputs in (("rss", RSS), ("perf", PERF)):
            requests = copy.deepcopy(outputs)
            for request in requests.values():
                request["fields"] = ["counter_value" if stream == "rss" else "count"]
            path = self.work / f"{stream}-minimal.selection.json"
            path.write_text(json.dumps({"version": 1, "outputs": requests}))
            for transport in ("capture", "mpmo"):
                prefix = self.work / f"{stream}-{transport}-minimal"
                run(BUILD / "original-reduce", "--input", FIXTURES / f"{stream}-{transport}.before.bc",
                    "--metadata", FIXTURES / f"{stream}-{transport}.before.json",
                    "--selection", path, "--output", prefix)
                after = json.loads(prefix.with_suffix(".after.json").read_text())
                text = prefix.with_suffix(".after.ll").read_text()
                run(BIN / "opt", "-passes=verify", "-disable-output", prefix.with_suffix(".after.bc"))
                for output, request in requests.items():
                    selected = body(text, request["program"])
                    self.assertNotIn("inttoptr (i64 5 to ptr)", selected)
                    self.assertNotIn("udiv i64", selected)
                    self.assertEqual(after["schemas"][output]["size"], 8 if stream == "rss" else 4)
                    if stream == "perf":
                        self.assertNotIn("inttoptr (i64 14 to ptr)", selected)
                        # The perf status guard is semantically live even if time fields disappear.
                        self.assertIn("inttoptr (i64 56 to ptr)", selected)
                        self.assertRegex(selected, r"icmp eq i32 %\S+, 0")
                    else:
                        self.assertIn("inttoptr (i64 3 to ptr)", selected)
                        self.assertEqual(body(text, "raw_tracepoint__rss_stat").count("inttoptr (i64 4 to ptr)"), 4)

    def test_helper4_dead_chains_but_live_status_are_preserved(self):
        original = self.cases["rss-capture"]["full_ir"]
        program = RSS["rss_stat_output"]["program"]
        entry = body(original, program)
        record = re.search(r"(%\S+) = alloca %struct.rss_stat_output", entry)[1]
        reads = (
            "  %read_owner = alloca ptr, align 8\n"
            "  %read_pid_dest = alloca i32, align 4\n"
            "  store ptr null, ptr %read_owner, align 8\n"
            "  store i32 0, ptr %read_pid_dest, align 4\n"
            "  %read_owner_status = call i32 inttoptr (i64 4 to ptr)(ptr %read_owner, i64 8, ptr %0)\n"
            "  %read_owner_value = load ptr, ptr %read_owner, align 8\n"
            "  %read_status = call i32 inttoptr (i64 4 to ptr)(ptr %read_pid_dest, i64 4, ptr %read_owner_value)\n"
            "  %read_pid = load i32, ptr %read_pid_dest, align 4\n"
        )
        changed, count = re.subn(rf"store i32 %\S+, ptr {re.escape(record)},", f"store i32 %read_pid, ptr {record},", entry)
        self.assertEqual(count, 1)
        for live_status in (False, True):
            replacement = reads + changed
            if live_status:
                replacement = replacement.replace("ret i32 0", "ret i32 %read_status")
            source = self.work / f"read-chain-{live_status}.input.ll"
            source.write_text(original.replace(entry, replacement))
            prefix = self.work / f"read-chain-{live_status}"
            run(BUILD / "original-reduce", "--input", source,
                "--metadata", FIXTURES / "rss-capture.before.json",
                "--selection", self.work / "rss.selection.json", "--output", prefix)
            after = prefix.with_suffix(".after.ll").read_text()
            self.assertEqual(body(after, program).count("inttoptr (i64 4 to ptr)"), 2 if live_status else 0)
            self.assertEqual(body(after, "raw_tracepoint__rss_stat").count("inttoptr (i64 4 to ptr)"), 4)
            run(BIN / "opt", "-passes=verify", "-disable-output", prefix.with_suffix(".after.bc"))

    def test_each_capture_submit_is_validated_and_projected(self):
        original = self.cases["rss-capture"]["full_ir"]
        submit = re.search(r"(?m)^  %\S+ = call i32 inttoptr \(i64 25 to ptr\)[^\n]*", original)[0]
        for valid in (True, False):
            second = re.sub(r"^  %\S+ =", "  %second_submit =", submit)
            if not valid:
                second = second.replace("i32 noundef 32)", "i32 noundef 31)")
            source = self.work / f"two-submits-{valid}.input.ll"
            source.write_text(original.replace(submit, submit + "\n" + second))
            prefix = self.work / f"two-submits-{valid}"
            result = run(BUILD / "original-reduce", "--input", source,
                         "--metadata", FIXTURES / "rss-capture.before.json",
                         "--selection", self.work / "rss.selection.json", "--output", prefix, ok=valid)
            if valid:
                text = prefix.with_suffix(".after.ll").read_text()
                calls = re.findall(r"inttoptr \(i64 25 to ptr\)[^\n]*", text)
                self.assertEqual(len(calls), 2)
                self.assertTrue(all("i32 noundef 24)" in call for call in calls))
            else:
                self.assertNotEqual(result.returncode, 0)
                self.assertFalse(prefix.with_suffix(".after.bc").exists())

    def test_independent_layouts_for_shared_perf_record(self):
        requests = copy.deepcopy(PERF)
        requests["dtlb_misses"]["fields"] = ["count"]
        requests["dtlb_accesses"]["fields"] = ["ts_uptime_us", "enabled_time_us"]
        path = self.work / "asymmetric.selection.json"
        path.write_text(json.dumps({"version": 1, "outputs": requests}))
        for transport in ("capture", "mpmo"):
            prefix = self.work / f"asymmetric-{transport}"
            run(BUILD / "original-reduce", "--input", FIXTURES / f"perf-{transport}.before.bc",
                "--metadata", FIXTURES / f"perf-{transport}.before.json", "--selection", path, "--output", prefix)
            metadata = json.loads(prefix.with_suffix(".after.json").read_text())
            replay = Replay(prefix.with_suffix(".after.ll"), metadata, self.work, f"asymmetric-{transport}")
            for output, expected in (("dtlb_misses", {"count": 0xFEDCBA98}),
                                     ("dtlb_accesses", {"ts_uptime_us": 9876543210, "enabled_time_us": 123456})):
                actual = replay.invoke("perf", output)
                schema = metadata["schemas"][output]
                fields = {f["name"]: (f["offset"], f["size"], f["signed"]) for f in schema["fields"]}
                self.assertEqual(values(actual["payload"], fields), expected)
                self.assertEqual(schema["size"], 4 if output == "dtlb_misses" else 16)

    def test_private_bcc_hook_compiles(self):
        # Compile against the actual private class/TableDesc definitions. This
        # neither rebuilds nor links BCC, and needs no Clang frontend headers.
        private = BUILD / "original_bcc/source/src/cc"
        flags = shlex.split(run(BIN / "llvm-config", "--cxxflags").stdout)
        run(BIN / "clang++", *flags, "-std=c++17", "-fno-rtti", "-Werror",
            "-I" + str(private), "-I" + str(POLICY), "-include", "bpf_module.h",
            "-include", "bcc_reduce.hpp", "-x", "c++", "-fsyntax-only", "/dev/null")

    def test_generic_cpu_slot_prefix_from_original_ir(self):
        # Derive a CPU-prefix fixture from the real export, not from a new C
        # probe/frontend. A future native KERMIT_CPU_METADATA export can replace
        # this fixture construction without changing the reduction/replay check.
        text = (FIXTURES / "perf-mpmo.before.ll").read_text()
        metadata = copy.deepcopy(self.cases["perf-mpmo"]["before"])
        next_id = max(map(int, re.findall(r"(?m)^!(\d+) =", text))) + 1
        for output in PERF:
            slot = "kermit_slot_" + output
            text = text.replace(f"%struct.{slot} = type {{ i64, %struct.perf_event_data }}",
                                f"%struct.{slot} = type {{ i64, i64, %struct.perf_event_data }}")
            pattern = rf"(?m)^(  %\S+ = getelementptr inbounds %struct\.{slot}, ptr (%\S+), i32 0, i32 )1([^\n]*)$"

            def cpu_store(match):
                return (match[1] + "2" + match[3] + "\n"
                        f"  %cpu_value_{output} = call i64 inttoptr (i64 8 to ptr)()\n"
                        f"  %cpu_address_{output} = getelementptr inbounds %struct.{slot}, ptr {match[2]}, i32 0, i32 1\n"
                        f"  store i64 %cpu_value_{output}, ptr %cpu_address_{output}, align 8")

            text, changed = re.subn(pattern, cpu_store, text)
            self.assertEqual(changed, 1)
            for type_name in (slot, "mpmo_buff_" + output + "_table_t", "____btf_map_mpmo_buff_" + output):
                match = re.search(rf'(?m)^!(\d+) = distinct !DICompositeType\(tag: DW_TAG_structure_type, name: "{type_name}",[^\n]*$', text)
                type_id = match[1]
                text = text.replace(match[0], re.sub(r"size: (\d+)", lambda m: f"size: {int(m[1]) + 64}", match[0], count=1))
                member_pattern = rf"(?m)^!(\d+) = !DIDerivedType\(tag: DW_TAG_member, name: \"([^\"]+)\", scope: !{type_id},[^\n]*$"
                member_lines = list(re.finditer(member_pattern, text))
                for member in member_lines:
                    line = member[0]
                    if member[2] in ("leaf", "value"):
                        line = re.sub(r"size: (\d+)", lambda m: f"size: {int(m[1]) + 64}", line, count=1)
                    elif member[2] != "valid":
                        line = re.sub(r"offset: (\d+)", lambda m: f"offset: {int(m[1]) + 64}", line)
                    text = text.replace(member[0], line)
                if type_name == slot:
                    elements = re.search(r"elements: !(\d+)", match[0])[1]
                    valid = member_lines[0]
                    cpu = valid[0].replace("!" + valid[1] + " =", f"!{next_id} =", 1)
                    cpu = cpu.replace('name: "valid"', 'name: "cpu"').removesuffix(")") + ", offset: 64)"
                    text += "\n" + cpu + "\n"
                    text = re.sub(rf"(?m)^!{elements} = !\{{!(\d+), !(\d+)\}}$",
                                  rf"!{elements} = !{{!\1, !{next_id}, !\2}}", text)
                    next_id += 1
            m = next(m for m in metadata["maps"] if m["name"] == "mpmo_buff_" + output)
            desc = json.loads(m["value_desc"])
            desc[1].insert(1, ["cpu", "unsigned long long"])
            m["value_desc"] = json.dumps(desc)
            m["value_size"] += 8
        text = re.sub(r"(getelementptr inbounds i8, ptr %\S+, i64 )-8", r"\g<1>-16", text)
        ir = self.work / "perf-cpu.input.ll"
        ir.write_text(text)
        meta = self.work / "perf-cpu.input.json"
        meta.write_text(json.dumps(metadata))
        run(BIN / "opt", "-passes=verify", "-disable-output", ir)
        prefix = self.work / "perf-cpu"
        run(BUILD / "original-reduce", "--input", ir, "--metadata", meta,
            "--selection", self.work / "perf.selection.json", "--output", prefix)
        after = json.loads(prefix.with_suffix(".after.json").read_text())
        full = self.work / "perf-cpu.full.ll"
        run(BIN / "opt", "-S", "-passes=globalopt,always-inline,function(sroa,mem2reg,instcombine,simplifycfg)",
            ir, "-o", full)
        original = Replay(full, metadata, self.work, "perf-cpu.full")
        selected = Replay(prefix.with_suffix(".after.ll"), after, self.work, "perf-cpu.selected")
        for output, schema in after["schemas"].items():
            self.assertEqual((schema["size"], schema["data_offset"], schema["map_value_size"], schema["slots"]),
                             (32, 16, 48, 32))
            for state in range(4):
                a = original.invoke("perf", output, state=state)
                b = selected.invoke("perf", output, state=state)
                for key in ("head", "valid", "prefix", "emitted"):
                    self.assertEqual(a[key], b[key])
                if b["emitted"]:
                    self.assertEqual(struct.unpack("<QQ", b["prefix"]), (1, 7))
                    self.assertIn(8, b["log"])
                    fields = {f["name"]: (f["offset"], f["size"], f["signed"]) for f in schema["fields"]}
                    old = values(a["payload"], FULL["perf"])
                    self.assertEqual(values(b["payload"], fields), {f: old[f] for f in fields})

    def test_emitted_btf_agrees_with_metadata_and_keeps_other_maps(self):
        for name, case in self.cases.items():
            optimized = self.work / f"{name}.codegen.bc"
            obj = self.work / f"{name}.bpf.o"
            btf = self.work / f"{name}.btf"
            run(BIN / "opt", "-passes=default<O3>", case["prefix"].with_suffix(".after.bc"), "-o", optimized)
            run(BIN / "llc", "-march=bpf", "-mcpu=v3", "-filetype=obj", optimized, "-o", obj)
            run(BIN / "llvm-objcopy", "--dump-section", f".BTF={btf}", obj)
            types = btf_types(btf)
            variables = {t["name"]: types[t["size"]] for t in types[1:] if t["kind"] == 14}

            def resolve(tid):
                t = types[tid]
                while t["kind"] in (8, 9, 10, 11, 18):
                    t = types[t["size"]]
                return t

            for m in case["after"]["maps"]:
                wrapper = variables.get("____btf_map_" + m["name"])
                if not wrapper:
                    self.assertEqual(m["type"], 4)
                    continue
                self.assertEqual(resolve(wrapper["members"][0][1])["size"], m["key_size"])
                leaf = resolve(wrapper["members"][1][1])
                self.assertEqual(leaf["size"], m["value_size"])
                table = variables[m["name"]]
                self.assertEqual(resolve(table["members"][1][1])["size"], m["value_size"])
                if m["name"].startswith("mpmo_buff_"):
                    schema = case["after"]["schemas"][m["name"].removeprefix("mpmo_buff_")]
                    data = leaf["members"][-1]
                    self.assertEqual((data[0], data[2]), ("data", 8 * schema["data_offset"]))
                    record = resolve(data[1])
                    self.assertEqual(record["size"], schema["size"])
                    self.assertEqual([(n, off // 8, resolve(tid)["size"]) for n, tid, off in record["members"]],
                                     [(f["name"], f["offset"], f["size"]) for f in schema["fields"]])

    def test_fail_closed_and_exclusive_evidence(self):
        cases = []
        for mutate in (lambda s: s.update(version=2),
                       lambda s: s["outputs"]["rss_stat_output"].update(fields=["missing"]),
                       lambda s: s["outputs"]["rss_stat_output"].update(fields=["tgid", "tgid"]),
                       lambda s: s["outputs"]["rss_stat_output"].update(fields=[]),
                       lambda s: s["outputs"]["rss_stat_output"].update(program="raw_tracepoint__rss_stat")):
            selection = {"version": 1, "outputs": copy.deepcopy(RSS)}
            mutate(selection)
            cases.append(selection)
        for index, selection in enumerate(cases):
            path = self.work / f"invalid-{index}.json"
            path.write_text(json.dumps(selection))
            prefix = self.work / f"invalid-{index}"
            result = run(BUILD / "original-reduce", "--input", FIXTURES / "rss-capture.before.bc",
                         "--metadata", FIXTURES / "rss-capture.before.json", "--selection", path,
                         "--output", prefix, ok=False)
            self.assertNotEqual(result.returncode, 0)
            self.assertTrue(prefix.with_suffix(".before.bc").exists())
            self.assertFalse(prefix.with_suffix(".after.bc").exists())
        prefix = self.cases["rss-capture"]["prefix"]
        digest = hashlib.sha256(prefix.with_suffix(".before.bc").read_bytes()).hexdigest()
        result = run(BUILD / "original-reduce", "--input", FIXTURES / "rss-capture.before.bc",
                     "--metadata", FIXTURES / "rss-capture.before.json", "--selection", self.work / "rss.selection.json",
                     "--output", prefix, ok=False)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("File exists", result.stderr)
        self.assertEqual(hashlib.sha256(prefix.with_suffix(".before.bc").read_bytes()).hexdigest(), digest)


if __name__ == "__main__":
    unittest.main(argv=[__file__, *TEST_ARGS])
