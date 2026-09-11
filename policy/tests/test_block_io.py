"""Compare full and extracted block-I/O programs without loading kernel BPF.

The production build uses CO-RE. A second build uses the same probe logic with
local request layouts; replacing helper addresses allows deterministic native
replay of both the original IR and the extracted IR with identical inputs.
"""

import argparse
import json
import re
import struct
import subprocess
import sys
import unittest
from pathlib import Path


POLICY = Path(__file__).resolve().parents[1]
SOURCE = POLICY / "bpf/block_io.bpf.c"
PROGRAMS = {"starts": "block_rq_issue", "ends": "block_rq_complete"}
EVENTS = {"starts": "block_io_start_perf_event", "ends": "block_io_end_perf_event"}
FULL_FIELDS = {
    "starts": {
        "device": (0, 4), "sector": (8, 8), "segments": (16, 4),
        "block_io_bytes": (20, 4), "block_io_start_uptime_us": (24, 8),
        "block_io_flags": (32, 8), "queue_length_segments": (40, 4),
        "queue_length_4ks": (44, 4),
    },
    "ends": {
        "device": (0, 4), "sector": (8, 8), "segments": (16, 4),
        "block_io_bytes": (20, 4), "block_io_end_uptime_us": (24, 8),
        "block_latency_us": (32, 8), "block_io_latency_us": (40, 8),
        "block_io_flags": (48, 8),
    },
}
PROFILES = {
    "latency": ("ends", "block_latency_us,block_io_latency_us", 16),
    "queue": ("starts", "queue_length_segments,queue_length_4ks", 8),
    "identity": ("ends", "device,block_io_end_uptime_us", 16),
}
HELPERS = {
    1: ("replay_lookup", "ptr (ptr, ptr)"),
    2: ("replay_update", "i64 (ptr, ptr, ptr, i64)"),
    3: ("replay_delete", "i64 (ptr, ptr)"),
    5: ("replay_time", "i64 ()"),
    113: ("replay_read", "i64 (ptr, i32, ptr)"),
    130: ("replay_output", "i64 (ptr, ptr, i64, i64)"),
}


def options():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--build", type=Path, default=POLICY / "build")
    parser.add_argument("--clang", default="clang-18")
    parser.add_argument("--objdump", default="llvm-objdump-18")
    parser.add_argument("--llvm-dis", default="llvm-dis-18")
    parser.add_argument("--multiarch", default="x86_64-linux-gnu")
    return parser.parse_known_args()


OPTIONS, TEST_ARGS = options()
BUILD = OPTIONS.build.resolve()
WORK = BUILD / "block_io"
EXTRACTOR = BUILD / "field-extract"


def run(*args):
    result = subprocess.run(list(map(str, args)), text=True, capture_output=True, timeout=120)
    if result.returncode:
        raise AssertionError(f"{' '.join(map(str, args))}\n{result.stdout}\n{result.stderr}")
    return result.stdout


def extract(source, name, stream, fields):
    output = WORK / f"{name}.bc"
    schema = WORK / f"{name}.schema.json"
    run(EXTRACTOR, "--input", source, "--output", output, "--schema", schema,
        "--program", PROGRAMS[stream], "--event", EVENTS[stream], "--fields", fields,
        "--preserve-other-programs")
    return output, json.loads(schema.read_text())


def disassemble(bitcode, name):
    obj = WORK / f"{name}.bpf.o"
    run(OPTIONS.clang, "-target", "bpf", "-g", "-O2", "-c", bitcode, "-o", obj)
    text = run(OPTIONS.objdump, "-hdr", "--no-show-raw-insn", obj)
    (WORK / f"{name}.disasm").write_text(text)
    result = {}
    for stream, program in PROGRAMS.items():
        section = f"raw_tp/{program}"
        size = re.search(rf"(?m)^\s*\d+\s+{re.escape(section)}\s+([0-9a-fA-F]+)\b", text)
        if not size:
            raise AssertionError(f"Missing companion section {section}\n{text}")
        body = text.split(f"Disassembly of section {section}:\n", 1)[1]
        body = body.split("Disassembly of section ", 1)[0]
        helpers = [int(x, 0) for x in re.findall(r"\bcall\s+(0x[0-9a-fA-F]+|[0-9]+)\b", body)]
        result[stream] = {"instructions": int(size[1], 16) // 8, "helpers": helpers}
    return result


def replay(bitcode, name):
    ir = WORK / f"{name}.native.ll"
    run(OPTIONS.llvm_dis, bitcode, "-o", ir)
    text = ir.read_text()
    used = set()

    def helper(match):
        number = int(match[1])
        if number not in HELPERS:
            raise AssertionError(f"Replay has no model for helper {number}")
        used.add(number)
        return "@" + HELPERS[number][0]

    text = re.sub(r"inttoptr \(i64 (\d+) to ptr\)", helper, text)
    if "llvm.preserve." in text or "llvm.bpf." in text:
        raise AssertionError("CO-RE intrinsics must not be sent to the native backend")
    for number in sorted(used):
        function, signature = HELPERS[number]
        return_type, arguments = signature.split(" ", 1)
        text += f"\ndeclare {return_type} @{function}{arguments}\n"
    ir.write_text(text)
    executable = WORK / f"{name}.replay"
    run(OPTIONS.clang, "-target", OPTIONS.multiarch, "-Wno-override-module", "-O2",
        ir, POLICY / "tests/block_io_replay.c", "-o", executable)
    output = run(executable)
    (WORK / f"{name}.replay.txt").write_text(output)
    events, states, counts = [], [], {}
    for line in output.splitlines():
        parts = line.split()
        if parts[0] == "E":
            _, step, stream, size, payload = parts
            data = bytes.fromhex(payload)
            if len(data) != int(size):
                raise AssertionError("Replay emitted an inconsistent payload length")
            events.append((int(step), stream, data))
        elif parts[0] == "S":
            states.append(line)
        elif parts[0] == "H":
            counts[int(parts[1])] = tuple(map(int, parts[2:]))
        else:
            raise AssertionError(f"Unexpected replay output: {line}")
    return events, states, counts


class BlockIOTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        WORK.mkdir(parents=True, exist_ok=True)
        for name, flags in (("original", []), ("replay_original", ["-DPOLICY_REPLAY"])):
            run(OPTIONS.clang, "-target", "bpf", "-g", "-O1", "-Xclang",
                "-disable-llvm-passes", f"-I/usr/include/{OPTIONS.multiarch}", *flags,
                "-emit-llvm", "-c", SOURCE, "-o", WORK / f"{name}.bc")
        cls.original = disassemble(WORK / "original.bc", "original")
        cls.reference = replay(WORK / "replay_original.bc", "original")
        cls.selected = {}
        cls.report = {"original": cls.original, "profiles": {}}
        for name, (stream, fields, _) in PROFILES.items():
            selected, schema = extract(WORK / "original.bc", name, stream, fields)
            machine = disassemble(selected, name)
            replay_bc, replay_schema = extract(
                WORK / "replay_original.bc", f"replay_{name}", stream, fields)
            if schema != replay_schema:
                raise AssertionError("Production and replay schemas differ")
            cls.selected[name] = (schema, machine, replay(replay_bc, name))
            cls.report["profiles"][name] = {"schema": schema, "code": machine}

    @classmethod
    def tearDownClass(cls):
        (WORK / "report.json").write_text(json.dumps(cls.report, indent=2) + "\n")
        for name, (stream, _, size) in PROFILES.items():
            before = cls.original[stream]
            after = cls.selected[name][1][stream]
            print(f"{name}: payload {48 if stream == 'starts' else 56}->{size}, "
                  f"instructions {before['instructions']}->{after['instructions']}, "
                  f"kernel reads {before['helpers'].count(113)}->{after['helpers'].count(113)}")

    def test_reference_values_and_failure_paths(self):
        events, states, counts = self.reference
        self.assertEqual(len(states), 25)
        self.assertEqual(len(counts), 25)
        self.assertEqual(len(events), 22)  # Two failed submissions and one suppressed issue.
        for step, stream, data in events:
            self.assertEqual(len(data), 48 if stream == "starts" else 56)
            if stream == "ends":
                self.assertEqual(struct.unpack_from("<QQ", data, 32), (8000, 7000))
            else:
                device = struct.unpack_from("<I", data)[0]
                state = states[step - 1].split(" T ")[0]
                value = re.search(rf"\b{device}:(-?\d+):(-?\d+)\b", state)
                self.assertIsNotNone(value)
                self.assertEqual(struct.unpack_from("<ii", data, 40),
                                 tuple(map(int, value.groups())))

    def test_selected_bytes_and_companion_outputs_match(self):
        reference, states, _ = self.reference
        for name, (stream, fields, size) in PROFILES.items():
            with self.subTest(profile=name):
                schema, _, (events, actual_states, _) = self.selected[name]
                self.assertEqual(schema["size"], size)
                self.assertEqual(schema["original_size"], 48 if stream == "starts" else 56)
                self.assertEqual([f["name"] for f in schema["fields"]], fields.split(","))
                self.assertEqual(schema["preserved_programs"],
                                 [PROGRAMS["starts" if stream == "ends" else "ends"]])
                self.assertEqual(actual_states, states)
                self.assertEqual(len(events), len(reference))
                for actual, original in zip(events, reference):
                    self.assertEqual(actual[:2], original[:2])
                    if actual[1] != stream:
                        self.assertEqual(actual[2], original[2])
                        continue
                    expected = bytearray(size)
                    for field in schema["fields"]:
                        old_offset, width = FULL_FIELDS[stream][field["name"]]
                        self.assertEqual(field["size"], width)
                        offset = field["offset"]
                        expected[offset:offset + width] = original[2][old_offset:old_offset + width]
                    self.assertEqual(actual[2], expected)

    def test_unused_reads_removed_but_queue_state_preserved(self):
        for name, (stream, _, _) in PROFILES.items():
            with self.subTest(profile=name):
                _, code, (_, _, counters) = self.selected[name]
                before, after = self.original[stream], code[stream]
                self.assertLess(after["instructions"], before["instructions"])
                self.assertLess(after["helpers"].count(113), before["helpers"].count(113))
                self.assertIn(1, after["helpers"])  # Queue/membership lookup remains.
                self.assertIn(3 if stream == "ends" else 2, after["helpers"])
                if name == "queue":
                    self.assertNotIn(5, after["helpers"])
                    for step, actual in counters.items():
                        if step in {2, 3, 4, 10, 12, 14, 17, 19, 21, 23}:
                            self.assertEqual(actual[0], 0)
                else:
                    self.assertIn(5, after["helpers"])

    def test_paired_programs_require_explicit_preservation(self):
        result = subprocess.run([
            str(EXTRACTOR), "--input", str(WORK / "original.bc"),
            "--output", str(WORK / "rejected.bc"), "--schema", str(WORK / "rejected.json"),
            "--program", "block_rq_complete", "--event", EVENTS["ends"],
            "--fields", PROFILES["latency"][1]], capture_output=True, text=True, timeout=120)
        self.assertEqual(result.returncode, 1, result.stderr)
        self.assertIn("--preserve-other-programs", result.stderr)


if __name__ == "__main__":
    unittest.main(argv=[sys.argv[0], *TEST_ARGS])
