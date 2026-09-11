"""End-to-end extractor tests, invoked by the policy Makefile.

BUILD, SOURCE, CLANG, EXTRACTOR, OBJDUMP, LLVM_DIS, and MULTIARCH can also be
supplied as lower-case, hyphenated command-line options. LLVM_DIS is optional;
helper checks always inspect the compiled ELF with llvm-objdump. No BPF program
is loaded, so the suite does not require privileges. CO-RE checks also use the
llvm-objcopy alongside OBJDUMP, retaining any version suffix.
"""

import argparse
import json
import os
import re
import shlex
import struct
import subprocess
import sys
import sysconfig
import tempfile
import unittest
from pathlib import Path


POLICY = Path(__file__).resolve().parents[1]
FIXTURES = Path(__file__).resolve().parent / "fixtures"


def parse_options(argv=()):
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--build", type=Path, default=os.environ.get("BUILD", POLICY / "build")
    )
    parser.add_argument(
        "--source",
        type=Path,
        default=os.environ.get("SOURCE", POLICY / "bpf" / "syscall.bpf.c"),
    )
    parser.add_argument("--clang", default=os.environ.get("CLANG", "clang-18"))
    parser.add_argument("--extractor", default=os.environ.get("EXTRACTOR"))
    parser.add_argument(
        "--objdump", default=os.environ.get("OBJDUMP", "llvm-objdump-18")
    )
    parser.add_argument("--llvm-dis", default=os.environ.get("LLVM_DIS"))
    parser.add_argument(
        "--multiarch",
        default=os.environ.get("MULTIARCH", sysconfig.get_config_var("MULTIARCH") or ""),
    )
    options, remaining = parser.parse_known_args(argv)
    options.build = options.build.resolve()
    options.source = options.source.resolve()
    if options.extractor is None:
        options.extractor = str(options.build / "field-extract")
    return options, remaining


OPTIONS, _ = parse_options()


class ExtractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        OPTIONS.build.mkdir(parents=True, exist_ok=True)

    def run_tool(self, *args, check=True):
        command = [str(arg) for arg in args]
        result = subprocess.run(
            command, capture_output=True, text=True, timeout=120, check=False
        )
        if check:
            self.assertEqual(
                result.returncode,
                0,
                f"{shlex.join(command)}\n{result.stdout}\n{result.stderr}",
            )
        return result

    def extract(
        self, fields, *, source=None, cflags=(), expect_error=False, inspect_core=False
    ):
        with tempfile.TemporaryDirectory(
            prefix="test-extract-", dir=OPTIONS.build
        ) as directory:
            work = Path(directory)
            original = work / "input.bc"
            selected = work / "selected.bc"
            schema_path = work / "selected.json"
            elf = work / "selected.bpf.o"
            includes = (
                [f"-I/usr/include/{OPTIONS.multiarch}"] if OPTIONS.multiarch else []
            )
            self.run_tool(
                OPTIONS.clang,
                "-target", "bpf", "-g", "-O1",
                "-Xclang", "-disable-llvm-passes",
                "-emit-llvm", "-c", source or OPTIONS.source,
                *includes, *cflags, "-o", original,
            )
            result = self.run_tool(
                OPTIONS.extractor,
                "--input", original,
                "--output", selected,
                "--schema", schema_path,
                "--program", "capture_syscall",
                "--event", "policy_event",
                "--fields", fields,
                check=not expect_error,
            )
            if expect_error:
                # A signal/crash is not an intentional unsupported-input error.
                self.assertGreater(
                    result.returncode, 0, f"{result.stdout}\n{result.stderr}"
                )
                self.assertTrue(
                    (result.stdout + result.stderr).strip(),
                    "Rejected input must have a diagnostic",
                )
                return None

            schema = json.loads(schema_path.read_text())
            if OPTIONS.llvm_dis:
                self.run_tool(OPTIONS.llvm_dis, selected, "-o", work / "selected.ll")
            self.run_tool(
                OPTIONS.clang,
                "-target", "bpf", "-g", "-O2", "-c", selected, "-o", elf,
            )
            disassembly = self.run_tool(
                OPTIONS.objdump, "-dr", "--no-show-raw-insn", "--section-headers", elf
            ).stdout
            elf_header = elf.read_bytes()[:16]
            self.assertEqual(elf_header[:4], b"\x7fELF")
            for section in ("raw_tp/sys_enter", ".maps", "license", ".BTF", ".BTF.ext"):
                self.assertRegex(
                    disassembly,
                    rf"(?m)^\s*\d+\s+{re.escape(section)}\s",
                    disassembly,
                )
            if inspect_core:
                objdump = Path(OPTIONS.objdump)
                objcopy = objdump.with_name(
                    objdump.name.replace("llvm-objdump", "llvm-objcopy", 1)
                )
                core_path = work / "btf.ext"
                self.run_tool(objcopy, "--dump-section", f".BTF.ext={core_path}", elf)
                data = core_path.read_bytes()
                self.assertIn(elf_header[5], (1, 2))
                endian = "<" if elf_header[5] == 1 else ">"
                self.assertGreaterEqual(len(data), 32)
                self.assertEqual(struct.unpack_from(endian + "H", data)[0], 0xEB9F)
                hdr_len = struct.unpack_from(endian + "I", data, 4)[0]
                self.assertGreaterEqual(hdr_len, 32)
                core_off, core_len = struct.unpack_from(endian + "II", data, 24)
                cursor = hdr_len + core_off
                end = cursor + core_len
                self.assertGreaterEqual(core_len, 4)
                self.assertLessEqual(end, len(data))
                record_size = struct.unpack_from(endian + "I", data, cursor)[0]
                self.assertGreaterEqual(record_size, 16)
                cursor += 4
                core_count = 0
                # Each section table has a name offset, count, then fixed-size records.
                while cursor < end:
                    self.assertLessEqual(cursor + 8, end)
                    _, count = struct.unpack_from(endian + "II", data, cursor)
                    cursor += 8 + record_size * count
                    self.assertLessEqual(cursor, end)
                    core_count += count
                return schema, disassembly, core_count
            return schema, disassembly

    def assert_schema(self, schema, *, size, alignment, fields, original_size=32):
        expected = {
            "program": "capture_syscall",
            "source_event": "policy_event",
            "event": "policy_event_selected",
            "original_size": original_size,
            "size": size,
            "alignment": alignment,
            "byte_order": "little",
        }
        for key, value in expected.items():
            self.assertEqual(schema[key], value, key)
        self.assertEqual(
            [field["name"] for field in schema["fields"]],
            [name for name, _, _, _ in fields],
        )
        for actual, (name, offset, field_size, signed) in zip(schema["fields"], fields):
            self.assertEqual(actual["offset"], offset, name)
            self.assertEqual(actual["size"], field_size, name)
            self.assertIs(actual["signed"], signed, name)

    def assert_helpers(self, disassembly, *, present, absent=()):
        helpers = {
            int(value, 16 if value.startswith("0x") else 10)
            for value in re.findall(r"\bcall\s+(0x[0-9a-fA-F]+|[0-9]+)\b", disassembly)
        }
        self.assertTrue(set(present) <= helpers, disassembly)
        self.assertFalse(set(absent) & helpers, disassembly)

    def test_syscall_fields_compact_in_declaration_order(self):
        for requested in (
            "syscall_nr,doubled_syscall_nr",
            "doubled_syscall_nr,syscall_nr",
        ):
            with self.subTest(fields=requested):
                schema, disassembly = self.extract(requested)
                self.assert_schema(
                    schema,
                    size=8,
                    alignment=4,
                    fields=[
                        ("syscall_nr", 0, 4, False),
                        ("doubled_syscall_nr", 4, 4, False),
                    ],
                )
                self.assert_helpers(disassembly, present=(113, 130), absent=(5, 14, 15))

    def test_derived_field_keeps_kernel_read_dependency(self):
        schema, disassembly = self.extract("doubled_syscall_nr")
        self.assert_schema(
            schema,
            size=4,
            alignment=4,
            fields=[("doubled_syscall_nr", 0, 4, False)],
        )
        self.assert_helpers(disassembly, present=(113, 130), absent=(5, 14, 15))

    def test_timestamp_and_pid_drop_kernel_read_and_checksum(self):
        schema, disassembly = self.extract("timestamp_ns,pid_tgid")
        self.assert_schema(
            schema,
            size=16,
            alignment=8,
            fields=[("timestamp_ns", 0, 8, False), ("pid_tgid", 8, 8, False)],
        )
        self.assert_helpers(disassembly, present=(5, 14, 130), absent=(113, 15))
        self.assertNotRegex(disassembly, r"\bgoto\b", disassembly)

    def test_mixed_width_selection_has_tail_padding(self):
        schema, disassembly = self.extract("timestamp_ns,syscall_nr")
        self.assert_schema(
            schema,
            size=16,
            alignment=8,
            fields=[("timestamp_ns", 0, 8, False), ("syscall_nr", 8, 4, False)],
        )
        self.assert_helpers(disassembly, present=(5, 113, 130), absent=(14, 15))

    def test_mixed_width_selection_has_interior_padding(self):
        schema, disassembly = self.extract("syscall_nr,checksum")
        self.assert_schema(
            schema,
            size=16,
            alignment=8,
            fields=[("syscall_nr", 0, 4, False), ("checksum", 8, 8, False)],
        )
        self.assert_helpers(disassembly, present=(113, 15, 130))

    def test_invalid_field_lists_are_rejected(self):
        for fields in (
            "syscall_nr,no_such_field",
            "syscall_nr,doubled_syscall_nr,syscall_nr",
            "",
            ",syscall_nr",
            "syscall_nr,",
            "syscall_nr,,doubled_syscall_nr",
        ):
            with self.subTest(fields=fields):
                self.extract(fields, expect_error=True)

    def test_used_kernel_read_status_keeps_call_and_guard(self):
        schema, disassembly = self.extract(
            "timestamp_ns", source=FIXTURES / "read_status.bpf.c"
        )
        self.assert_schema(
            schema,
            original_size=16,
            size=8,
            alignment=8,
            fields=[("timestamp_ns", 0, 8, False)],
        )
        self.assert_helpers(disassembly, present=(5, 113, 130), absent=(8,))
        self.assertRegex(disassembly, r"\bif\b[^\n]*\bgoto\b", disassembly)

    def test_small_scalars_preserve_signedness(self):
        schema, disassembly = self.extract(
            "tag,cpu,syscall_nr", source=FIXTURES / "read_status.bpf.c"
        )
        self.assert_schema(
            schema,
            original_size=16,
            size=8,
            alignment=4,
            fields=[
                ("tag", 0, 1, True),
                ("cpu", 2, 2, False),
                ("syscall_nr", 4, 4, True),
            ],
        )
        self.assert_helpers(disassembly, present=(8, 113, 130), absent=(5,))

    def test_map_update_and_its_value_dependency_survive(self):
        schema, disassembly = self.extract(
            "syscall_nr", source=FIXTURES / "map_update.bpf.c"
        )
        self.assert_schema(
            schema,
            original_size=16,
            size=4,
            alignment=4,
            fields=[("syscall_nr", 0, 4, False)],
        )
        self.assert_helpers(disassembly, present=(2, 14, 130))
        self.assertRegex(disassembly, r"\bR_BPF_64_64\s+counts\b", disassembly)

    def test_array_field_is_rejected(self):
        self.extract(
            "payload", source=FIXTURES / "unsupported_array.bpf.c", expect_error=True
        )

    def test_extra_ringbuf_output_is_rejected(self):
        for cflags in ((), ("-DNOINLINE",)):
            with self.subTest(cflags=cflags):
                self.extract(
                    "syscall_nr",
                    source=FIXTURES / "extra_output.bpf.c",
                    cflags=cflags,
                    expect_error=True,
                )

    def test_core_read_preserves_only_selected_relocation(self):
        schema, disassembly, core_count = self.extract(
            "pid", source=FIXTURES / "core_read.bpf.c", inspect_core=True
        )
        self.assert_schema(
            schema,
            original_size=16,
            size=4,
            alignment=4,
            fields=[("pid", 0, 4, False)],
        )
        self.assert_helpers(disassembly, present=(113, 158, 130), absent=(5,))
        self.assertEqual(
            len(re.findall(r"\bcall\s+(?:113|0x71)\b", disassembly)), 1, disassembly
        )
        self.assertEqual(core_count, 1)


if __name__ == "__main__":
    OPTIONS, unittest_args = parse_options(sys.argv[1:])
    unittest.main(argv=[sys.argv[0], *unittest_args])
