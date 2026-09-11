"""Abstract definition of a BPF program."""

import re
import subprocess
from pathlib import Path

from bcc import BPF
from data_schema import CollectionTable
from typing_extensions import Final, Protocol

POLL_TIMEOUT_MS: Final[int] = 5


def compile_bpf(text: str, *, cflags: tuple[str, ...] = ()) -> BPF:
    """Expand Kermit macros before BCC rewrites map operations."""
    bpf_dir = Path(__file__).parent / "bpf"
    directives: list[str] = []

    # Leave kernel includes and conditionals for BCC's kernel-aware preprocessor.
    def defer_directive(match: re.Match[str]) -> str:
        directives.append(match.group().rstrip("\n"))
        return f"__KERMIT_DIRECTIVE_{len(directives) - 1}__\n"

    text = re.sub(r"(?m)^[ \t]*#(?:[^\n]*\\\n)*[^\n]*(?:\n|$)", defer_directive, text)
    text = subprocess.run(
        ["cpp", "-P", "-undef", "-imacros", str(bpf_dir / "kermit.h"), *cflags, "-"],
        input=text,
        text=True,
        stdout=subprocess.PIPE,
        check=True,
    ).stdout
    text = re.sub(r"__KERMIT_DIRECTIVE_(\d+)__", lambda m: directives[int(m[1])], text)
    return BPF(text=text.encode(), cflags=[f"-I{bpf_dir}", *cflags])


def capture_bpf(text: str) -> BPF:
    """Compile instrumentation for Kermit's capture runner."""
    return compile_bpf(text, cflags=("-DKERMIT_CAPTURE",))


class BPFProgram(Protocol):
    """Loadable BPF program that returns performance data."""

    @classmethod
    def name(cls) -> str: ...

    def load(self, collection_id: str) -> None: ...

    def poll(self) -> None: ...

    def close(self) -> None: ...

    def data(self) -> list[CollectionTable]: ...

    # def last_k_ms(self, ms: int) -> list[CollectionTable]: ...

    # def last_k_data(self, k: int) -> list[CollectionTable]:
    #   return [
    #     table.limit(k)
    #     for table in self.data()
    #   ]

    # def last_data(self) -> list[CollectionTable]:
    #   return self.last_data(1)

    def clear(self): ...

    def pop_data(self) -> list[CollectionTable]: ...
