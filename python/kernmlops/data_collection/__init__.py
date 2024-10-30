from dataclasses import dataclass, field, make_dataclass
from pathlib import Path

from data_collection import bpf_instrumentation as bpf
from data_collection.system_info import machine_info
from kernmlops_config import ConfigBase


@dataclass(frozen=True)
class GenericCollectorConfig(ConfigBase):
    poll_rate: float = 0.5
    output_dir: str = "data"
    output_graphs: bool = False
    hooks: list[str] = field(default_factory=bpf.hook_names)

    def get_output_dir(self) -> Path:
        return Path(self.output_dir)

    def get_hooks(self) -> list[bpf.BPFProgram]:
        return [
            hook()
            for hook_name, hook in bpf.all_hooks.items()
            if hook_name in self.hooks
        ]


CollectorConfig = make_dataclass(
    cls_name="CollectorConfig",
    bases=(ConfigBase,),
    fields=[
        (
            "generic",
            GenericCollectorConfig,
            field(default=GenericCollectorConfig()),
        )
    ],
    frozen=True,
)


__all__ = [
    "bpf",
    "machine_info",
    "CollectorConfig",
    "GenericCollectorConfig",
]
