from dataclasses import dataclass, field, make_dataclass
from pathlib import Path

from data_collection import bpf_instrumentation as bpf
from data_collection.bpf_instrumentation.perf.perf_hook import PerfBPFHook
from data_collection.system_info import machine_info
from data_schema.perf import perf_table_types
from kernmlops_config import ConfigBase


@dataclass(frozen=True)
class GenericCollectorConfig(ConfigBase):
    poll_rate: float = 0.5
    output_interval: str = "1m"
    output_dir: str = "data"
    output_dfs: bool = False
    output_graphs: bool = False
    hooks: list[str] = field(default_factory=bpf.hook_names)
    perf_events: list[str] | None = None
    extracted_rss_tlb: str | None = None
    original_rss_tlb: str | None = None
    original_rss_tlb_transport: str = "capture"

    def get_output_dir(self) -> Path:
        return Path(self.output_dir)

    def get_hooks(self) -> list[bpf.BPFProgram]:
        self.get_factory_metadata()
        if self.original_rss_tlb is not None:
            from data_collection.bpf_instrumentation.original_rss_tlb import (
                OriginalRSSDTLBHook,
            )

            return [
                OriginalRSSDTLBHook(
                    self.original_rss_tlb, self.original_rss_tlb_transport
                )
            ]
        if self.extracted_rss_tlb is not None:
            from data_collection.bpf_instrumentation.rss_tlb_policy import (
                ExtractedRSSDTLBHook,
            )

            return [ExtractedRSSDTLBHook(self.extracted_rss_tlb)]
        return [
            PerfBPFHook(events=[perf_table_types[name] for name in self.perf_events])
            if hook_name == "perf" and self.perf_events is not None
            else hook()
            for hook_name, hook in bpf.all_hooks.items()
            if hook_name in self.hooks
        ]

    def get_factory_metadata(self) -> dict:
        unknown = set(self.hooks) - set(bpf.all_hooks)
        if unknown:
            raise ValueError(f"unknown collector hooks: {sorted(unknown)}")
        if self.perf_events is not None:
            if "perf" not in self.hooks:
                raise ValueError("perf_events requires the perf hook")
            if not isinstance(self.perf_events, list) or any(
                not isinstance(name, str) for name in self.perf_events
            ):
                raise ValueError("perf_events must be a list of counter names")
            if len(set(self.perf_events)) != len(self.perf_events):
                raise ValueError("perf_events must contain unique counter names")
            unknown_events = set(self.perf_events) - set(perf_table_types)
            if unknown_events:
                raise ValueError(f"unknown perf counters: {sorted(unknown_events)}")
        if self.original_rss_tlb_transport not in ("capture", "mpmo"):
            raise ValueError("original RSS/DTLB transport must be capture or mpmo")
        if self.original_rss_tlb is not None:
            if self.extracted_rss_tlb is not None:
                raise ValueError(
                    "original and extracted RSS/DTLB are mutually exclusive"
                )
            if self.hooks:
                raise ValueError(
                    "original RSS/DTLB requires hooks: [] to avoid double attachment"
                )
            from data_collection.bpf_instrumentation.original_rss_tlb import (
                original_manifest,
            )

            return {
                "hooks": [f"original_rss_dtlb_{self.original_rss_tlb_transport}"],
                "manifest": original_manifest(self.original_rss_tlb),
            }
        if self.original_rss_tlb_transport == "mpmo":
            raise ValueError("original RSS/DTLB mpmo requires an original manifest")
        if self.extracted_rss_tlb is not None:
            if self.hooks:
                raise ValueError(
                    "extracted RSS/DTLB requires hooks: [] to avoid double attachment"
                )
            from data_collection.bpf_instrumentation.rss_tlb_policy import (
                policy_manifest,
            )

            return {
                "hooks": ["extracted_rss_dtlb"],
                "manifest": policy_manifest(self.extracted_rss_tlb),
            }
        metadata = {"hooks": self.hooks.copy()}
        if self.perf_events is not None:
            metadata["perf_events"] = self.perf_events.copy()
        return metadata


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
