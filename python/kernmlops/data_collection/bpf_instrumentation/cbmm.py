from dataclasses import dataclass
from pathlib import Path

import polars as pl
from bcc import BPF
from data_collection.bpf_instrumentation.bpf_hook import POLL_TIMEOUT_MS, BPFProgram
from data_schema import CollectionTable
from data_schema.cbmm import (
    CBMMEagerDataTable,
    CBMMPrezeroingDataTable,
)


@dataclass(frozen=True)
class CBMMEagerTracingRuntimeData:
    freq_cycles: int
    greatest_range_benefit: int
    decision: bool

@dataclass(frozen=True)
class CBMMPrezeroingTracingRuntimeData:
    load: int
    daemon_cost: int
    prezero_n: int
    nfree: int
    critical_section_cost: int
    zeroing_per_page_cost: int
    recent_used: int
    decision: bool


class CBMMBPFHook(BPFProgram):

    @classmethod
    def name(cls) -> str:
        return "cbmm"

    def __init__(self):
        self.is_support_raw_tp = True #  BPF.support_raw_tracepoint()
        self.bpf_text = open(Path(__file__).parent / "bpf/cbmm.bpf.c", "r").read()
        self.cbmm_eager = list[CBMMEagerTracingRuntimeData]()
        self.cbmm_prezero = list[CBMMPrezeroingTracingRuntimeData]()

    def load(self, collection_id: str):
        self.collection_id = collection_id
        self.bpf = BPF(text = self.bpf_text)
        self.bpf.attach_kprobe(event=b"mm_estimate_changes", fn_name=b"kprobe__mm_estimate_changes")
        self.bpf.attach_kretprobe(event=b"mm_decide", fn_name=b"kretprobe__mm_decide")
        self.bpf.attach_kprobe(event=b"mm_estimate_eager_page_cost_benefit", fn_name=b"kprobe__mm_estimate_eager_page_cost_benefit")
        self.bpf.attach_kretprobe(event=b"mm_estimate_eager_page_cost_benefit", fn_name=b"kretprobe__mm_estimate_eager_page_cost_benefit")
        self.bpf.attach_kprobe(event=b"mm_estimate_daemon_cost", fn_name=b"kprobe__mm_estimate_daemon_cost")
        self.bpf.attach_kprobe(event=b"get_avenrun", fn_name=b"kprobe__get_avenrun")
        self.bpf.attach_kretprobe(event=b"get_avenrun", fn_name=b"kretprobe__get_avenrun")
        #self.bpf.attach_kprobe(event=b"mm_estimate_async_prezeroing_lock_contention_cost",
        #   fn_name=b"kprobe__mm_estimate_async_prezeroing_lock_contention_cost")
        self.bpf.attach_kretprobe(event=b"mm_estimated_prezeroed_used", fn_name=b"kretprobe__mm_estimated_prezeroed_used")
        self.bpf["cbmm_eager"].open_perf_buffer(self._cbmm_eager_eh, page_cnt=64)
        self.bpf["cbmm_prezero"].open_perf_buffer(self._cbmm_prezero_eh, page_cnt=64)

    def poll(self):
        self.bpf.perf_buffer_poll(timeout=POLL_TIMEOUT_MS)

    def close(self):
        self.bpf.cleanup()

    def data(self) -> list[CollectionTable]:
        return [
            CBMMPrezeroingDataTable.from_df_id(
                pl.DataFrame(self.cbmm_prezero),
                collection_id=self.collection_id,
            ),
            CBMMEagerDataTable.from_df_id(
                pl.DataFrame(self.cbmm_eager),
                collection_id=self.collection_id,
            ),
        ]

    def pop_data(self) -> list[CollectionTable]:
        tables = self.data()
        self.clear()
        return tables

    def clear(self):
        self.cbmm_eager.clear()
        self.cbmm_prezero.clear()

    def _cbmm_eager_eh(self, cpu, cbmm_eager_paging_inputs, size):
        event = self.bpf["cbmm_eager"].event(cbmm_eager_paging_inputs)
        self.cbmm_eager.append(
            CBMMEagerTracingRuntimeData(
                freq_cycles=event.freq_cycles,
                greatest_range_benefit=event.greatest_range_benefit,
                decision=bool(event.decision),
            )
        )

    def _cbmm_prezero_eh(self, cpu, cbmm_async_prezeroing_inputs, size):
        event = self.bpf["cbmm_prezero"].event(cbmm_async_prezeroing_inputs)
        x = CBMMPrezeroingTracingRuntimeData(
                load=event.load,
                daemon_cost=event.daemon_cost,
                prezero_n=event.prezero_n,
                nfree=event.nfree,
                critical_section_cost=event.critical_section_cost,
                zeroing_per_page_cost=event.zeroing_per_page_cost,
                recent_used=event.recent_used,
                decision=bool(event.decision),
            )
        self.cbmm_prezero.append(x
        )
