from dataclasses import dataclass
from pathlib import Path

import polars as pl
from bcc import BPF
from data_collection.bpf_instrumentation.bpf_hook import POLL_TIMEOUT_MS, BPFProgram
from data_schema import CollectionTable
from data_schema.generic_table import MadviseDataTable

ADVICE_ASSIGN_DICT = {
    0:"MADV_NORMAL",
    1:"MADV_RANDOM",
    2:"MADV_SEQUENTIAL",
    3:"MADV_WILLNEED",
    4:"MADV_DONTNEED",
    8:"MADV_FREE",
    9:"MADV_REMOVE",
    10:"MADV_DONTFORK",
    11:"MADV_DOFORK",
    100:"MADV_HWPOISON",
    101:"MADV_SOFT_OFFLINE",
    12:"MADV_MERGEABLE",
    13:"MADV_UNMERGEABLE",
    14:"MADV_HUGEPAGE",
    15:"MADV_NOHUGEPAGE",
    16:"MADV_DONTDUMP",
    17:"MADV_DODUMP",
    18:"MADV_WIPEONFORK",
    19:"MADV_KEEPONFORK",
    20:"MADV_COLD",
    21:"MADV_PAGEOUT",
    22:"MADV_POPULATE_READ",
    23:"MADV_POPULATE_WRITE",
    24:"MADV_DONTNEED_LOCKED",
    25:"MADV_COLLAPSE",
}

@dataclass(frozen=True)
class MadviseStat:
  tgid: int
  ts_ns: int
  address: int
  length: int
  advice: str

class MadviseBPFHook(BPFProgram):

  @classmethod
  def name(cls) -> str:
    return "madvise"

  def __init__(self):
    self.is_support_raw_tp = True #  BPF.support_raw_tracepoint()
    self.bpf_text = open(Path(__file__).parent / "bpf/madvise.bpf.c", "r").read()
    self.madvise_stat = list[MadviseStat]()

  def load(self, collection_id: str):
    self.collection_id = collection_id
    self.bpf = BPF(text = self.bpf_text)
    self.bpf.attach_kprobe(event=b"do_madvise", fn_name=b"kprobe__do_madvise")
    self.bpf.attach_kretprobe(event=b"do_madvise", fn_name=b"kretprobe__do_madvise")
    self.bpf["madvise_output"].open_perf_buffer(self._madvise_eh, page_cnt=64)

  def poll(self):
    self.bpf.perf_buffer_poll(timeout=POLL_TIMEOUT_MS)

  def close(self):
    self.bpf.cleanup()

  def data(self) -> list[CollectionTable]:
    return [
            MadviseDataTable.from_df_id(
                pl.DataFrame(self.madvise_stat),
                collection_id=self.collection_id,
            ),
        ]

  def clear(self):
    self.madvise_stat.clear()

  def pop_data(self) -> list[CollectionTable]:
    tables = self.data()
    self.clear()
    return tables

  def _madvise_eh(self, cpu, madvise_struct, size):
      event = self.bpf["madvise_output"].event(madvise_struct)
      advice = ADVICE_ASSIGN_DICT[event.advice] if event.advice in ADVICE_ASSIGN_DICT.keys() else "UNKNOWN"
      self.madvise_stat.append(
        MadviseStat(
          tgid=event.tgid,
          ts_ns=event.ts_ns,
          address=event.address,
          length=event.length,
          advice=advice,
        )
      )
