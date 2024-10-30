import os
import subprocess
from dataclasses import dataclass
from enum import Enum
from functools import cache
from pathlib import Path
from typing import Any, Final, Mapping, Protocol

import polars as pl
from bcc import BPF, PerfType
from data_collection.bpf_instrumentation.bpf_hook import BPFProgram
from data_schema import CollectionTable, perf_table_types


class PerfHWCacheConfig:
  # From perf_hw_cache_id in uapi/linux/perf_event.h
  class Cache(Enum):
    PERF_COUNT_HW_CACHE_L1D = 0
    PERF_COUNT_HW_CACHE_L1I = 1
    PERF_COUNT_HW_CACHE_LL = 2
    PERF_COUNT_HW_CACHE_DTLB = 3
    PERF_COUNT_HW_CACHE_ITLB = 4
    PERF_COUNT_HW_CACHE_BPU = 5
    PERF_COUNT_HW_CACHE_NODE = 6

  # From perf_hw_cache_op_id in uapi/linux/perf_event.h
  class Op(Enum):
    PERF_COUNT_HW_CACHE_OP_READ = 0
    PERF_COUNT_HW_CACHE_OP_WRITE = 1
    PERF_COUNT_HW_CACHE_OP_PREFETCH = 2

  # From perf_hw_cache_op_result_id  in uapi/linux/perf_event.h
  class Result(Enum):
    PERF_COUNT_HW_CACHE_RESULT_ACCESS = 0
    PERF_COUNT_HW_CACHE_RESULT_MISS = 1

  # From https://man7.org/linux/man-pages/man2/perf_event_open.2.html
  @classmethod
  def config(cls, cache: Cache, op: Op, result: Result) -> int:
    return (cache.value) | (op.value << 8) | (result.value << 16)


@dataclass(frozen=True)
class CustomHWEventID:
  name: str
  umask: str | None


class CustomHWEvent(Protocol):
  @classmethod
  def name(cls) -> str: ...

  @classmethod
  def hw_ids(cls) -> list[CustomHWEventID]: ...


class TLBFlushEvent(CustomHWEvent):
  @classmethod
  def name(cls) -> str:
    return "tlb_flushes"

  @classmethod
  def hw_ids(cls) -> list[CustomHWEventID]:
    return [
      CustomHWEventID(name="TLB_FLUSHES", umask="All"),
      CustomHWEventID(name="TLB_FLUSH", umask="STLB_ANY"),
    ]


@dataclass(frozen=True)
class CustomHWConfigUmask:
  id: str
  code: int
  source: str
  name: str
  # flags are actually a list of strings if present
  flags: str | None
  description: str

  def dump(self) -> str:
    return f"{self.id} : {self.code} : {self.source} : {self.name} : {self.flags} : {self.description}"

  @classmethod
  def from_evtline(cls, evt_line: str) -> "CustomHWConfigUmask | None":
    fields = [
      field.lstrip().rstrip()
      for field in evt_line.split(":")
    ]
    if len(fields) < 6:
      return None
    return CustomHWConfigUmask(
      id=fields[0],
      code=int(fields[1], base=0),
      source=fields[2],
      name=fields[3][1:-1],
      flags=fields[4],
      description=fields[5],
    )


@dataclass(frozen=True)
class CustomHWConfig:
  id: int
  pmu_name: str
  name: str
  equiv: str | None
  # flags are actually a list of strings if present
  flags: str | None
  description: str
  code: int
  code_length_hex: int
  umasks: Mapping[str, CustomHWConfigUmask]
  # modifier entries are actually objects, not strings
  modifiers: list[str]

  def config(self, id: CustomHWEventID) -> int | None:
    if id.name.upper() != self.name:
      return None
    if id.umask is None:
      return self.code
    umask = self.umasks.get(id.umask.upper())
    if umask is None:
      return None
    # TODO(Patrick): confirm this left shift for Umasks works everywhere
    return (self.code) | (umask.code << (4 * self.code_length_hex))

  def dump(self) -> str:
    return f"""
#-----------------------------
IDX : {self.id}
PMU name : {self.pmu_name}
Name : {self.name}
Equiv : {self.equiv}
Flags : {self.flags}
Desc : {self.description}
Code : {self.code}
{"\n".join([
  umask.dump()
  for umask in self.umasks.values()
])}
{"\n".join(self.modifiers)}
"""

  @classmethod
  def from_evtinfo(cls, evt_lines: list[str]) -> "CustomHWConfig | None":
    id: int | None = None
    pmu_name: str | None = None
    name: str | None = None
    equiv: str | None = None
    flags: str | None = None
    description: str | None = None
    code: int | None = None
    code_length_hex: int = 2
    umasks = dict[str, CustomHWConfigUmask]()
    modifiers = list[str]()

    for evt_line in evt_lines:
      if not evt_line:
        continue
      split_line = evt_line.split(":", maxsplit=1)
      if len(split_line) != 2:
        continue
      key = split_line[0]
      value = split_line[1]
      key = key.lstrip().rstrip()
      value = value.lstrip().rstrip()
      match key:
        case "IDX":
          id = int(value)
        case "PMU name":
          pmu_name = value
        case "Name":
          name = value.upper()
        case "Equiv":
          equiv = None if value == "None" else value
        case "Flags":
          flags = None if value == "None" else value
        case "Desc":
          description = value
        case "Code":
          code = int(value, base=0)
          code_length_hex = len(value) - 2
      if key.startswith("Umask"):
        umask = CustomHWConfigUmask.from_evtline(evt_line)
        if umask is not None:
          umasks[umask.name.upper()] = umask
        else:
          print("warning: could not parse a hardware event's umask info")
          print(evt_line)
          return None
      if key.startswith("Modif"):
        modifiers.append(evt_line)
    if id is None or pmu_name is None or name is None or description is None or code is None:
      print("warning: could not parse a hardware event's info")
      print(evt_lines)
      return None
    return CustomHWConfig(
      id=id,
      pmu_name=pmu_name,
      name=name,
      equiv=equiv,
      flags=flags,
      description=description,
      code=code,
      code_length_hex=code_length_hex,
      umasks=umasks,
      modifiers=modifiers,
    )


class CustomHWConfigManager:
  @classmethod
  @cache
  def hw_event_map(cls) -> Mapping[str, CustomHWConfig]:
    hw_events = dict[str, CustomHWConfig]()
    pfm4_dir = os.environ.get("LIB_PFM4_DIR")
    if not pfm4_dir:
      # TODO(Patrick): make this a warning
      print("warning: LIB_PFM4_DIR has not been set, disabling custom perf counters")
      return hw_events
    showevtinfo = Path(pfm4_dir) / "examples" / "showevtinfo"
    if not showevtinfo.is_file():
      print("warning: libpfm4 has not been properly built, disabling custom perf counters")
      return hw_events
    raw_event_info = subprocess.check_output(str(showevtinfo))
    if isinstance(raw_event_info, bytes):
      raw_event_info = raw_event_info.decode("utf-8")
    if not isinstance(raw_event_info, str):
      return hw_events
    raw_event_info = raw_event_info.lstrip().rstrip()
    events_info = raw_event_info.split("#-----------------------------\n")
    if events_info:
      events_info = events_info[1:]
    for event_info in events_info:
      hw_config = CustomHWConfig.from_evtinfo(event_info.splitlines())
      if hw_config is not None:
        hw_events[hw_config.name] = hw_config
    return hw_events

  @classmethod
  def get_hw_event(cls, event: CustomHWEvent) -> CustomHWConfig | None:
    for hw_id in event.hw_ids():
      hw_event_config = cls.hw_event_map().get(hw_id.name.upper())
      if hw_event_config is not None:
        hw_config_value = hw_event_config.config(hw_id)
        if hw_config_value is not None:
          return hw_event_config
    return None

  @classmethod
  def get_hw_config(cls, event: CustomHWEvent) -> int | None:
    for hw_id in event.hw_ids():
      hw_event_config = cls.hw_event_map().get(hw_id.name.upper())
      if hw_event_config is not None:
        hw_config_value = hw_event_config.config(hw_id)
        if hw_config_value is not None:
          return hw_config_value
    return None


PERF_HANDLER: Final[str] = """
BPF_PERF_OUTPUT(NAME);
int NAME_on(struct bpf_perf_event_data* ctx) {
  struct bpf_perf_event_value value_buf;
  if (bpf_perf_prog_read_value(ctx, (void*)&value_buf, sizeof(struct bpf_perf_event_value))) {
    return 0;
  }
  struct perf_event_data data;
  __builtin_memset(&data, 0, sizeof(data));
  u32 pid = bpf_get_current_pid_tgid();
  u32 tgid = bpf_get_current_pid_tgid() >> 32;
  u64 ts = bpf_ktime_get_ns();
  data.pid = pid;
  data.tgid = tgid;
  data.ts_uptime_us = ts / 1000;
  data.count = value_buf.counter;
  data.enabled_time_us = value_buf.enabled / 1000;
  data.running_time_us = value_buf.running / 1000;
  NAME.perf_submit(ctx, &data, sizeof(data));
  return 0;
}
"""

@dataclass(frozen=True)
class PerfData:
  cpu: int
  pid: int
  tgid: int
  ts_uptime_us: int
  cumulative_count: int
  pmu_enabled_time_us: int
  pmu_running_time_us: int

  @classmethod
  def from_event(cls, cpu: int, event: Any):
    return PerfData(
        cpu=cpu,
        pid=event.pid,
        tgid=event.tgid,
        ts_uptime_us=event.ts_uptime_us,
        cumulative_count=event.count,
        pmu_enabled_time_us=event.enabled_time_us,
        pmu_running_time_us=event.running_time_us,
    )


class TLBPerfBPFHook(BPFProgram):

  @classmethod
  def name(cls) -> str:
    return "tlb_perf"

  def __init__(self):
    self._perf_data = dict[str, list[PerfData]]()
    self.bpf_text = open(Path(__file__).parent / "bpf/tlb_perf.bpf.c", "r").read()
    self.loaded_custom_hw_event_configs = dict[str, int]()

    # add perf handlers
    def init_perf_handler(perf_event: CustomHWEvent | str):
      if isinstance(perf_event, str):
          self.bpf_text += PERF_HANDLER.replace("NAME", perf_event)
          self._perf_data[perf_event] = list[PerfData]()
      else:
        hw_config_value = CustomHWConfigManager.get_hw_config(perf_event)
        if hw_config_value is not None:
          self.bpf_text += PERF_HANDLER.replace("NAME", perf_event.name())
          self._perf_data[perf_event.name()] = list[PerfData]()
          self.loaded_custom_hw_event_configs[perf_event.name()] = hw_config_value
        else:
          print(f"info: could not enable perf counter for {perf_event.name()}")

    universal_perf_events = ["dtlb_misses", "itlb_misses"]
    hw_specific_perf_events = [TLBFlushEvent()]
    for perf_event in universal_perf_events:
      init_perf_handler(perf_event)
    for perf_event in hw_specific_perf_events:
      init_perf_handler(perf_event)

  def load(self, collection_id: str):
    self.collection_id = collection_id
    self.bpf = BPF(text = self.bpf_text)
    # sample frequency is in hertz
    self.bpf.attach_perf_event(
      ev_type=PerfType.HW_CACHE,
      ev_config=PerfHWCacheConfig.config(
        cache=PerfHWCacheConfig.Cache.PERF_COUNT_HW_CACHE_DTLB,
        op=PerfHWCacheConfig.Op.PERF_COUNT_HW_CACHE_OP_READ,
        result=PerfHWCacheConfig.Result.PERF_COUNT_HW_CACHE_RESULT_MISS,
      ),
      fn_name=b"dtlb_misses_on",
      sample_freq=1000,
    )
    self.bpf.attach_perf_event(
      ev_type=PerfType.HW_CACHE,
      ev_config=PerfHWCacheConfig.config(
        cache=PerfHWCacheConfig.Cache.PERF_COUNT_HW_CACHE_ITLB,
        op=PerfHWCacheConfig.Op.PERF_COUNT_HW_CACHE_OP_READ,
        result=PerfHWCacheConfig.Result.PERF_COUNT_HW_CACHE_RESULT_MISS,
      ),
      fn_name=b"itlb_misses_on",
      sample_freq=1000,
    )
    for name, custom_hw_config in self.loaded_custom_hw_event_configs.items():
      self.bpf.attach_perf_event(
        ev_type=PerfType.RAW,
        ev_config=custom_hw_config,
        fn_name=bytes(f"{str(name)}_on", encoding="utf-8"),
        sample_freq=1000,
      )
    for event_name in self._perf_data.keys():
      self.bpf[event_name].open_perf_buffer(self._perf_handler(event_name), page_cnt=64)

  def poll(self):
    self.bpf.perf_buffer_poll()

  def close(self):
    self.bpf.cleanup()

  def data(self) -> list[CollectionTable]:
    return [
      perf_table_types[event_name].from_df_id(
        pl.DataFrame(self._perf_data[event_name]),
        collection_id=self.collection_id,
      )
      for event_name in self._perf_data.keys()
      if event_name in perf_table_types
    ]

  def clear(self):
    self._perf_data.clear()

  def pop_data(self) -> list[CollectionTable]:
    tlb_miss_tables = self.data()
    self.clear()
    return tlb_miss_tables

  def _perf_handler(self, event_name: str):
    def _perf_event_handler(cpu, perf_event_data, size):
      event = self.bpf[event_name].event(perf_event_data)
      try:
          self._perf_data[event_name].append(PerfData.from_event(cpu, event))
      except Exception as _:
        pass
    return _perf_event_handler
