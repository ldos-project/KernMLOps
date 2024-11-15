from dataclasses import dataclass
from fcntl import ioctl
from pathlib import Path
from typing import Any, Final

import polars as pl
from bcc import BPF, PerfType
from data_collection.bpf_instrumentation.bpf_hook import POLL_TIMEOUT_MS, BPFProgram
from data_collection.bpf_instrumentation.perf.perf_config import (
  PERF_EVENT_IOC_DISABLE,
  PERF_EVENT_IOC_ENABLE,
  PERF_IOC_FLAG_GROUP,
  CustomHWConfigManager,
)
from data_schema import CollectionTable
from data_schema.perf import PerfCollectionTable, perf_table_types

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


class PerfBPFHook(BPFProgram):

  @classmethod
  def name(cls) -> str:
    return "perf"

  def __init__(self):
    self._perf_data = dict[str, list[PerfData]]()
    self.bpf_text = open(Path(__file__).parent / "../bpf/perf.bpf.c", "r").read()
    self.loaded_hw_event_configs = dict[type[PerfCollectionTable], int]()
    self.group_fds: dict[int, int] | None = None

    # add perf handlers
    def init_perf_handler(perf_event: type[PerfCollectionTable]):
      if perf_event.ev_type() == PerfType.RAW:
        hw_config_value = CustomHWConfigManager.get_hw_config(perf_event)
        if hw_config_value is not None:
          self.bpf_text += PERF_HANDLER.replace("NAME", perf_event.name())
          self._perf_data[perf_event.name()] = list[PerfData]()
          self.loaded_hw_event_configs[perf_event] = hw_config_value
        else:
          print(f"info: could not enable perf counter for {perf_event.name()}")
      else:
        self.bpf_text += PERF_HANDLER.replace("NAME", perf_event.name())
        self._perf_data[perf_event.name()] = list[PerfData]()
        self.loaded_hw_event_configs[perf_event] = perf_event.ev_config()

    for perf_event in list(perf_table_types.values()):
      init_perf_handler(perf_event)

  def _attach_perf_event(
      self,
      ev_type: int,
      ev_config: int,
      fn_name: bytes,
      sample_freq: int,
  ) -> None:
    if self.group_fds is None:
      self.bpf.attach_perf_event(
        ev_type=ev_type,
        ev_config=ev_config,
        fn_name=fn_name,
        sample_freq=sample_freq,
        cpu=-1,
        group_fd=-1,
      )
      self.group_fds = self.bpf.open_perf_events[(ev_type, ev_config)]
    else:
      for cpu, group_fd in self.group_fds.items():
        self.bpf.attach_perf_event(
          ev_type=ev_type,
          ev_config=ev_config,
          fn_name=fn_name,
          sample_freq=sample_freq,
          cpu=cpu,
          group_fd=group_fd,
        )


  def load(self, collection_id: str):
    self.collection_id = collection_id
    self.bpf = BPF(text = self.bpf_text)
    # sample frequency is in hertz
    for event, hw_config in self.loaded_hw_event_configs.items():
      self._attach_perf_event(
        ev_type=event.ev_type(),
        ev_config=hw_config,
        fn_name=bytes(f"{str(event.name())}_on", encoding="utf-8"),
        sample_freq=1000,
      )
    for event_name in self._perf_data.keys():
      self.bpf[event_name].open_perf_buffer(self._perf_handler(event_name), page_cnt=64)

  def disable_counters(self) -> None:
    if self.group_fds is None:
      return
    for _, group_fd in self.group_fds.items():
      ioctl(group_fd, PERF_EVENT_IOC_DISABLE, PERF_IOC_FLAG_GROUP)

  def enable_counters(self) -> None:
    if self.group_fds is None:
      return
    for _, group_fd in self.group_fds.items():
      ioctl(group_fd, PERF_EVENT_IOC_ENABLE, PERF_IOC_FLAG_GROUP)

  def poll(self):
    self.bpf.perf_buffer_poll(timeout=POLL_TIMEOUT_MS)

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
    miss_tables = self.data()
    self.clear()
    return miss_tables

  def _perf_handler(self, event_name: str):
    def _perf_event_handler(cpu, perf_event_data, size):
      event = self.bpf[event_name].event(perf_event_data)
      try:
          self._perf_data[event_name].append(PerfData.from_event(cpu, event))
      except Exception as _:
        pass
    return _perf_event_handler
