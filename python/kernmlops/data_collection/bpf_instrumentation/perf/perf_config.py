import os
import subprocess
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from typing import Final, Mapping

from data_schema.perf import CustomHWEventID, PerfCollectionTable

# https://stackoverflow.com/questions/14626395/how-to-properly-convert-a-c-ioctl-call-to-a-python-fcntl-ioctl-call
# https://github.com/torvalds/linux/blob/0a9b9d17f3a781dea03baca01c835deaa07f7cc3/include/uapi/linux/perf_event.h#L551
PERF_EVENT_IOC_ENABLE: Final[int] = ord('$') << (4*2) | 0
PERF_EVENT_IOC_DISABLE: Final[int] = ord('$') << (4*2) | 1
PERF_IOC_FLAG_GROUP: Final[int] = 1


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
  def get_hw_event(cls, event: type[PerfCollectionTable]) -> CustomHWConfig | None:
    for hw_id in event.hw_ids():
      hw_event_config = cls.hw_event_map().get(hw_id.name.upper())
      if hw_event_config is not None:
        hw_config_value = hw_event_config.config(hw_id)
        if hw_config_value is not None:
          return hw_event_config
    return None

  @classmethod
  def get_hw_config(cls, event: type[PerfCollectionTable]) -> int | None:
    for hw_id in event.hw_ids():
      hw_event_config = cls.hw_event_map().get(hw_id.name.upper())
      if hw_event_config is not None:
        hw_config_value = hw_event_config.config(hw_id)
        if hw_config_value is not None:
          return hw_config_value
    return None
