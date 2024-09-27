import os
from dataclasses import dataclass, fields
from functools import cache
from pathlib import Path
from typing import Any, Mapping

import osquery
import osquery.extensions
import polars as pl
from osquery.extensions.ttypes import ExtensionStatus

from data_collection.bpf_instrumentation.bpf_hook import BPFProgram


@dataclass(frozen=True)
class ProcessMetadata:
  pid: int
  name: str
  cmdline: str
  start_time: int
  parent: int
  nice: int
  cgroup_path: str

class ProcessMetadataHook(BPFProgram):

  @classmethod
  def name(cls) -> str:
    return "process_metadata"

  @classmethod
  def expected_socket(cls) -> Path:
    return Path.home() / ".osquery/osqueryd.sock"

  @classmethod
  @cache
  def _select_columns(cls) -> list[str]:
    return [field.name for field in fields(ProcessMetadata)]

  @classmethod
  @cache
  def _query_select_columns(cls) -> str:
    return ", ".join(cls._select_columns())

  def __init__(self):
    self.collector_pid = os.getpid()
    self.process_metadata = list[Mapping[str, Any]]()

  def load(self):
    if self.expected_socket().exists():
      self.osquery_instance = osquery.ExtensionClient(str(self.expected_socket()))
      self.osquery_instance.open()
      self.osquery_client = self.osquery_instance.extension_client()
    else:
      self.osquery_instance = osquery.SpawnInstance()
      self.osquery_instance.open()
      self.osquery_client = self.osquery_instance.client

    initial_processes_query = self.osquery_client.query(
      f"SELECT {self._query_select_columns()} FROM processes"
    )
    # TODO(Patrick): Add error handling
    assert isinstance(initial_processes_query.status, ExtensionStatus)
    assert initial_processes_query.status.code == 0
    assert isinstance(initial_processes_query.response, list)
    self.process_metadata = initial_processes_query.response

  def poll(self):
    new_processes_query = self.osquery_client.query(
      f"SELECT {self._query_select_columns()} FROM processes WHERE pid > {self.collector_pid}"
    )
    # TODO(Patrick): Add error handling
    assert isinstance(new_processes_query.status, ExtensionStatus)
    assert new_processes_query.status.code == 0
    assert isinstance(new_processes_query.response, list)
    self.process_metadata.extend(new_processes_query.response)

  def data(self) -> pl.DataFrame:
    return pl.DataFrame(
      self.process_metadata
    ).unique(
      "pid"
    ).cast({
      "pid": pl.Int64(),
      "start_time": pl.Int64(),
      "parent": pl.Int64(),
      "nice": pl.Int64(),
    }).rename({
      "parent": "parent_pid",
      "start_time": "start_time_unix_sec",
    })

  def clear(self):
    self.process_metadata.clear()

  def pop_data(self) -> pl.DataFrame:
    process_df = self.data()
    self.clear()
    return process_df
