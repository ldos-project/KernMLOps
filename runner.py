from python.kernmlops.analysis.collector import Collector

import random
import argparse
from enum import Enum
from pathlib import Path
import random
from uuid import uuid4
import sys

class HugeSetting(Enum):
    NEVER = "never"
    ALWAYS = "always"

class YCSBType(Enum):
    REDIS = "redis"
    MEMCACHED = "memcached"
    MONGODB = "mongodb"

def create_ycsb_yaml(output_dir: Path, ycsb_type: YCSBType, huge: HugeSetting,
                     record_count: int, op_count: int,
                     read: int, update: int, rmw: int, scan:int,
                     delete: int, repeat: int, uuid_assign: str) -> Path | None:
    hun = 100.0
    purge = ""
    if ycsb_type == YCSBType.REDIS:
        purge = "explicit_purge: true"
    file_data = f"""
---
benchmark_config:
  generic:
    benchmark: {ycsb_type.value}
    cpus: 0
    skip_clear_page_cache: false
    transparent_hugepages: {huge.value}
    overcommit_memory: never_check
  {ycsb_type.value}:
    repeat: {repeat}
    field_count: 1
    min_field_length: 4096
    field_length: 2097152
    thread_count: 48
    record_count: {record_count}
    operation_count: {op_count}
    read_proportion: {read/hun}
    update_proportion: {update/hun}
    insert_proportion: 0
    rmw_proportion: {rmw/hun}
    scan_proportion: {scan/hun}
    delete_proportion: {delete/hun}
    field_length_distribution: "uniform"
    {purge}
collector_config:
  generic:
    poll_rate: 0.1
    output_dir: data
    output_graphs: false
    hooks:
      - mm_rss_stat
      - process_trace
      - madvise
      - perf
      - unmap_range
      - collapse_huge_pages
"""

    file_name: Path = Path(f"{ycsb_type.value}-{huge.value}-{record_count}-{op_count}-{repeat}--{read}-{update}-{rmw}-{scan}-{delete}--{uuid_assign}.yaml")
    output_file: Path = output_dir / file_name
    if output_file.is_file():
        return None
    with open(output_file, "w") as f:
        f.write(file_data)
    return output_file

parser = argparse.ArgumentParser(description="Runner script")

parser.add_argument("--log", type=Path, required=True, help="Info for logs")
parser.add_argument("--output_dir", type=Path, required=True, help="Info for outputs")
parser.add_argument("--number", type=int, default=10, help="Number of times")

args = parser.parse_args()

ycsb = [YCSBType.REDIS, YCSBType.MEMCACHED, YCSBType.MONGODB]
huge = [HugeSetting.NEVER, HugeSetting.ALWAYS]

print("Output Dir", args.output_dir)
yamls = []
for i in range(args.number):
    remaining_prop = 100
    remaining_size_log = 37 - 20

    remaining_size = 1 << remaining_size_log
    record_count = random.randint(1, remaining_size)
    op_count = random.randint(1, max(1, remaining_size//record_count))
    repeat = random.randint(1, max(1, remaining_size//record_count//op_count))

    read = random.randint(0, remaining_prop)
    remaining_prop -= read
    update = random.randint(0, remaining_prop)
    remaining_prop -= update
    rmw = random.randint(0, remaining_prop)
    remaining_prop -= rmw
    scan = random.randint(0, remaining_prop)
    remaining_prop -= scan
    delete = remaining_prop

    ycsb_index = random.randint(0, 2)
    uuid_assign = i
    to_add = create_ycsb_yaml(args.output_dir, ycsb[ycsb_index], huge[0],
                              record_count, op_count,
                              read, update, rmw, scan,
                              delete, repeat, uuid_assign)
    yamls.append(to_add)
    to_add = create_ycsb_yaml(args.output_dir, ycsb[ycsb_index], huge[1],
                              record_count, op_count,
                              read, update, rmw, scan,
                              delete, repeat, uuid_assign)
    yamls.append(to_add)

file = open(args.log, "w")
sys.stdout = file

for yaml in yamls:
    collect = Collector(yaml)
    collect.start_collection()
    collect.wait()
