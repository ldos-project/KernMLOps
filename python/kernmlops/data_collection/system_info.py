import platform
import subprocess
import time
import uuid
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path

import polars as pl
import psutil


@dataclass(frozen=True)
class MachineIDInfo:
    collection_id: str
    hostname: str
    start_time_sec: float
    uptime_sec: float


@dataclass(frozen=True)
class MachineSoftwareConfiguration:
    os: str
    kernel_version: str
    swap_size_bytes: int
    transparent_hugepages: str
    huge_pages: int
    huge_page_size_bytes: int
    quanta_length: int # TODO(Patrick): add this
    is_vm: bool = False


@dataclass(frozen=True)
class MachineHardwareCacheConfiguration:
    l1_instruction_cache_bytes: int
    l1_cache_bytes: int
    l2_cache_bytes: int
    l3_cache_bytes: int
    cache_alignment: int
    tlb_size_pages: int


@dataclass(frozen=True)
class DiskPartitions:
    name: str
    mount_point: str
    filesystem: str
    total_size_bytes: int
    used_bytes: int
    free_bytes: int


@dataclass(frozen=True)
class DiskConfiguration:
    interface: str
    disk_size_bytes: int
    block_size_bytes: int
    disk_speed_mhz: int  # TODO(Patrick): calculate and include this later
    total_read_bytes: int
    total_write_bytes: int
    partitions: list[DiskPartitions]


@dataclass(frozen=True)
class NICConfiguration:
    name: str
    manufacturer: str
    type: int
    mtu: int
    link_speed_mbps: int
    flags: int
    total_sent_packets: int
    total_recv_packets: int
    total_sent_bytes: int
    total_recv_bytes: int


@dataclass(frozen=True)
class MachineIOHardwareConfiguration:
    disks: list[DiskConfiguration]
    nics: list[NICConfiguration]


@dataclass(frozen=True)
class MachineHardwareConfiguration:
    architecture: str
    manufacturer: str
    sockets: int
    cores: int
    logical_cores: int
    cpu_freq_boost_enabled: bool
    max_cpu_speed_mhz: int
    min_cpu_speed_mhz: int
    ram_bytes: int
    ram_speed_mhz: int  # TODO(Patrick): include this later (sudo lshw -C memory)


@dataclass(frozen=True)
class MachineInfo:
    identification: MachineIDInfo
    software: MachineSoftwareConfiguration
    hardware: MachineHardwareConfiguration
    cache: MachineHardwareCacheConfiguration
    # io_hardware: MachineIOHardwareConfiguration

    def to_polars(self) -> pl.DataFrame:
        return pl.from_dict(asdict(self))


def machine_info() -> MachineInfo:
    identification = machine_id_info()
    software = machine_software_config()
    hardware = machine_hardware_config()
    cache = machine_hardware_cache_config()
    return MachineInfo(
        identification=identification,
        software=software,
        hardware=hardware,
        cache=cache,
        # io_hardware=io_hardware,
    )


def machine_id_info() -> MachineIDInfo:
    uname = platform.uname()
    collection_id = str(uuid.uuid4())
    hostname = uname.node
    start_time_sec = psutil.boot_time()
    uptime_sec = time.time() - start_time_sec
    return MachineIDInfo(
        collection_id=collection_id,
        hostname=hostname,
        start_time_sec=start_time_sec,
        uptime_sec=uptime_sec,
    )


@lru_cache
def raw_lscpu_output() -> list[str]:
    return subprocess.check_output(["lscpu"]).strip().decode().splitlines()


def _proc_memory_info() -> Mapping[str, str]:
    mem_info = dict[str, str]()
    with open("/proc/meminfo", "r") as proc_mem_info:
        lines = proc_mem_info.readlines()
        for line in lines:
            key, value = line.split(maxsplit=1)
            key = key.removesuffix(":").lstrip().rstrip()
            value = value.lstrip().rstrip()
            mem_info[key] = value
    return mem_info


@lru_cache
def proc_cpu_info() -> list[Mapping[str, str]]:
    processors_data = list[Mapping[str, str]]()
    with open("/proc/cpuinfo", "r") as proc_cpu_info:
        raw_data = proc_cpu_info.read()
        for cpu_data in raw_data.split("\n\n"):
            processor_data = dict[str, str]()
            lines = cpu_data.splitlines()
            for line in lines:
                key, value = line.split(":", maxsplit=1)
                key = key.lstrip().rstrip()
                value = value.lstrip().rstrip()
                processor_data[key] = value
            processors_data.append(processor_data)
    return processors_data


def transparent_hugepages() -> str:
    raw_enabled = Path("/sys/kernel/mm/transparent_hugepage/enabled").read_text()
    if "[always]" in raw_enabled:
        return "always"
    if "[madvise]" in raw_enabled:
        return "madvise"
    if "[never]" in raw_enabled:
        return "never"
    print("warning: could not parse transparent hugepage setting")
    return ""


def convert_to_bytes(value: int, unit: str) -> int:
    if unit.lower() in ["kib"]:
        return value * 1024
    if unit.lower() in ["mib"]:
        return value * 1024 * 1024
    if unit.lower() in ["gib"]:
        return value * 1024 * 1024 * 1024
    return value


def machine_software_config() -> MachineSoftwareConfiguration:
    uname = platform.uname()
    swap = psutil.swap_memory()
    mem_info = _proc_memory_info()
    os = uname.system
    kernel_version = uname.release
    swap_size_bytes = swap.total
    huge_pages = int(mem_info.get("HugePages_Total", 0))
    huge_page_size_bytes = int(mem_info.get("Hugepagesize", "0 kB").split()[0]) * 1024
    thp = transparent_hugepages()

    return MachineSoftwareConfiguration(
        os=os,
        kernel_version=kernel_version,
        swap_size_bytes=swap_size_bytes,
        quanta_length=0,
        huge_pages=huge_pages,
        huge_page_size_bytes=huge_page_size_bytes,
        transparent_hugepages=thp,
        is_vm=False,
    )


def machine_hardware_config() -> MachineHardwareConfiguration:
    uname = platform.uname()
    cores = psutil.cpu_count(logical=False)
    lscpu_output = raw_lscpu_output()
    architecture = uname.processor
    logical_cores = psutil.cpu_count(logical=True)
    _, min_cpu_speed_mhz, max_cpu_speed_mhz = psutil.cpu_freq()
    ram_bytes = psutil.virtual_memory().total

    manufacturer = ""
    sockets = 1
    cpu_freq_boost_enabled = False
    for line in lscpu_output:
        if "Vendor ID:" in line:
            manufacturer = line.split(":", 1)[1].lstrip().rstrip()
        if "NUMA node(s):" in line:
            sockets = int(line.split(":", 1)[1].lstrip().rstrip())
        if "Frequency boost:" in line:
            cpu_freq_boost_enabled = (
                line.split(":", 1)[1].lstrip().rstrip().lower() == "enabled"
            )

    return MachineHardwareConfiguration(
        architecture=architecture,
        manufacturer=manufacturer,
        sockets=sockets,
        cores=cores,
        logical_cores=logical_cores,
        cpu_freq_boost_enabled=cpu_freq_boost_enabled,
        max_cpu_speed_mhz=int(max_cpu_speed_mhz),
        min_cpu_speed_mhz=int(min_cpu_speed_mhz),
        ram_bytes=ram_bytes,
        ram_speed_mhz=0,
    )


def _convert_cache_size_to_bytes(raw_line: str) -> int:
    cache_size_data = raw_line.split(":", 1)[1].lstrip().rstrip().split()
    value = int(float(cache_size_data[0])) if cache_size_data else 0
    unit = cache_size_data[1] if len(cache_size_data) > 1 else ""
    return convert_to_bytes(value=value, unit=unit)


def machine_hardware_cache_config() -> MachineHardwareCacheConfiguration:
    lscpu_output = raw_lscpu_output()
    raw_proc_cpu_info = proc_cpu_info()
    l1_instruction_cache_bytes = 0
    l1_cache_bytes = 0
    l2_cache_bytes = 0
    l3_cache_bytes = 0
    for line in lscpu_output:
        if "L1i cache:" in line:
            l1_instruction_cache_bytes = _convert_cache_size_to_bytes(line)
        if "L1d cache:" in line:
            l1_cache_bytes = _convert_cache_size_to_bytes(line)
        if "L2 cache:" in line:
            l2_cache_bytes = _convert_cache_size_to_bytes(line)
        if "L3 cache:" in line:
            l3_cache_bytes = _convert_cache_size_to_bytes(line)
    cache_alignment = int(raw_proc_cpu_info[0].get("cache_alignment", 0))
    tlb_size_pages = int(raw_proc_cpu_info[0].get("TLB size", "0 4K pages").split()[0])
    return MachineHardwareCacheConfiguration(
        l1_instruction_cache_bytes=l1_instruction_cache_bytes,
        l1_cache_bytes=l1_cache_bytes,
        l2_cache_bytes=l2_cache_bytes,
        l3_cache_bytes=l3_cache_bytes,
        cache_alignment=cache_alignment,
        tlb_size_pages=tlb_size_pages,
    )
