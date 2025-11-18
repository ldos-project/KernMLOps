import os
from pathlib import Path
from typing import cast

import pexpect

""" Collector class has helper methods to interact with kermit"""
class Collector:
    def __init__(self, config: Path, verbose: bool=True):
        self.env = os.environ.copy()
        self.env["INTERACTIVE"] = "it"
        self.env["CONTAINER_CMD"] = f"bash -lc 'KERNMLOPS_CONFIG_FILE={config} make collect-data'"
        self.collect_process : pexpect.spawn | None = None
        self.config: Path = config
        self.verbose = verbose

    def start_collection(self, logfile=None) -> None:
        env = cast(os._Environ[str], {**os.environ, **self.env})
        self.collect_process = pexpect.spawn("make docker", env=env, timeout=None, logfile=logfile)
        self.collect_process.expect_exact(["Started benchmark"])

    @staticmethod
    def _after_run_generate_file_data() -> dict[str, list[Path]]:
        start_path : Path = Path("./data")
        list_of_collect_id_dirs = start_path.glob("*/*/*")
        latest_collect_id = max(list_of_collect_id_dirs, key=os.path.getctime)
        list_of_files = latest_collect_id.glob("*.*.parquet")
        output = {}
        for f in list_of_files:
            index = str(f).removeprefix(str(f.parent) + "/").split(".")[0]
            if index not in output.keys():
                output[index] = []
            output[index].append(f)
        return output

    def wait(self) -> dict[str, list[Path]]:
        if self.collect_process is None:
            return {}
        self.collect_process.expect([pexpect.EOF])
        self.collect_process.wait()
        ret = Collector._after_run_generate_file_data()
        if self.verbose:
            print(self.config, "results in", ret)
        return ret

    def stop_collection(self):
        if self.collect_process is None:
            return
        self.collect_process.sendline("END")
        return self.wait()
