from pathlib import Path

import polars as pl


def read_parquet_dir(data_dir: Path | str, *, benchmark_name: str | None = None) -> dict[str, pl.DataFrame]:
    if isinstance(data_dir, str):
        data_dir = Path(data_dir)
    kernmlops_dfs = dict[str, pl.DataFrame]()
    dataframe_dirs = [x for x in data_dir.iterdir() if x.is_dir()]
    for dataframe_dir in dataframe_dirs:
        dfs = [
          pl.read_parquet(x) for x in dataframe_dir.iterdir()
          if x.is_file() and x.suffix == ".parquet" and
          (benchmark_name is None or x.suffixes[-2] == f".{benchmark_name}")
        ]
        kernmlops_dfs[dataframe_dir.name] = pl.concat(dfs, how="diagonal_relaxed")
    return kernmlops_dfs


__all__ = [
    "read_parquet_dir",
]
