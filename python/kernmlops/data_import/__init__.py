from pathlib import Path

import polars as pl


def read_parquet_dir(data_dir: Path) -> dict[str, pl.DataFrame]:
    kernmlops_dfs = dict[str, pl.DataFrame]()
    dataframe_dirs = [x for x in data_dir.iterdir() if x.is_dir()]
    print(dataframe_dirs)
    for dataframe_dir in dataframe_dirs:
        dfs = [
          pl.read_parquet(x) for x in dataframe_dir.iterdir()
          if x.is_file() and x.suffix == ".parquet"
        ]
        kernmlops_dfs[dataframe_dir.name] = pl.concat(dfs)
    return kernmlops_dfs


__all__ = [
    "read_parquet_dir",
]
