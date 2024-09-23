import sys
from pathlib import Path

import click
import data_collection
import data_import
from click_default_group import DefaultGroup

from cli import collect


@click.group()
def cli():
    """Run kernmlops operations."""


@cli.group("collect", cls=DefaultGroup, default="data", default_if_no_args=True)
def cli_collect():
    """Collect things."""


@cli_collect.command("data")
@click.option(
    "-p",
    "--poll-rate",
    "poll_rate",
    default=.5,
    required=False,
    type=float,
)
@click.option(
    "-v",
    "--verbose",
    "verbose",
    default=False,
    is_flag=True,
    type=bool,
)
@click.option(
    "-d",
    "--output-dir",
    "output_dir",
    default=Path("data/curated"),
    required=True,
    type=click.Path(exists=True, file_okay=False, path_type=Path),
)
def cli_collect_data(output_dir: Path, poll_rate: float, verbose: bool):
    """Run data collection tooling."""
    bpf_programs = data_collection.bpf.hooks()
    collect.run_collect(
        output_dir=output_dir,
        bpf_programs=bpf_programs,
        poll_rate=poll_rate,
        verbose=verbose,
    )


@cli_collect.command("dump")
@click.option(
    "-d",
    "--input-dir",
    "input_dir",
    default=Path("data/curated"),
    required=True,
    type=click.Path(exists=True, file_okay=False, path_type=Path),
)
def cli_collect_dump(input_dir: Path):
    """Debug tool to dump collected data."""
    kernmlops_dfs = data_import.read_parquet_dir(input_dir)
    for name, kernmlops_df in kernmlops_dfs.items():
        print(f"{name}: {kernmlops_df}")



def main():
    try:
        # TODO(Patrick): use logging
        cli.main(prog_name="kernmlops")
    except Exception as e:
        print("Error: ", e)
        sys.exit(1)
