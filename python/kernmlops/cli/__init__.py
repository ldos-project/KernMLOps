import multiprocessing
import sys
from pathlib import Path

import click
import data_collection
import data_import
from click_default_group import DefaultGroup
from kernmlops_benchmark import FauxBenchmark, benchmarks

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
    "-b",
    "--benchmark",
    "benchmark_name",
    default=FauxBenchmark.name(),
    type=click.Choice(list(benchmarks.keys())),
)
@click.option(
    "--cpus",
    "cpus",
    default=multiprocessing.cpu_count(),
    required=False,
    type=int,
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
    "-o",
    "--output-dir",
    "output_dir",
    default=Path("data"),
    required=True,
    type=click.Path(exists=True, file_okay=False, path_type=Path),
)
@click.option(
    "-d",
    "--benchmark-dir",
    "benchmark_dir",
    default=Path.home() / "kernmlops-benchmark",
    required=True,
    type=click.Path(exists=True, file_okay=False, path_type=Path),
)
def cli_collect_data(
    output_dir: Path,
    benchmark_name: str,
    benchmark_dir: Path,
    cpus: int | None,
    poll_rate: float,
    verbose: bool,
):
    """Run data collection tooling."""
    bpf_programs = data_collection.bpf.hooks()
    benchmark_args = {
        "benchmark_dir": benchmark_dir,
        "cpus": cpus,
    }
    benchmark = benchmarks[benchmark_name](**benchmark_args)  # pyright: ignore [reportCallIssue]
    collect.run_collect(
        data_dir=output_dir,
        benchmark=benchmark,
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
@click.option(
    "-b",
    "--benchmark",
    "benchmark_name",
    default=None,
    required=False,
    help="Benchmark to filter by, default is to dump all data",
    type=click.Choice(list(benchmarks.keys())),
)
def cli_collect_dump(input_dir: Path, benchmark_name: str | None):
    """Debug tool to dump collected data."""
    kernmlops_dfs = data_import.read_parquet_dir(input_dir, benchmark_name=benchmark_name)
    for name, kernmlops_df in kernmlops_dfs.items():
        print(f"{name}: {kernmlops_df}")



def main():
    try:
        # TODO(Patrick): use logging
        cli.main(prog_name="kernmlops")
    except Exception as e:
        print("Error: ", e)
        sys.exit(1)
