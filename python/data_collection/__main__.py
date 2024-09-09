"""CLI"""

from system_info import machine_info

if __name__ == "__main__":
    system_info = machine_info().to_polars()
    print(system_info.unnest(system_info.columns))
