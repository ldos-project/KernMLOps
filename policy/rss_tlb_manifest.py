"""Freeze generated RSS/DTLB layouts and artifact identities for the collector."""

import argparse
import hashlib
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    files = {
        "rss": "rss.selected.bpf.o",
        "dtlb": "dtlb.selected.bpf.o",
        "library": "librss_tlb.so",
    }
    manifest = {
        "version": 1,
        "sample_frequency": 1000,
        "files": {
            name: {
                "name": file,
                "sha256": hashlib.sha256(
                    (args.directory / file).read_bytes()
                ).hexdigest(),
            }
            for name, file in files.items()
        },
        "schemas": {
            name: json.loads((args.directory / f"{name}.schema.json").read_text())
            for name in ("rss", "dtlb")
        },
        "rss_definition": "RSS component pages by TGID; member 2 is swap, not resident RSS",
        "dtlb_definition": "system-wide per-CPU DTLB READ MISS / READ ACCESS with enabled/running correction",
    }
    (args.directory / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
