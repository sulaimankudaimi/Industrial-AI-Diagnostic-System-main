"""Inspection CLI for the MIMII data layer.

Discovers recordings for the configured machine type/SNR, prints per-machine-ID
label counts, and optionally builds and saves file-level split manifests.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from diagnostic.config import load_data_config
from diagnostic.data.mimii import Recording, discover_recordings
from diagnostic.data.splits import make_split, save_split


def summarise(recordings: list[Recording]) -> dict[str, dict[str, int]]:
    """Count normal/abnormal recordings per machine ID.

    Args:
        recordings: Discovered recordings (any machine IDs).

    Returns:
        A mapping ``machine_id -> {"normal": int, "abnormal": int}``.
    """
    summary: dict[str, dict[str, int]] = {}
    for recording in recordings:
        counts = summary.setdefault(recording.machine_id, {"normal": 0, "abnormal": 0})
        counts[recording.label] += 1
    return summary


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser for the inspect CLI."""
    parser = argparse.ArgumentParser(
        prog="diagnostic.data.inspect",
        description="Inspect MIMII recordings and optionally build split manifests.",
    )
    parser.add_argument("--config", required=True, help="Path to a data config YAML file.")
    parser.add_argument(
        "--save-splits",
        action="store_true",
        help="Build and save one split manifest per machine ID and seed.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Entry point for ``python -m diagnostic.data.inspect``."""
    args = build_parser().parse_args(argv)
    config = load_data_config(args.config)
    recordings = discover_recordings(config.raw_dir, config.machine_type, config.snr_db)

    summary = summarise(recordings)
    for machine_id, counts in sorted(summary.items()):
        print(f"{machine_id}: normal={counts['normal']} abnormal={counts['abnormal']}")
    print(f"total: {len(recordings)} recording(s) across {len(summary)} machine ID(s)")

    if not args.save_splits:
        return 0

    failures = 0
    for machine_id in sorted(summary):
        for seed in config.seeds:
            try:
                split = make_split(recordings, machine_id, seed, config.val_fraction)
            except ValueError as exc:
                print(f"skipped {machine_id} seed {seed}: {exc}", file=sys.stderr)
                failures += 1
                continue
            path = save_split(split, config.raw_dir, config.splits_dir)
            print(
                f"{Path(path).name}: train={len(split.train)} "
                f"val={len(split.val)} test={len(split.test)}"
            )
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
