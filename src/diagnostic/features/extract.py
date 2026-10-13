"""Feature-extraction CLI: split manifest -> cached features (decisions D5, D16).

Reads the F2 split manifest for one machine ID and seed, extracts and caches the
log-mel features for the train/val/test groups, and prints per-group statistics.
It can also pack the cache for one config hash into a single ``.tar.gz``
(``--pack``) or restore such an archive (``--unpack``).
"""

from __future__ import annotations

import argparse
import json
import math
import tarfile
import time
from pathlib import Path

import numpy as np

from diagnostic.config import load_data_config, load_feature_config
from diagnostic.data.splits import load_split, manifest_path
from diagnostic.features.pipeline import FeaturePipeline, build_matrix, cached_log_mel

DEFAULT_CACHE_DIR = "data/cache/features"
DEFAULT_OUT_DIR = "runs/features"


def log_mel_stats(
    recordings: list,
    pipeline: FeaturePipeline,
    raw_dir: str | Path,
    cache_dir: str | Path,
) -> tuple[float, float]:
    """Return the mean and std of the pooled log-mel values for ``recordings``.

    Values are read from the cache (no recomputation on a hit) and accumulated in
    float64 for numerical stability.
    """
    total = 0.0
    total_sq = 0.0
    count = 0
    for recording in recordings:
        log_mel = cached_log_mel(recording.path, raw_dir, cache_dir, pipeline)
        values = log_mel.astype(np.float64, copy=False)
        total += float(values.sum())
        total_sq += float(np.square(values).sum())
        count += values.size
    if count == 0:
        return float("nan"), float("nan")
    mean = total / count
    variance = max(total_sq / count - mean * mean, 0.0)
    return mean, math.sqrt(variance)


def _safe_extract(archive: tarfile.TarFile, destination: Path) -> None:
    """Extract members of ``archive`` into ``destination``, rejecting traversal."""
    root = destination.resolve()
    for member in archive.getmembers():
        target = (root / member.name).resolve()
        if root != target and root not in target.parents:
            raise ValueError(f"refusing to extract unsafe path: {member.name}")
    archive.extractall(root)


def pack_cache(archive_path: str | Path, cache_dir: str | Path, config_hash: str) -> Path:
    """Write ``<cache_dir>/<config_hash>`` into a gzip-compressed tar archive."""
    source = Path(cache_dir) / config_hash
    if not source.is_dir():
        raise FileNotFoundError(f"no cache to pack at {source}")
    archive = Path(archive_path)
    archive.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive, "w:gz") as handle:
        handle.add(source, arcname=config_hash)
    return archive


def unpack_cache(archive_path: str | Path, cache_dir: str | Path) -> Path:
    """Restore a tar archive produced by :func:`pack_cache` into ``cache_dir``."""
    archive = Path(archive_path)
    if not archive.is_file():
        raise FileNotFoundError(f"archive not found: {archive}")
    destination = Path(cache_dir)
    destination.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive, "r:gz") as handle:
        _safe_extract(handle, destination)
    return destination


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser for the extract CLI."""
    parser = argparse.ArgumentParser(
        prog="diagnostic.features.extract",
        description="Extract and cache log-mel context features for one split manifest.",
    )
    parser.add_argument("--data-config", help="Path to a data config YAML file.")
    parser.add_argument("--features-config", help="Path to a feature config YAML file.")
    parser.add_argument("--machine-id", help="Machine ID, e.g. id_00.")
    parser.add_argument("--seed", type=int, help="Split seed to use.")
    parser.add_argument(
        "--limit", type=int, default=None, help="Use at most N files per split (loudly reported)."
    )
    parser.add_argument(
        "--cache-dir",
        default=DEFAULT_CACHE_DIR,
        help=f"Feature cache root (default {DEFAULT_CACHE_DIR}).",
    )
    parser.add_argument(
        "--out-dir",
        default=DEFAULT_OUT_DIR,
        help=f"Summary output dir (default {DEFAULT_OUT_DIR}).",
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--pack", help="After extraction, write the cache as a .tar.gz to PATH.")
    group.add_argument(
        "--unpack", help="Restore a .tar.gz cache archive from PATH into --cache-dir."
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Entry point for ``python -m diagnostic.features.extract``."""
    parser = build_parser()
    args = parser.parse_args(argv)

    cache_dir = Path(args.cache_dir)
    if args.unpack:
        unpack_cache(args.unpack, cache_dir)
        print(f"unpacked {args.unpack} into {cache_dir}")
        return 0

    if not (
        args.data_config and args.features_config and args.machine_id and args.seed is not None
    ):
        parser.error("--data-config, --features-config, --machine-id and --seed are required")

    data_config = load_data_config(args.data_config)
    feature_config = load_feature_config(args.features_config)
    pipeline = FeaturePipeline(feature_config, channel=data_config.channel)

    manifest = manifest_path(
        data_config.splits_dir,
        data_config.machine_type,
        data_config.snr_db,
        args.machine_id,
        args.seed,
    )
    if not manifest.is_file():
        raise FileNotFoundError(f"split manifest not found: {manifest}")
    split = load_split(manifest, data_config.raw_dir)

    if args.limit is not None:
        print("=" * 68)
        print(f"LIMIT ACTIVE: at most {args.limit} file(s) per split (NOT the full split).")
        print("=" * 68)

    groups = {"train": list(split.train), "val": list(split.val), "test": list(split.test)}
    if args.limit is not None:
        groups = {name: files[: args.limit] for name, files in groups.items()}

    print(f"machine_id={args.machine_id} seed={args.seed} config_hash={pipeline.config_hash}")
    summary: dict[str, object] = {
        "machine_id": args.machine_id,
        "seed": args.seed,
        "config_hash": pipeline.config_hash,
        "limit": args.limit,
        "splits": {},
    }

    for name, recordings in groups.items():
        start = time.perf_counter()
        matrix, _file_index, _paths = build_matrix(
            recordings, pipeline, data_config.raw_dir, cache_dir
        )
        elapsed = time.perf_counter() - start
        mean, std = log_mel_stats(recordings, pipeline, data_config.raw_dir, cache_dir)
        memory_mb = matrix.nbytes / (1024.0 * 1024.0)

        print(
            f"{name}: files={len(recordings)} shape={matrix.shape} "
            f"time={elapsed:.3f}s mem={memory_mb:.2f}MB "
            f"log_mel_mean={mean:.4f} log_mel_std={std:.4f}"
        )
        summary["splits"][name] = {
            "files": len(recordings),
            "shape": list(matrix.shape),
            "seconds": round(elapsed, 3),
            "memory_mb": round(memory_mb, 3),
            "log_mel_mean": mean,
            "log_mel_std": std,
        }

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / f"features_{args.machine_id}_seed{args.seed}.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {summary_path}")

    if args.pack:
        archive = pack_cache(args.pack, cache_dir, pipeline.config_hash)
        print(f"packed cache/{pipeline.config_hash} into {archive}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
