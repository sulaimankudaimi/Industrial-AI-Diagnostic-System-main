"""Typed configuration objects and YAML loaders for the diagnostic pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

VALID_MACHINE_TYPES: tuple[str, ...] = ("fan", "pump", "slider", "valve")
VALID_SNR_DB: tuple[int, ...] = (-6, 0, 6)


def _require_int(value: Any, field_name: str) -> int:
    """Return ``value`` as an int or raise ``ValueError`` for non-integers."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{field_name} must be an integer, got {value!r}")
    return value


@dataclass(frozen=True)
class DataConfig:
    """Configuration for the MIMII data layer.

    Attributes:
        raw_dir: Root directory holding extracted MIMII recordings.
        splits_dir: Directory where split manifests (JSON) are written.
        machine_type: Machine class, one of fan/pump/slider/valve.
        snr_db: Signal-to-noise ratio of the recordings, one of -6, 0, 6 (dB).
        sample_rate: Expected sampling rate (Hz).
        channel: Channel index selected from multi-channel recordings.
        val_fraction: Fraction of remaining normal files used for validation.
        seeds: Explicit random seeds used to generate splits.
    """

    raw_dir: Path
    splits_dir: Path
    machine_type: str
    snr_db: int
    sample_rate: int = 16000
    channel: int = 0
    val_fraction: float = 0.1
    seeds: tuple[int, ...] = (0,)

    def __post_init__(self) -> None:
        object.__setattr__(self, "raw_dir", Path(self.raw_dir))
        object.__setattr__(self, "splits_dir", Path(self.splits_dir))
        object.__setattr__(self, "seeds", tuple(self.seeds))

        if self.machine_type not in VALID_MACHINE_TYPES:
            raise ValueError(
                f"machine_type must be one of {VALID_MACHINE_TYPES}, got {self.machine_type!r}"
            )
        if self.snr_db not in VALID_SNR_DB:
            raise ValueError(f"snr_db must be one of {VALID_SNR_DB}, got {self.snr_db!r}")
        if not 0.0 < float(self.val_fraction) < 0.5:
            raise ValueError(f"val_fraction must be in (0, 0.5), got {self.val_fraction!r}")
        if _require_int(self.sample_rate, "sample_rate") <= 0:
            raise ValueError(f"sample_rate must be positive, got {self.sample_rate!r}")
        if _require_int(self.channel, "channel") < 0:
            raise ValueError(f"channel must be >= 0, got {self.channel!r}")
        if not self.seeds:
            raise ValueError("seeds must contain at least one seed")
        for seed in self.seeds:
            if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
                raise ValueError(f"each seed must be a non-negative integer, got {seed!r}")
        if len(set(self.seeds)) != len(self.seeds):
            raise ValueError(f"seeds must be unique, got {self.seeds!r}")


_REQUIRED_DATA_KEYS: tuple[str, ...] = ("raw_dir", "splits_dir", "machine_type", "snr_db")
_ALLOWED_DATA_KEYS: tuple[str, ...] = _REQUIRED_DATA_KEYS + (
    "sample_rate",
    "channel",
    "val_fraction",
    "seeds",
)


def load_data_config(path: str | Path) -> DataConfig:
    """Load and validate a :class:`DataConfig` from a YAML file.

    Args:
        path: Path to a YAML file describing the data layer.

    Returns:
        A validated frozen :class:`DataConfig`.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the YAML is malformed or a field is missing/invalid.
    """
    file_path = Path(path)
    if not file_path.is_file():
        raise FileNotFoundError(f"config file not found: {file_path}")

    with file_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)

    if raw is None:
        raise ValueError(f"config file is empty: {file_path}")
    if not isinstance(raw, dict):
        raise ValueError(f"config root must be a mapping in {file_path}, got {type(raw).__name__}")

    missing = [key for key in _REQUIRED_DATA_KEYS if key not in raw]
    if missing:
        raise ValueError(f"missing required key(s) {missing} in {file_path}")
    unknown = [key for key in raw if key not in _ALLOWED_DATA_KEYS]
    if unknown:
        raise ValueError(f"unknown key(s) {unknown} in {file_path}")

    seeds_raw = raw.get("seeds", (0,))
    if isinstance(seeds_raw, int) and not isinstance(seeds_raw, bool):
        seeds = (seeds_raw,)
    else:
        seeds = tuple(seeds_raw)

    return DataConfig(
        raw_dir=Path(raw["raw_dir"]),
        splits_dir=Path(raw["splits_dir"]),
        machine_type=raw["machine_type"],
        snr_db=raw["snr_db"],
        sample_rate=raw.get("sample_rate", 16000),
        channel=raw.get("channel", 0),
        val_fraction=raw.get("val_fraction", 0.1),
        seeds=seeds,
    )
