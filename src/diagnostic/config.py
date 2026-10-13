"""Typed configuration objects and YAML loaders for the diagnostic pipeline."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

import yaml

VALID_MACHINE_TYPES: tuple[str, ...] = ("fan", "pump", "slider", "valve")
VALID_SNR_DB: tuple[int, ...] = (-6, 0, 6)
VALID_POWER: tuple[float, ...] = (1.0, 2.0)


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


CONFIG_HASH_LENGTH = 10


def _require_float(value: Any, field_name: str) -> float:
    """Return ``value`` as a float or raise ``ValueError`` for booleans/strings."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field_name} must be a number, got {value!r}")
    return float(value)


@dataclass(frozen=True)
class FeatureConfig:
    """Configuration for the shared log-mel feature pipeline.

    All fields are tunable via YAML only (no magic numbers in code). Frequencies
    are in Hz, ``power`` selects the spectrogram exponent, ``eps`` is the small
    additive floor (linear power units) that keeps the log finite.

    Attributes:
        sample_rate: Expected sampling rate in Hz.
        n_fft: STFT window length in samples (> 0).
        hop_length: STFT hop in samples (> 0).
        n_mels: Number of mel filterbank channels (> 0).
        fmin: Lowest mel band edge in Hz (>= 0).
        fmax: Highest mel band edge in Hz (<= sample_rate / 2).
        power: Spectrogram exponent, either 1.0 (magnitude) or 2.0 (power).
        eps: Additive floor inside the logarithm (> 0, linear units).
        context_frames: Number of consecutive frames stacked per vector (>= 1).
    """

    sample_rate: int = 16000
    n_fft: int = 1024
    hop_length: int = 512
    n_mels: int = 64
    fmin: float = 0.0
    fmax: float = 8000.0
    power: float = 2.0
    eps: float = 1e-10
    context_frames: int = 5

    def __post_init__(self) -> None:
        object.__setattr__(self, "fmin", float(self.fmin))
        object.__setattr__(self, "fmax", float(self.fmax))
        object.__setattr__(self, "power", float(self.power))
        object.__setattr__(self, "eps", float(self.eps))

        if _require_int(self.sample_rate, "sample_rate") <= 0:
            raise ValueError(f"sample_rate must be positive, got {self.sample_rate!r}")
        if _require_int(self.n_fft, "n_fft") <= 0:
            raise ValueError(f"n_fft must be positive, got {self.n_fft!r}")
        if _require_int(self.hop_length, "hop_length") <= 0:
            raise ValueError(f"hop_length must be positive, got {self.hop_length!r}")
        if _require_int(self.n_mels, "n_mels") <= 0:
            raise ValueError(f"n_mels must be positive, got {self.n_mels!r}")
        if _require_int(self.context_frames, "context_frames") < 1:
            raise ValueError(f"context_frames must be >= 1, got {self.context_frames!r}")
        if self.fmin < 0.0:
            raise ValueError(f"fmin must be >= 0, got {self.fmin!r}")
        if self.fmax <= self.fmin:
            raise ValueError(f"fmax must be greater than fmin, got fmax={self.fmax!r}")
        nyquist = self.sample_rate / 2.0
        if self.fmax > nyquist:
            raise ValueError(f"fmax must be <= sample_rate / 2 ({nyquist}), got {self.fmax!r}")
        if self.power not in VALID_POWER:
            raise ValueError(f"power must be one of {VALID_POWER}, got {self.power!r}")
        if self.eps <= 0.0:
            raise ValueError(f"eps must be > 0, got {self.eps!r}")

    def config_hash(self) -> str:
        """Return a stable short hash (first 10 hex chars of SHA-1) of the config.

        The hash is computed from the canonical (sorted-key, compact) JSON of all
        fields, so any parameter change yields a different cache folder.
        """
        payload = {field.name: getattr(self, field.name) for field in fields(self)}
        canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha1(canonical.encode("utf-8")).hexdigest()[:CONFIG_HASH_LENGTH]


_ALLOWED_FEATURE_KEYS: tuple[str, ...] = tuple(field.name for field in fields(FeatureConfig))


def load_feature_config(path: str | Path) -> FeatureConfig:
    """Load and validate a :class:`FeatureConfig` from a YAML file.

    Args:
        path: Path to a YAML file describing the feature pipeline.

    Returns:
        A validated frozen :class:`FeatureConfig`.

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

    unknown = [key for key in raw if key not in _ALLOWED_FEATURE_KEYS]
    if unknown:
        raise ValueError(f"unknown key(s) {unknown} in {file_path}")

    return FeatureConfig(**raw)
