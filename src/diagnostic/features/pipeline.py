"""Shared feature pipeline and per-file log-mel cache (decisions D5, D7, D16).

One code path turns a waveform into context-stacked log-mel vectors, used
identically by training and inference (decision D7). Log-mel arrays are cached
on disk per file, keyed by the feature config hash, so a changed configuration
never reuses stale features (decision D16).
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import numpy as np

from diagnostic.config import FeatureConfig, load_feature_config
from diagnostic.data.audio import load_wav
from diagnostic.data.mimii import Recording
from diagnostic.features.context import stack_context
from diagnostic.features.logmel import compute_log_mel


class FeaturePipeline:
    """Waveform -> log-mel -> context-vector extractor.

    Attributes:
        config: Feature configuration driving extraction.
        channel: Channel index selected from multi-channel audio.
    """

    def __init__(self, config: FeatureConfig, channel: int = 0) -> None:
        self.config = config
        self.channel = channel

    @classmethod
    def from_yaml(cls, path: str | Path) -> FeaturePipeline:
        """Build a pipeline from a feature-config YAML file."""
        return cls(load_feature_config(path))

    @property
    def config_hash(self) -> str:
        """Short stable hash of the feature config (cache key, decision D16)."""
        return self.config.config_hash()

    def wave_to_vectors(self, wave: np.ndarray) -> np.ndarray:
        """Convert a 1-D waveform to context vectors ``(rows, n_mels * C)``."""
        log_mel = compute_log_mel(wave, self.config)
        return stack_context(log_mel, self.config.context_frames)

    def file_to_log_mel(self, path: str | Path, channel: int | None = None) -> np.ndarray:
        """Load one file and return its log-mel ``(n_mels, T)`` ``float32``."""
        selected = self.channel if channel is None else channel
        wave = load_wav(path, channel=selected, expected_sr=self.config.sample_rate)
        return compute_log_mel(wave, self.config)

    def file_to_vectors(self, path: str | Path, channel: int | None = None) -> np.ndarray:
        """Load one file and return its context vectors ``(rows, n_mels * C)``."""
        log_mel = self.file_to_log_mel(path, channel)
        return stack_context(log_mel, self.config.context_frames)


def _relative_key(path: str | Path, raw_dir: str | Path) -> str:
    """Return ``path`` relative to ``raw_dir`` in posix form (fallback: name)."""
    candidate = Path(path)
    try:
        return candidate.relative_to(raw_dir).as_posix()
    except ValueError:
        return candidate.name


def cache_path_for(
    path: str | Path, raw_dir: str | Path, cache_dir: str | Path, config_hash: str
) -> Path:
    """Return ``<cache_dir>/<config_hash>/<relative path>.npy`` for a recording."""
    return Path(cache_dir) / config_hash / (_relative_key(path, raw_dir) + ".npy")


def _atomic_save(target: Path, array: np.ndarray) -> None:
    """Write ``array`` to ``target`` atomically via a temp file + rename."""
    handle, temp_name = tempfile.mkstemp(dir=target.parent, suffix=".npy")
    os.close(handle)
    try:
        np.save(temp_name, array)
        os.replace(temp_name, target)
    finally:
        if os.path.exists(temp_name):
            os.remove(temp_name)


def cached_log_mel(
    path: str | Path,
    raw_dir: str | Path,
    cache_dir: str | Path,
    pipeline: FeaturePipeline,
) -> np.ndarray:
    """Return the cached log-mel for ``path``, computing it only on a miss.

    Args:
        path: Recording file path.
        raw_dir: Base directory used to key the cache by relative path.
        cache_dir: Cache root directory.
        pipeline: Feature pipeline (provides config hash and extraction).

    Returns:
        ``float32`` log-mel array of shape ``(n_mels, T)``.
    """
    target = cache_path_for(path, raw_dir, cache_dir, pipeline.config_hash)
    if target.is_file():
        return np.load(target)

    log_mel = pipeline.file_to_log_mel(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    _atomic_save(target, log_mel)
    return log_mel


def build_matrix(
    recordings: list[Recording],
    pipeline: FeaturePipeline,
    raw_dir: str | Path | None = None,
    cache_dir: str | Path | None = None,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Build the stacked feature matrix for ``recordings`` in the given order.

    Memory is bounded by preallocating the output once the per-file row counts
    are known. Log-mel arrays are read from ``cache_dir`` when provided.

    Args:
        recordings: Recordings to extract, in the output row order.
        pipeline: Feature pipeline to use.
        raw_dir: Base directory for relative cache keys (required with cache_dir).
        cache_dir: Cache root; if ``None`` features are computed every time.

    Returns:
        ``(X, file_index, paths)`` where ``X`` is ``float32`` ``(N, D)``,
        ``file_index`` is ``int32`` ``(N,)`` mapping each row to its file, and
        ``paths`` is the per-file path list (length = number of recordings).

    Raises:
        ValueError: If a recording has fewer frames than ``context_frames``.
    """
    context = pipeline.config.context_frames
    dim = pipeline.config.n_mels * context

    log_mels: list[np.ndarray] = []
    for recording in recordings:
        if cache_dir is not None and raw_dir is not None:
            log_mels.append(cached_log_mel(recording.path, raw_dir, cache_dir, pipeline))
        else:
            log_mels.append(pipeline.file_to_log_mel(recording.path))

    rows_per_file: list[int] = []
    for log_mel in log_mels:
        if log_mel.shape[1] < context:
            raise ValueError(f"recording has {log_mel.shape[1]} frames < context_frames={context}")
        rows_per_file.append(log_mel.shape[1] - context + 1)

    total_rows = int(sum(rows_per_file))
    matrix = np.empty((total_rows, dim), dtype=np.float32)
    file_index = np.empty(total_rows, dtype=np.int32)

    offset = 0
    for index, log_mel in enumerate(log_mels):
        vectors = stack_context(log_mel, context)
        count = vectors.shape[0]
        matrix[offset : offset + count] = vectors
        file_index[offset : offset + count] = index
        offset += count

    paths = [str(recording.path) for recording in recordings]
    return matrix, file_index, paths
