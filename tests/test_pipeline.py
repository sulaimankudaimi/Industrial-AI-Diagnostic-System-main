"""Tests for diagnostic.features.pipeline (pipeline and per-file cache)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from diagnostic.config import FeatureConfig
from diagnostic.data.mimii import Recording
from diagnostic.features import pipeline as pipeline_module
from diagnostic.features.pipeline import (
    FeaturePipeline,
    build_matrix,
    cache_path_for,
    cached_log_mel,
)


def _sine(n: int = 16000, sr: int = 16000, freq: float = 1000.0) -> np.ndarray:
    t = np.arange(n, dtype=np.float32) / sr
    return np.sin(2 * np.pi * freq * t, dtype=np.float32)


def _write_wav(path: Path, data: np.ndarray, sr: int = 16000) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, data.astype(np.float32), sr, subtype="FLOAT")
    return path


def test_wave_to_vectors_shape_and_dtype() -> None:
    pipeline = FeaturePipeline(FeatureConfig())
    vectors = pipeline.wave_to_vectors(_sine())
    assert vectors.dtype == np.float32
    assert vectors.shape == (28, 64 * 5)  # T=32 frames -> 32-5+1 rows


def test_file_to_log_mel_and_vectors(tmp_path: Path) -> None:
    pipeline = FeaturePipeline(FeatureConfig())
    path = _write_wav(tmp_path / "mimii" / "a.wav", _sine())
    assert pipeline.file_to_log_mel(path).shape == (64, 32)
    assert pipeline.file_to_vectors(path).shape == (28, 320)


def test_from_yaml_matches_default() -> None:
    pipeline = FeaturePipeline.from_yaml("configs/features/logmel_default.yaml")
    assert pipeline.config == FeatureConfig()
    assert pipeline.config_hash == FeatureConfig().config_hash()


def test_cache_hit_does_not_recompute(tmp_path: Path, monkeypatch) -> None:
    raw_dir = tmp_path / "mimii"
    wav = _write_wav(raw_dir / "6_dB_fan" / "id_00" / "normal" / "a.wav", _sine())
    cache_dir = tmp_path / "cache"
    pipeline = FeaturePipeline(FeatureConfig())

    calls = {"count": 0}
    real = pipeline_module.compute_log_mel

    def counting(wave: np.ndarray, config: FeatureConfig) -> np.ndarray:
        calls["count"] += 1
        return real(wave, config)

    monkeypatch.setattr(pipeline_module, "compute_log_mel", counting)

    first = cached_log_mel(wav, raw_dir, cache_dir, pipeline)
    assert calls["count"] == 1
    second = cached_log_mel(wav, raw_dir, cache_dir, pipeline)
    assert calls["count"] == 1  # served from disk
    np.testing.assert_array_equal(first, second)


def test_different_config_uses_different_folder(tmp_path: Path) -> None:
    raw_dir = tmp_path / "mimii"
    wav = _write_wav(raw_dir / "6_dB_fan" / "id_00" / "normal" / "a.wav", _sine())
    cache_dir = tmp_path / "cache"

    default = FeaturePipeline(FeatureConfig())
    other = FeaturePipeline(FeatureConfig(n_mels=32))
    assert default.config_hash != other.config_hash

    path_a = cache_path_for(wav, raw_dir, cache_dir, default.config_hash)
    path_b = cache_path_for(wav, raw_dir, cache_dir, other.config_hash)
    assert path_a != path_b
    assert path_a.relative_to(cache_dir).parts[0] == default.config_hash
    assert path_b.relative_to(cache_dir).parts[0] == other.config_hash

    cached_log_mel(wav, raw_dir, cache_dir, default)
    cached_log_mel(wav, raw_dir, cache_dir, other)
    assert path_a.is_file()
    assert path_b.is_file()


def test_cache_path_layout(tmp_path: Path) -> None:
    raw_dir = tmp_path / "mimii"
    wav = raw_dir / "6_dB_fan" / "id_00" / "normal" / "a.wav"
    path = cache_path_for(wav, raw_dir, tmp_path / "cache", "abc123")
    assert path == tmp_path / "cache" / "abc123" / "6_dB_fan" / "id_00" / "normal" / "a.wav.npy"


def _recordings(raw_dir: Path, lengths: list[int]) -> list[Recording]:
    recordings = []
    for index, length in enumerate(lengths):
        wav = _write_wav(raw_dir / "6_dB_fan" / "id_00" / "normal" / f"r{index}.wav", _sine(length))
        recordings.append(Recording(wav, "fan", "id_00", 6, "normal"))
    return recordings


def test_build_matrix_index_mapping(tmp_path: Path) -> None:
    raw_dir = tmp_path / "mimii"
    cache_dir = tmp_path / "cache"
    pipeline = FeaturePipeline(FeatureConfig())
    recordings = _recordings(raw_dir, [16000, 8192, 12288])  # 32, 17, 25 frames

    matrix, file_index, paths = build_matrix(recordings, pipeline, raw_dir, cache_dir)

    assert matrix.dtype == np.float32
    assert file_index.dtype == np.int32
    assert paths == [str(r.path) for r in recordings]
    assert matrix.shape == (file_index.size, 320)

    offset = 0
    for index, recording in enumerate(recordings):
        expected = pipeline.file_to_vectors(recording.path)
        count = expected.shape[0]
        np.testing.assert_allclose(matrix[offset : offset + count], expected, atol=1e-5)
        assert np.all(file_index[offset : offset + count] == index)
        offset += count
    assert offset == matrix.shape[0]


def test_build_matrix_matches_without_cache(tmp_path: Path) -> None:
    raw_dir = tmp_path / "mimii"
    pipeline = FeaturePipeline(FeatureConfig())
    recordings = _recordings(raw_dir, [16000, 12000])

    cached, _, _ = build_matrix(recordings, pipeline, raw_dir, tmp_path / "cache")
    uncached, _, _ = build_matrix(recordings, pipeline)
    np.testing.assert_allclose(cached, uncached)


def test_build_matrix_too_short_raises(tmp_path: Path) -> None:
    raw_dir = tmp_path / "mimii"
    pipeline = FeaturePipeline(FeatureConfig())
    recordings = _recordings(raw_dir, [1024])  # T = 3 frames < context_frames=5
    with pytest.raises(ValueError, match="context_frames"):
        build_matrix(recordings, pipeline)
