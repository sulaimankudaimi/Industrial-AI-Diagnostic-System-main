"""Tests for diagnostic.features.logmel."""

from __future__ import annotations

import librosa
import numpy as np
import pytest

from diagnostic.config import FeatureConfig
from diagnostic.features.logmel import compute_log_mel


def _sine(freq: float, seconds: float = 1.0, sr: int = 16000) -> np.ndarray:
    t = np.arange(int(seconds * sr), dtype=np.float32) / sr
    return np.sin(2 * np.pi * freq * t, dtype=np.float32)


def test_shape_and_dtype() -> None:
    cfg = FeatureConfig()
    wave = _sine(1000.0)
    log_mel = compute_log_mel(wave, cfg)
    assert log_mel.dtype == np.float32
    assert log_mel.shape[0] == cfg.n_mels
    assert log_mel.shape[1] == 1 + wave.size // cfg.hop_length
    assert np.all(np.isfinite(log_mel))


def test_deterministic() -> None:
    cfg = FeatureConfig()
    wave = _sine(440.0)
    np.testing.assert_array_equal(compute_log_mel(wave, cfg), compute_log_mel(wave, cfg))


def test_absolute_level_preserved() -> None:
    cfg = FeatureConfig()
    rng = np.random.default_rng(0)
    wave = rng.standard_normal(16000).astype(np.float32) * 0.1
    base = compute_log_mel(wave, cfg)
    doubled = compute_log_mel(wave * 2.0, cfg)
    expected = 20.0 * np.log10(2.0)
    assert abs(float(np.mean(doubled) - np.mean(base)) - expected) < 0.1


def test_silence_is_finite() -> None:
    cfg = FeatureConfig()
    log_mel = compute_log_mel(np.zeros(16000, dtype=np.float32), cfg)
    assert np.all(np.isfinite(log_mel))
    np.testing.assert_allclose(log_mel, 10.0 * np.log10(cfg.eps), atol=1e-3)


def test_1khz_sine_peaks_in_nearest_mel_bin() -> None:
    cfg = FeatureConfig()
    log_mel = compute_log_mel(_sine(1000.0), cfg)
    peak_bin = int(np.argmax(log_mel.mean(axis=1)))
    centers = librosa.mel_frequencies(n_mels=cfg.n_mels, fmin=cfg.fmin, fmax=cfg.fmax)
    nearest = int(np.argmin(np.abs(centers - 1000.0)))
    assert abs(peak_bin - nearest) <= 1


@pytest.mark.parametrize(
    "wave",
    [
        np.zeros((2, 100), dtype=np.float32),  # 2-D
        np.zeros(0, dtype=np.float32),  # empty
        np.zeros(100, dtype=np.float32),  # shorter than n_fft
        np.array([np.nan] * 2048, dtype=np.float32),  # NaN
    ],
)
def test_invalid_wave_raises(wave: np.ndarray) -> None:
    with pytest.raises(ValueError):
        compute_log_mel(wave, FeatureConfig())
