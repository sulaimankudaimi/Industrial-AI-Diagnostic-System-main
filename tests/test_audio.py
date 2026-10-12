"""Tests for diagnostic.data.audio loading."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from diagnostic.data.audio import load_wav


def _write_wav(path: Path, data: np.ndarray, sr: int = 16000) -> Path:
    sf.write(path, data, sr, subtype="FLOAT")
    return path


def test_load_channel_zero(tmp_path: Path) -> None:
    sr = 16000
    left = np.linspace(-0.5, 0.5, 1000, dtype=np.float32)
    right = np.zeros(1000, dtype=np.float32)
    stereo = np.stack([left, right], axis=1)
    path = _write_wav(tmp_path / "stereo.wav", stereo, sr)

    mono = load_wav(path, channel=0, expected_sr=sr)

    assert mono.dtype == np.float32
    assert mono.ndim == 1
    assert mono.shape == (1000,)
    np.testing.assert_allclose(mono, left, atol=1e-6)


def test_load_second_channel(tmp_path: Path) -> None:
    sr = 16000
    stereo = np.zeros((100, 2), dtype=np.float32)
    stereo[:, 1] = 0.25
    path = _write_wav(tmp_path / "stereo.wav", stereo, sr)

    mono = load_wav(path, channel=1, expected_sr=sr)

    np.testing.assert_allclose(mono, np.full(100, 0.25, dtype=np.float32), atol=1e-6)


def test_wrong_sample_rate_raises(tmp_path: Path) -> None:
    path = _write_wav(tmp_path / "a.wav", np.zeros(100, dtype=np.float32), sr=8000)
    with pytest.raises(ValueError, match="sample rate"):
        load_wav(path, expected_sr=16000)


def test_channel_out_of_range_raises(tmp_path: Path) -> None:
    path = _write_wav(tmp_path / "mono.wav", np.zeros(100, dtype=np.float32))
    with pytest.raises(ValueError, match="out of range"):
        load_wav(path, channel=1, expected_sr=16000)


def test_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_wav(tmp_path / "missing.wav")


def test_empty_audio_raises(tmp_path: Path) -> None:
    path = tmp_path / "empty.wav"
    with sf.SoundFile(path, "w", samplerate=16000, channels=1, subtype="FLOAT"):
        pass
    with pytest.raises(ValueError, match="empty"):
        load_wav(path, expected_sr=16000)


def test_unreadable_file_raises(tmp_path: Path) -> None:
    path = tmp_path / "not_audio.wav"
    path.write_bytes(b"this is not a wav file")
    with pytest.raises(ValueError, match="could not read"):
        load_wav(path, expected_sr=16000)
