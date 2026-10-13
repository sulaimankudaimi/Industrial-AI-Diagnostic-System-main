"""Tests for diagnostic.config.FeatureConfig and its YAML loader."""

from __future__ import annotations

from pathlib import Path

import pytest

from diagnostic.config import (
    CONFIG_HASH_LENGTH,
    FeatureConfig,
    load_feature_config,
)


def _write_yaml(tmp_path: Path, body: str) -> Path:
    path = tmp_path / "features.yaml"
    path.write_text(body, encoding="utf-8")
    return path


def test_defaults_are_valid() -> None:
    cfg = FeatureConfig()
    assert cfg.sample_rate == 16000
    assert cfg.n_fft == 1024
    assert cfg.hop_length == 512
    assert cfg.n_mels == 64
    assert cfg.fmin == 0.0
    assert cfg.fmax == 8000.0
    assert cfg.power == 2.0
    assert cfg.eps == 1e-10
    assert cfg.context_frames == 5


def test_config_hash_is_stable_and_short() -> None:
    first = FeatureConfig().config_hash()
    second = FeatureConfig().config_hash()
    assert first == second
    assert len(first) == CONFIG_HASH_LENGTH


def test_config_hash_changes_with_field() -> None:
    base = FeatureConfig()
    assert FeatureConfig(n_mels=32).config_hash() != base.config_hash()
    assert FeatureConfig(context_frames=3).config_hash() != base.config_hash()
    assert FeatureConfig(power=1.0).config_hash() != base.config_hash()


def test_power_one_is_allowed() -> None:
    assert FeatureConfig(power=1.0).power == 1.0


@pytest.mark.parametrize(
    "kwargs",
    [
        {"fmax": 9000.0},  # > sample_rate / 2
        {"n_fft": 0},
        {"hop_length": 0},
        {"n_mels": 0},
        {"context_frames": 0},
        {"eps": 0.0},
        {"power": 3.0},
        {"fmin": -1.0},
        {"fmax": 0.0},  # <= fmin
        {"sample_rate": 0},
    ],
)
def test_invalid_fields_raise(kwargs: dict) -> None:
    with pytest.raises(ValueError):
        FeatureConfig(**kwargs)


def test_load_default_yaml() -> None:
    cfg = load_feature_config("configs/features/logmel_default.yaml")
    assert cfg == FeatureConfig()


def test_load_custom_yaml(tmp_path: Path) -> None:
    path = _write_yaml(tmp_path, "n_mels: 32\ncontext_frames: 3\npower: 1.0\n")
    cfg = load_feature_config(path)
    assert cfg.n_mels == 32
    assert cfg.context_frames == 3
    assert cfg.power == 1.0


def test_load_unknown_key_raises(tmp_path: Path) -> None:
    path = _write_yaml(tmp_path, "bogus: 1\n")
    with pytest.raises(ValueError, match="unknown key"):
        load_feature_config(path)


def test_load_empty_raises(tmp_path: Path) -> None:
    path = _write_yaml(tmp_path, "")
    with pytest.raises(ValueError, match="empty"):
        load_feature_config(path)


def test_load_non_mapping_raises(tmp_path: Path) -> None:
    path = _write_yaml(tmp_path, "- a\n- b\n")
    with pytest.raises(ValueError, match="mapping"):
        load_feature_config(path)


def test_load_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_feature_config(tmp_path / "missing.yaml")
