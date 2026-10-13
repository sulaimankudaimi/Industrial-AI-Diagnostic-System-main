"""Tests for diagnostic.features.extract (CLI, pack/unpack)."""

from __future__ import annotations

import json
import tarfile
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from diagnostic.data.mimii import Recording
from diagnostic.data.splits import make_split, save_split
from diagnostic.features.extract import main, unpack_cache


def _write_wav(path: Path, freq: float, seconds: float = 1.0, sr: int = 16000) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    t = np.arange(int(seconds * sr), dtype=np.float32) / sr
    data = (0.1 * np.sin(2 * np.pi * freq * t)).astype(np.float32)
    sf.write(path, data, sr, subtype="FLOAT")
    return path


def _build_dataset(tmp_path: Path, n_normal: int = 16, n_abnormal: int = 5) -> tuple[Path, Path]:
    raw_dir = tmp_path / "mimii"
    base = raw_dir / "6_dB_fan" / "id_00"
    recordings: list[Recording] = []
    for i in range(n_normal):
        path = _write_wav(base / "normal" / f"n{i:04d}.wav", freq=800.0 + i)
        recordings.append(Recording(path, "fan", "id_00", 6, "normal"))
    for i in range(n_abnormal):
        path = _write_wav(base / "abnormal" / f"a{i:04d}.wav", freq=1200.0 + i)
        recordings.append(Recording(path, "fan", "id_00", 6, "abnormal"))

    splits_dir = tmp_path / "splits"
    save_split(make_split(recordings, "id_00", seed=0), raw_dir, splits_dir)
    return raw_dir, splits_dir


def _write_configs(tmp_path: Path, raw_dir: Path, splits_dir: Path) -> tuple[Path, Path]:
    data_config = tmp_path / "data.yaml"
    data_config.write_text(
        "\n".join(
            [
                f"raw_dir: {raw_dir.as_posix()}",
                f"splits_dir: {splits_dir.as_posix()}",
                "machine_type: fan",
                "snr_db: 6",
                "seeds: [0]",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    feature_config = tmp_path / "features.yaml"
    feature_config.write_text(
        "\n".join(
            [
                "sample_rate: 16000",
                "n_fft: 1024",
                "hop_length: 512",
                "n_mels: 64",
                "context_frames: 5",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return data_config, feature_config


def _extract_args(tmp_path: Path) -> tuple[Path, Path, list[str]]:
    raw_dir, splits_dir = _build_dataset(tmp_path)
    data_config, feature_config = _write_configs(tmp_path, raw_dir, splits_dir)
    cache_dir = tmp_path / "cache"
    out_dir = tmp_path / "runs"
    args = [
        "--data-config",
        str(data_config),
        "--features-config",
        str(feature_config),
        "--machine-id",
        "id_00",
        "--seed",
        "0",
        "--cache-dir",
        str(cache_dir),
        "--out-dir",
        str(out_dir),
    ]
    return cache_dir, out_dir, args


def test_main_extracts_and_writes_summary(tmp_path: Path) -> None:
    cache_dir, out_dir, args = _extract_args(tmp_path)

    assert main(args) == 0

    summary = json.loads((out_dir / "features_id_00_seed0.json").read_text(encoding="utf-8"))
    assert summary["splits"]["train"]["files"] == 10
    assert summary["splits"]["val"]["files"] == 1
    assert summary["splits"]["test"]["files"] == 10
    assert summary["splits"]["train"]["shape"][1] == 64 * 5
    assert (cache_dir / summary["config_hash"]).is_dir()


def test_limit_is_reported_prominently(tmp_path: Path, capsys) -> None:
    _cache_dir, out_dir, args = _extract_args(tmp_path)

    assert main([*args, "--limit", "2"]) == 0

    out = capsys.readouterr().out
    assert "LIMIT ACTIVE" in out
    summary = json.loads((out_dir / "features_id_00_seed0.json").read_text(encoding="utf-8"))
    assert summary["limit"] == 2
    assert summary["splits"]["train"]["files"] == 2
    assert summary["splits"]["test"]["files"] == 2
    assert summary["splits"]["val"]["files"] <= 2


def test_pack_then_unpack_round_trip(tmp_path: Path) -> None:
    cache_dir, _out_dir, args = _extract_args(tmp_path)
    archive = tmp_path / "features.tar.gz"

    assert main([*args, "--pack", str(archive)]) == 0
    assert archive.is_file()

    fresh_cache = tmp_path / "restored"
    assert main(["--unpack", str(archive), "--cache-dir", str(fresh_cache)]) == 0

    original_files = sorted(p.relative_to(cache_dir).as_posix() for p in cache_dir.rglob("*.npy"))
    restored_files = sorted(
        p.relative_to(fresh_cache).as_posix() for p in fresh_cache.rglob("*.npy")
    )
    assert original_files
    assert original_files == restored_files


def test_unpack_rejects_path_traversal(tmp_path: Path) -> None:
    payload = tmp_path / "payload.txt"
    payload.write_text("x", encoding="utf-8")
    archive = tmp_path / "evil.tar.gz"
    with tarfile.open(archive, "w:gz") as handle:
        handle.add(payload, arcname="../evil.txt")

    with pytest.raises(ValueError, match="unsafe"):
        unpack_cache(archive, tmp_path / "dest")


def test_conflicting_pack_and_unpack_rejected(tmp_path: Path) -> None:
    _cache_dir, _out_dir, args = _extract_args(tmp_path)
    with pytest.raises(SystemExit):
        main([*args, "--pack", str(tmp_path / "a.tar.gz"), "--unpack", str(tmp_path / "b.tar.gz")])
