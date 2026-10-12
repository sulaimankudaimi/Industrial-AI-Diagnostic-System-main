"""Tests for diagnostic.data.splits (design decision D3)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from diagnostic.data.mimii import Recording
from diagnostic.data.splits import (
    MIN_TRAIN,
    content_hash,
    load_split,
    make_split,
    manifest_path,
    save_split,
)


def _recs(
    raw_dir: Path, machine_id: str = "id_00", n_normal: int = 100, n_abnormal: int = 20
) -> list[Recording]:
    base = raw_dir / "6_dB_fan" / machine_id
    recordings: list[Recording] = []
    for i in range(n_normal):
        recordings.append(
            Recording(base / "normal" / f"n{i:05d}.wav", "fan", machine_id, 6, "normal")
        )
    for i in range(n_abnormal):
        recordings.append(
            Recording(base / "abnormal" / f"a{i:05d}.wav", "fan", machine_id, 6, "abnormal")
        )
    return recordings


def _paths(recordings: tuple[Recording, ...]) -> set[str]:
    return {str(r.path) for r in recordings}


def test_sizes_default_fraction(tmp_path: Path) -> None:
    raw = tmp_path / "mimii"
    split = make_split(_recs(raw), "id_00", seed=0)

    assert len(split.test_abnormal) == 20
    assert len(split.test_normal) == 20
    assert len(split.val) == 8  # round(0.1 * 80)
    assert len(split.train) == 72


def test_disjoint_and_full_coverage(tmp_path: Path) -> None:
    raw = tmp_path / "mimii"
    split = make_split(_recs(raw), "id_00", seed=0)

    groups = [split.train, split.val, split.test_normal, split.test_abnormal]
    all_paths = [str(r.path) for group in groups for r in group]
    assert len(all_paths) == len(set(all_paths)) == 120
    assert all(r.label == "normal" for r in split.train)
    assert all(r.label == "normal" for r in split.val)
    assert all(r.label == "abnormal" for r in split.test_abnormal)


def test_deterministic_for_same_seed(tmp_path: Path) -> None:
    raw = tmp_path / "mimii"
    recordings = _recs(raw)
    first = make_split(recordings, "id_00", seed=7)
    second = make_split(recordings, "id_00", seed=7)
    assert _paths(first.train) == _paths(second.train)
    assert _paths(first.val) == _paths(second.val)
    assert _paths(first.test_normal) == _paths(second.test_normal)


def test_different_seed_changes_selection(tmp_path: Path) -> None:
    raw = tmp_path / "mimii"
    recordings = _recs(raw)
    a = make_split(recordings, "id_00", seed=0)
    b = make_split(recordings, "id_00", seed=1)
    assert _paths(a.test_normal) != _paths(b.test_normal)
    assert _paths(a.train) != _paths(b.train)


def test_not_enough_normals_raises(tmp_path: Path) -> None:
    raw = tmp_path / "mimii"
    with pytest.raises(ValueError, match="not enough normal recordings"):
        make_split(_recs(raw, n_normal=20, n_abnormal=20), "id_00", seed=0)


def test_unknown_machine_id_raises(tmp_path: Path) -> None:
    raw = tmp_path / "mimii"
    with pytest.raises(ValueError, match="no recordings"):
        make_split(_recs(raw), "id_99", seed=0)


def test_val_fraction_out_of_range_raises(tmp_path: Path) -> None:
    raw = tmp_path / "mimii"
    with pytest.raises(ValueError, match="val_fraction"):
        make_split(_recs(raw), "id_00", seed=0, val_fraction=0.6)


def test_val_has_at_least_one(tmp_path: Path) -> None:
    raw = tmp_path / "mimii"
    split = make_split(_recs(raw, n_normal=16, n_abnormal=5), "id_00", seed=0)
    assert len(split.val) == 1
    assert len(split.train) == MIN_TRAIN


def test_train_too_small_raises(tmp_path: Path) -> None:
    raw = tmp_path / "mimii"
    with pytest.raises(ValueError, match="train"):
        make_split(_recs(raw, n_normal=16, n_abnormal=5), "id_00", seed=0, val_fraction=0.49)


def test_manifest_path_naming(tmp_path: Path) -> None:
    path = manifest_path(tmp_path / "splits", "fan", 6, "id_00", 2)
    assert path == tmp_path / "splits" / "fan_6dB_id_00_seed2.json"


def test_manifest_round_trip(tmp_path: Path) -> None:
    raw = tmp_path / "mimii"
    splits_dir = tmp_path / "splits"
    split = make_split(_recs(raw), "id_00", seed=3)

    path = save_split(split, raw, splits_dir)
    assert path.is_file()
    loaded = load_split(path, raw)

    assert loaded.machine_type == split.machine_type
    assert loaded.machine_id == split.machine_id
    assert loaded.snr_db == split.snr_db
    assert loaded.seed == split.seed
    assert loaded.val_fraction == split.val_fraction
    assert _paths(loaded.train) == _paths(split.train)
    assert _paths(loaded.val) == _paths(split.val)
    assert _paths(loaded.test_normal) == _paths(split.test_normal)
    assert _paths(loaded.test_abnormal) == _paths(split.test_abnormal)


def test_manifest_stores_relative_paths_and_counts(tmp_path: Path) -> None:
    raw = tmp_path / "mimii"
    splits_dir = tmp_path / "splits"
    split = make_split(_recs(raw), "id_00", seed=0)
    path = save_split(split, raw, splits_dir)

    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["counts"] == {
        "train": 72,
        "val": 8,
        "test_normal": 20,
        "test_abnormal": 20,
    }
    assert not any(Path(rel).is_absolute() for rel in payload["files"]["train"])
    assert payload["files"]["train"][0].startswith("6_dB_fan/id_00/normal/")


def test_content_hash_stable_and_seed_dependent(tmp_path: Path) -> None:
    raw = tmp_path / "mimii"
    recordings = _recs(raw)
    a = make_split(recordings, "id_00", seed=0)
    b = make_split(recordings, "id_00", seed=0)
    c = make_split(recordings, "id_00", seed=1)

    assert content_hash(a, raw) == content_hash(b, raw)
    assert content_hash(a, raw) != content_hash(c, raw)
    assert len(content_hash(a, raw)) == 12
