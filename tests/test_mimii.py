"""Tests for diagnostic.data.mimii discovery."""

from __future__ import annotations

from pathlib import Path

import pytest

from diagnostic.data.mimii import (
    Recording,
    _extract_machine_id,
    discover_recordings,
    machine_type_dir,
)


def _touch(base: Path, rel: str) -> Path:
    path = base / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"RIFF")
    return path


def test_machine_type_dir() -> None:
    assert machine_type_dir("data/raw/mimii", 6, "fan") == Path("data/raw/mimii/6_dB_fan")
    assert machine_type_dir("data/raw/mimii", -6, "pump") == Path("data/raw/mimii/-6_dB_pump")


def test_discover_multiple_ids_both_labels(tmp_path: Path) -> None:
    base = tmp_path / "mimii"
    _touch(base, "6_dB_fan/id_00/normal/n2.wav")
    _touch(base, "6_dB_fan/id_00/normal/n1.wav")
    _touch(base, "6_dB_fan/id_00/abnormal/a1.wav")
    _touch(base, "6_dB_fan/id_02/normal/x.wav")

    recordings = discover_recordings(base, "fan", 6)

    assert len(recordings) == 4
    assert all(isinstance(r, Recording) for r in recordings)
    assert all(r.machine_type == "fan" and r.snr_db == 6 for r in recordings)
    assert {r.machine_id for r in recordings} == {"id_00", "id_02"}
    keys = [(r.machine_id, r.label, str(r.path).lower()) for r in recordings]
    assert keys == sorted(keys)


def test_discover_tolerates_nested_zip_layout(tmp_path: Path) -> None:
    base = tmp_path / "mimii"
    _touch(base, "6_dB_fan/fan/id_00/normal/a.wav")
    _touch(base, "6_dB_fan/some/deep/nesting/id_02/abnormal/b.wav")

    recordings = discover_recordings(base, "fan", 6)

    assert len(recordings) == 2
    assert {r.machine_id for r in recordings} == {"id_00", "id_02"}


def test_discover_ignores_non_wav_and_unknown_labels(tmp_path: Path) -> None:
    base = tmp_path / "mimii"
    _touch(base, "6_dB_fan/id_00/normal/a.wav")
    _touch(base, "6_dB_fan/id_00/normal/readme.txt")
    _touch(base, "6_dB_fan/id_00/otherlabel/c.wav")

    recordings = discover_recordings(base, "fan", 6)

    assert [r.path.name for r in recordings] == ["a.wav"]


def test_discover_is_deterministically_sorted(tmp_path: Path) -> None:
    base = tmp_path / "mimii"
    for name in ("n10.wav", "n2.wav", "n1.wav"):
        _touch(base, f"6_dB_fan/id_00/normal/{name}")

    first = discover_recordings(base, "fan", 6)
    second = discover_recordings(base, "fan", 6)

    assert [r.path.name for r in first] == ["n1.wav", "n10.wav", "n2.wav"]
    assert [r.path for r in first] == [r.path for r in second]


def test_discover_missing_dir_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="not found"):
        discover_recordings(tmp_path / "mimii", "fan", 6)


def test_discover_empty_dir_raises(tmp_path: Path) -> None:
    base = tmp_path / "mimii"
    (base / "6_dB_fan").mkdir(parents=True)
    with pytest.raises(ValueError, match="no .wav recordings"):
        discover_recordings(base, "fan", 6)


def test_extract_machine_id_ok_and_error(tmp_path: Path) -> None:
    assert _extract_machine_id(Path("a/id_04/normal/x.wav")) == "id_04"
    with pytest.raises(ValueError, match="id_XX"):
        _extract_machine_id(Path("a/notanid/normal/x.wav"))
