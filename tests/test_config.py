"""Tests for diagnostic.config typed YAML loading."""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest
import yaml

from diagnostic.config import DataConfig, load_data_config

REPO_ROOT = Path(__file__).resolve().parents[1]


def _write_yaml(tmp_path: Path, data: object) -> Path:
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    return path


def _minimal() -> dict[str, object]:
    return {
        "raw_dir": "data/raw/mimii",
        "splits_dir": "data/splits",
        "machine_type": "fan",
        "snr_db": 6,
    }


def test_load_real_config() -> None:
    config = load_data_config(REPO_ROOT / "configs" / "data" / "mimii_fan_6db.yaml")
    assert config.machine_type == "fan"
    assert config.snr_db == 6
    assert config.channel == 0
    assert config.sample_rate == 16000
    assert config.val_fraction == 0.1
    assert config.seeds == (0, 1, 2, 3, 4)
    assert config.raw_dir == Path("data/raw/mimii")
    assert config.splits_dir == Path("data/splits")


def test_minimal_defaults(tmp_path: Path) -> None:
    config = load_data_config(_write_yaml(tmp_path, _minimal()))
    assert config.sample_rate == 16000
    assert config.channel == 0
    assert config.val_fraction == 0.1
    assert config.seeds == (0,)


def test_seeds_scalar_is_coerced_to_tuple(tmp_path: Path) -> None:
    data = _minimal() | {"seeds": 3}
    config = load_data_config(_write_yaml(tmp_path, data))
    assert config.seeds == (3,)


def test_missing_required_key(tmp_path: Path) -> None:
    data = _minimal()
    del data["snr_db"]
    with pytest.raises(ValueError, match="missing required key"):
        load_data_config(_write_yaml(tmp_path, data))


def test_unknown_key(tmp_path: Path) -> None:
    data = _minimal() | {"bogus": 1}
    with pytest.raises(ValueError, match="unknown key"):
        load_data_config(_write_yaml(tmp_path, data))


def test_root_not_mapping(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="must be a mapping"):
        load_data_config(_write_yaml(tmp_path, [1, 2, 3]))


def test_empty_file(tmp_path: Path) -> None:
    path = tmp_path / "empty.yaml"
    path.write_text("", encoding="utf-8")
    with pytest.raises(ValueError, match="empty"):
        load_data_config(path)


def test_file_not_found(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_data_config(tmp_path / "does_not_exist.yaml")


@pytest.mark.parametrize("snr", [3, -3, 12, "6"])
def test_invalid_snr(tmp_path: Path, snr: object) -> None:
    with pytest.raises(ValueError, match="snr_db"):
        load_data_config(_write_yaml(tmp_path, _minimal() | {"snr_db": snr}))


@pytest.mark.parametrize("fraction", [0.0, 0.5, -0.1, 0.9, 1.0])
def test_invalid_val_fraction(tmp_path: Path, fraction: float) -> None:
    with pytest.raises(ValueError, match="val_fraction"):
        load_data_config(_write_yaml(tmp_path, _minimal() | {"val_fraction": fraction}))


def test_invalid_machine_type(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="machine_type"):
        load_data_config(_write_yaml(tmp_path, _minimal() | {"machine_type": "rocket"}))


def test_duplicate_seeds(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="unique"):
        load_data_config(_write_yaml(tmp_path, _minimal() | {"seeds": [0, 0]}))


def test_negative_seed(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="non-negative"):
        load_data_config(_write_yaml(tmp_path, _minimal() | {"seeds": [-1]}))


def test_empty_seeds(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="at least one seed"):
        load_data_config(_write_yaml(tmp_path, _minimal() | {"seeds": []}))


def test_dataclass_is_frozen() -> None:
    config = DataConfig(raw_dir=Path("a"), splits_dir=Path("b"), machine_type="fan", snr_db=0)
    with pytest.raises(FrozenInstanceError):
        config.snr_db = 6  # type: ignore[misc]


def test_direct_dataclass_validation() -> None:
    with pytest.raises(ValueError, match="sample_rate"):
        DataConfig(
            raw_dir=Path("a"),
            splits_dir=Path("b"),
            machine_type="fan",
            snr_db=0,
            sample_rate=0,
        )
