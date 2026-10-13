"""Tests for diagnostic.features.normalize.Standardizer."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from diagnostic.features.normalize import STD_FLOOR, Standardizer


def test_train_round_trip_mean_zero_std_one() -> None:
    rng = np.random.default_rng(0)
    x = rng.standard_normal((200, 8)).astype(np.float32) * 3.0 + 5.0
    standardizer = Standardizer().fit(x, config_hash="abc123")

    z = standardizer.transform(x)
    assert z.dtype == np.float32
    np.testing.assert_allclose(z.mean(axis=0), 0.0, atol=1e-5)
    np.testing.assert_allclose(z.std(axis=0), 1.0, atol=1e-5)
    assert standardizer.config_hash == "abc123"


def test_test_data_is_not_used_for_stats() -> None:
    train = np.zeros((10, 3), dtype=np.float32)
    train[:, 0] = 2.0
    standardizer = Standardizer().fit(train)
    test = np.full((4, 3), 100.0, dtype=np.float32)
    z = standardizer.transform(test)
    # mean is 2.0 for dim 0 -> transformed test value is far from zero
    assert abs(float(z[0, 0])) > 1.0


def test_std_floor_for_constant_column() -> None:
    x = np.ones((5, 2), dtype=np.float32)
    standardizer = Standardizer().fit(x)
    assert standardizer.std is not None
    np.testing.assert_allclose(standardizer.std, STD_FLOOR)
    z = standardizer.transform(x)
    np.testing.assert_allclose(z, 0.0, atol=1e-6)


def test_transform_dimension_mismatch_raises() -> None:
    standardizer = Standardizer().fit(np.zeros((4, 3), dtype=np.float32))
    with pytest.raises(ValueError, match="dimension mismatch"):
        standardizer.transform(np.zeros((4, 5), dtype=np.float32))


def test_transform_before_fit_raises() -> None:
    with pytest.raises(ValueError, match="not fitted"):
        Standardizer().transform(np.zeros((2, 2), dtype=np.float32))


@pytest.mark.parametrize("x", [np.zeros(4, dtype=np.float32), np.zeros((0, 3), dtype=np.float32)])
def test_fit_invalid_raises(x: np.ndarray) -> None:
    with pytest.raises(ValueError):
        Standardizer().fit(x)


def test_json_round_trip(tmp_path: Path) -> None:
    x = np.random.default_rng(3).standard_normal((50, 4)).astype(np.float32)
    original = Standardizer().fit(x, config_hash="deadbeef")
    path = original.to_json(tmp_path / "nested" / "scaler.json")

    restored = Standardizer.from_json(path)
    assert restored.config_hash == "deadbeef"
    np.testing.assert_array_equal(restored.mean, original.mean)
    np.testing.assert_array_equal(restored.std, original.std)
    np.testing.assert_array_equal(restored.transform(x), original.transform(x))


def test_to_json_before_fit_raises(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="not fitted"):
        Standardizer().to_json(tmp_path / "scaler.json")
