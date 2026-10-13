"""Tests for diagnostic.features.context."""

from __future__ import annotations

import numpy as np
import pytest

from diagnostic.features.context import stack_context


def test_exact_ordering_on_ramp() -> None:
    n_mels, n_frames, context = 3, 6, 3
    log_mel = np.arange(n_mels * n_frames, dtype=np.float32).reshape(n_mels, n_frames)
    stacked = stack_context(log_mel, context)

    assert stacked.shape == (n_frames - context + 1, n_mels * context)
    expected_row0 = np.concatenate([log_mel[:, 0], log_mel[:, 1], log_mel[:, 2]])
    expected_row1 = np.concatenate([log_mel[:, 1], log_mel[:, 2], log_mel[:, 3]])
    np.testing.assert_array_equal(stacked[0], expected_row0)
    np.testing.assert_array_equal(stacked[1], expected_row1)


def test_dtype_and_contiguity() -> None:
    log_mel = np.random.default_rng(1).standard_normal((4, 10)).astype(np.float32)
    stacked = stack_context(log_mel, 2)
    assert stacked.dtype == np.float32
    assert stacked.flags["C_CONTIGUOUS"]


def test_context_one_is_transpose() -> None:
    log_mel = np.random.default_rng(2).standard_normal((5, 8)).astype(np.float32)
    np.testing.assert_array_equal(stack_context(log_mel, 1), log_mel.T)


@pytest.mark.parametrize(
    ("log_mel", "context"),
    [
        (np.zeros((4, 2), dtype=np.float32), 3),  # T < C
        (np.zeros(5, dtype=np.float32), 2),  # not 2-D
        (np.zeros((4, 5), dtype=np.float32), 0),  # C < 1
    ],
)
def test_error_cases(log_mel: np.ndarray, context: int) -> None:
    with pytest.raises(ValueError):
        stack_context(log_mel, context)


def test_context_equals_frames_is_allowed() -> None:
    log_mel = np.arange(12, dtype=np.float32).reshape(3, 4)
    stacked = stack_context(log_mel, 4)
    assert stacked.shape == (1, 12)
    np.testing.assert_array_equal(stacked[0], log_mel.T.reshape(-1))
