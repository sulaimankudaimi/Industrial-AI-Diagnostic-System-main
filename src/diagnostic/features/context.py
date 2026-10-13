"""Context stacking of consecutive log-mel frames (decision D5)."""

from __future__ import annotations

import numpy as np


def stack_context(log_mel: np.ndarray, context_frames: int) -> np.ndarray:
    """Stack ``context_frames`` consecutive frames into one vector per position.

    For a log-mel array of shape ``(n_mels, T)`` and context ``C`` the output has
    shape ``(T - C + 1, n_mels * C)``. Row ``i`` is the time-ordered
    concatenation ``[frame_i, frame_{i+1}, ..., frame_{i+C-1}]`` where each frame
    contributes its ``n_mels`` values in mel-bin order.

    Args:
        log_mel: 2-D log-mel array of shape ``(n_mels, T)``.
        context_frames: Number of consecutive frames per vector (``>= 1``).

    Returns:
        ``float32`` array of shape ``(T - context_frames + 1, n_mels * context_frames)``.

    Raises:
        ValueError: If ``log_mel`` is not 2-D, ``context_frames < 1``, or
            ``T < context_frames``.
    """
    features = np.asarray(log_mel)
    if features.ndim != 2:
        raise ValueError(f"log_mel must be 2-D, got shape {features.shape}")
    if context_frames < 1:
        raise ValueError(f"context_frames must be >= 1, got {context_frames}")
    n_mels, n_frames = features.shape
    if n_frames < context_frames:
        raise ValueError(f"not enough frames: T={n_frames} < context_frames={context_frames}")

    windows = np.lib.stride_tricks.sliding_window_view(features, context_frames, axis=1)
    windows = np.moveaxis(windows, 1, 0)  # (rows, n_mels, C)
    windows = np.transpose(windows, (0, 2, 1))  # (rows, C, n_mels)
    stacked = windows.reshape(n_frames - context_frames + 1, n_mels * context_frames)
    return np.ascontiguousarray(stacked, dtype=np.float32)
