"""Audio loading for the diagnostic pipeline.

A single, explicit loading path: files are read with ``soundfile`` and returned
as mono ``float32`` arrays sampled at the expected rate.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf


def load_wav(path: str | Path, channel: int = 0, expected_sr: int = 16000) -> np.ndarray:
    """Load one channel of a WAV file as a mono float32 array.

    Args:
        path: Path to the ``.wav`` file.
        channel: Channel index to select (0-based).
        expected_sr: Required sampling rate in Hz; mismatches raise.

    Returns:
        A contiguous 1-D ``float32`` numpy array (unitless, normalised amplitude).

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file cannot be read, is empty, has the wrong sample
            rate, or the channel index is out of range.
    """
    file_path = Path(path)
    if not file_path.is_file():
        raise FileNotFoundError(f"audio file not found: {file_path}")

    try:
        data, sample_rate = sf.read(file_path, always_2d=True, dtype="float32")
    except Exception as exc:  # soundfile raises various error types
        raise ValueError(f"could not read audio file {file_path}: {exc}") from exc

    if sample_rate != expected_sr:
        raise ValueError(
            f"unexpected sample rate {sample_rate} Hz in {file_path}; expected {expected_sr} Hz"
        )
    frame_count, channel_count = data.shape
    if frame_count == 0:
        raise ValueError(f"audio file is empty: {file_path}")
    if not 0 <= channel < channel_count:
        raise ValueError(
            f"channel {channel} out of range for {file_path} with {channel_count} channel(s)"
        )

    return np.ascontiguousarray(data[:, channel], dtype=np.float32)
