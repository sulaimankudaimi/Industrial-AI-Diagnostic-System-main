"""Log-mel spectrogram extraction (decisions D5 and D7).

The single preprocessing entry point shared by training and inference. Absolute
level is preserved: no per-file normalization, no ``top_db`` clipping and no
reference-based dB conversion (decision D14).
"""

from __future__ import annotations

import librosa
import numpy as np

from diagnostic.config import FeatureConfig


def compute_log_mel(wave: np.ndarray, config: FeatureConfig) -> np.ndarray:
    """Compute a log-mel spectrogram with the absolute level preserved.

    The mel spectrogram is ``mel = |STFT|**power`` (``center=True``) and the
    output is ``(20 / power) * log10(mel + eps)``. Because ``eps`` is added in
    linear units and no per-file scaling is applied, a uniform amplitude change
    of the waveform shifts every bin by a constant dB offset.

    Args:
        wave: 1-D mono waveform in an arbitrary linear unit (must be finite).
        config: Feature pipeline configuration.

    Returns:
        ``float32`` array of shape ``(n_mels, n_frames)`` where
        ``n_frames = 1 + len(wave) // hop_length`` (``center=True``).

    Raises:
        ValueError: If ``wave`` is empty or not 1-D, contains non-finite values,
            or has fewer than ``n_fft`` samples.
    """
    samples = np.asarray(wave)
    if samples.ndim != 1:
        raise ValueError(f"wave must be 1-D, got shape {samples.shape}")
    if samples.size == 0:
        raise ValueError("wave must not be empty")
    if not np.all(np.isfinite(samples)):
        raise ValueError("wave contains non-finite values (NaN or inf)")
    if samples.size < config.n_fft:
        raise ValueError(f"wave too short: {samples.size} samples < n_fft={config.n_fft}")

    mel = librosa.feature.melspectrogram(
        y=samples.astype(np.float32, copy=False),
        sr=config.sample_rate,
        n_fft=config.n_fft,
        hop_length=config.hop_length,
        n_mels=config.n_mels,
        fmin=config.fmin,
        fmax=config.fmax,
        power=config.power,
        center=True,
    )
    log_mel = (20.0 / config.power) * np.log10(mel + config.eps)
    return np.ascontiguousarray(log_mel, dtype=np.float32)
