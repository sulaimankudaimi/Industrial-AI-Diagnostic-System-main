"""Training-only feature standardization (decision D15).

A :class:`Standardizer` stores the per-dimension mean and standard deviation of
the feature vectors. It must be fitted **only on the training split** (never on
validation or test data) so that evaluation reflects genuine generalization.
The statistics are stored together with the feature ``config_hash`` so a stored
standardizer can be tied to the exact feature pipeline that produced it.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

STD_FLOOR = 1e-6


@dataclass
class Standardizer:
    """Per-dimension feature standardizer.

    Attributes:
        mean: Per-dimension means, shape ``(D,)``; ``None`` until fitted.
        std: Per-dimension standard deviations (floored at ``STD_FLOOR``), shape
            ``(D,)``; ``None`` until fitted.
        config_hash: Feature config hash this standardizer was fitted for.
    """

    mean: np.ndarray | None = None
    std: np.ndarray | None = None
    config_hash: str = ""

    def fit(self, x: np.ndarray, config_hash: str = "") -> Standardizer:
        """Compute per-dimension mean and std from training vectors ``x``.

        Use training vectors only; fitting on validation or test data leaks
        information and is forbidden (decision D15).

        Args:
            x: 2-D float array of shape ``(N, D)`` (N > 0).
            config_hash: Feature config hash associated with ``x``.

        Returns:
            ``self``, for chaining.

        Raises:
            ValueError: If ``x`` is not 2-D or has no rows.
        """
        arr = np.asarray(x, dtype=np.float64)
        if arr.ndim != 2:
            raise ValueError(f"X must be 2-D, got shape {arr.shape}")
        if arr.shape[0] == 0:
            raise ValueError("cannot fit a Standardizer on zero rows")

        self.mean = arr.mean(axis=0)
        self.std = np.maximum(arr.std(axis=0), STD_FLOOR)
        self.config_hash = config_hash
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        """Standardize ``x`` using the stored statistics.

        Args:
            x: 2-D float array of shape ``(N, D)`` with the fitted ``D``.

        Returns:
            ``float32`` array of shape ``(N, D)``.

        Raises:
            ValueError: If the standardizer is not fitted or the dimension of
                ``x`` does not match the fitted dimension.
        """
        if self.mean is None or self.std is None:
            raise ValueError("Standardizer is not fitted")
        arr = np.asarray(x, dtype=np.float32)
        if arr.ndim != 2:
            raise ValueError(f"X must be 2-D, got shape {arr.shape}")
        if arr.shape[1] != self.mean.size:
            raise ValueError(
                f"dimension mismatch: X has {arr.shape[1]} columns, expected {self.mean.size}"
            )
        standardized = (arr - self.mean.astype(np.float32)) / self.std.astype(np.float32)
        return np.ascontiguousarray(standardized, dtype=np.float32)

    def to_json(self, path: str | Path) -> Path:
        """Write mean, std and ``config_hash`` to ``path`` as JSON."""
        if self.mean is None or self.std is None:
            raise ValueError("Standardizer is not fitted")
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "config_hash": self.config_hash,
            "mean": self.mean.tolist(),
            "std": self.std.tolist(),
        }
        destination.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        return destination

    @classmethod
    def from_json(cls, path: str | Path) -> Standardizer:
        """Load a standardizer previously written by :meth:`to_json`."""
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(
            mean=np.asarray(payload["mean"], dtype=np.float64),
            std=np.asarray(payload["std"], dtype=np.float64),
            config_hash=payload.get("config_hash", ""),
        )
