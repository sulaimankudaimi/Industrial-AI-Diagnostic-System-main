"""File-level train/validation/test splits following design decision D3.

For one machine ID and one seed: the test set is every abnormal recording plus
an equal number of seeded random normal recordings; validation takes a fraction
of the remaining normal recordings (at least one); training gets the rest.
Splits are always at recording-file level and never overlap.
"""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass
from pathlib import Path

from diagnostic.data.mimii import Recording

MIN_TRAIN = 10
_GROUPS: tuple[str, ...] = ("train", "val", "test_normal", "test_abnormal")
_NORMAL_GROUPS: tuple[str, ...] = ("train", "val", "test_normal")


@dataclass(frozen=True)
class Split:
    """A file-level split for one machine ID and one seed.

    Attributes:
        machine_type: Machine class (fan/pump/slider/valve).
        machine_id: Machine identifier (e.g. ``id_00``).
        snr_db: Signal-to-noise ratio in dB.
        seed: Random seed used to sample the test normals.
        val_fraction: Fraction of remaining normals assigned to validation.
        train: Normal recordings used for training.
        val: Normal recordings used for threshold/early-stopping validation.
        test_normal: Normal recordings held out for testing.
        test_abnormal: All abnormal recordings held out for testing.
    """

    machine_type: str
    machine_id: str
    snr_db: int
    seed: int
    val_fraction: float
    train: tuple[Recording, ...]
    val: tuple[Recording, ...]
    test_normal: tuple[Recording, ...]
    test_abnormal: tuple[Recording, ...]

    @property
    def test(self) -> tuple[Recording, ...]:
        """All test recordings (normal followed by abnormal)."""
        return self.test_normal + self.test_abnormal


def _rel(path: str | Path, raw_dir: str | Path) -> str:
    """Return ``path`` relative to ``raw_dir`` with forward slashes."""
    candidate = Path(path)
    try:
        return candidate.relative_to(raw_dir).as_posix()
    except ValueError:
        return candidate.as_posix()


def content_hash(split: Split, raw_dir: str | Path) -> str:
    """Return a short deterministic hash of all file paths in the split.

    Args:
        split: The split to hash.
        raw_dir: Base directory used to make paths relative and portable.

    Returns:
        The first 12 hex characters of the SHA-256 of the sorted relative paths.
    """
    parts: list[str] = []
    for group in _GROUPS:
        for recording in sorted(getattr(split, group), key=lambda r: _rel(r.path, raw_dir)):
            parts.append(_rel(recording.path, raw_dir))
    return hashlib.sha256("\n".join(parts).encode("utf-8")).hexdigest()[:12]


def _assert_disjoint(split: Split) -> None:
    """Raise ``ValueError`` if any recording appears in more than one group."""
    seen: dict[str, str] = {}
    for group in _GROUPS:
        for recording in getattr(split, group):
            key = str(recording.path)
            if key in seen:
                raise ValueError(f"split overlap: {key} in {seen[key]} and {group}")
            seen[key] = group


def make_split(
    recordings: list[Recording],
    machine_id: str,
    seed: int,
    val_fraction: float = 0.1,
) -> Split:
    """Build a file-level split for one machine ID and one seed (D3).

    Args:
        recordings: All recordings (any IDs); only ``machine_id`` is used.
        machine_id: Machine identifier to split.
        seed: Seed controlling which normal recordings enter the test set.
        val_fraction: Fraction of remaining normals used for validation.

    Returns:
        A validated, non-overlapping :class:`Split`.

    Raises:
        ValueError: If ``val_fraction`` is out of range, there are no recordings
            for the ID, or there are not enough normal recordings.
    """
    if not 0.0 < val_fraction < 0.5:
        raise ValueError(f"val_fraction must be in (0, 0.5), got {val_fraction!r}")

    normals = sorted(
        (r for r in recordings if r.machine_id == machine_id and r.label == "normal"),
        key=lambda r: str(r.path).lower(),
    )
    abnormals = sorted(
        (r for r in recordings if r.machine_id == machine_id and r.label == "abnormal"),
        key=lambda r: str(r.path).lower(),
    )
    n_normal, n_abnormal = len(normals), len(abnormals)
    if n_normal == 0 and n_abnormal == 0:
        raise ValueError(f"no recordings found for machine ID {machine_id!r}")
    if n_normal < n_abnormal + MIN_TRAIN + 1:
        raise ValueError(
            f"not enough normal recordings for {machine_id!r}: "
            f"normal={n_normal}, abnormal={n_abnormal}, need normal >= abnormal + {MIN_TRAIN + 1}"
        )

    rng = random.Random(seed)
    shuffled = list(normals)
    rng.shuffle(shuffled)

    test_normal = sorted(shuffled[:n_abnormal], key=lambda r: str(r.path).lower())
    remaining = shuffled[n_abnormal:]
    n_val = max(1, round(val_fraction * len(remaining)))
    val = sorted(remaining[:n_val], key=lambda r: str(r.path).lower())
    train = sorted(remaining[n_val:], key=lambda r: str(r.path).lower())

    if len(train) < MIN_TRAIN:
        raise ValueError(
            f"not enough normal recordings for {machine_id!r} to build a split with "
            f"train >= {MIN_TRAIN}: normal={n_normal}, abnormal={n_abnormal}, "
            f"train={len(train)}, val={len(val)}, test_normal={len(test_normal)}"
        )

    machine_type = normals[0].machine_type if normals else abnormals[0].machine_type
    snr_db = normals[0].snr_db if normals else abnormals[0].snr_db
    split = Split(
        machine_type=machine_type,
        machine_id=machine_id,
        snr_db=snr_db,
        seed=seed,
        val_fraction=val_fraction,
        train=tuple(train),
        val=tuple(val),
        test_normal=tuple(test_normal),
        test_abnormal=tuple(abnormals),
    )
    _assert_disjoint(split)
    return split


def manifest_path(
    splits_dir: str | Path, machine_type: str, snr_db: int, machine_id: str, seed: int
) -> Path:
    """Return the canonical manifest path for a (type, snr, id, seed) split."""
    name = f"{machine_type}_{snr_db}dB_{machine_id}_seed{seed}.json"
    return Path(splits_dir) / name


def save_split(split: Split, raw_dir: str | Path, splits_dir: str | Path) -> Path:
    """Write a split manifest as JSON and return its path.

    Paths are stored relative to ``raw_dir`` so manifests are portable.
    """
    path = manifest_path(splits_dir, split.machine_type, split.snr_db, split.machine_id, split.seed)
    path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "machine_type": split.machine_type,
        "machine_id": split.machine_id,
        "snr_db": split.snr_db,
        "seed": split.seed,
        "val_fraction": split.val_fraction,
        "counts": {group: len(getattr(split, group)) for group in _GROUPS},
        "content_hash": content_hash(split, raw_dir),
        "files": {
            group: [_rel(r.path, raw_dir) for r in getattr(split, group)] for group in _GROUPS
        },
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def load_split(path: str | Path, raw_dir: str | Path) -> Split:
    """Load a split manifest written by :func:`save_split`.

    Args:
        path: Path to the JSON manifest.
        raw_dir: Base directory used to resolve the stored relative paths.

    Returns:
        The reconstructed :class:`Split`.
    """
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    machine_type = payload["machine_type"]
    machine_id = payload["machine_id"]
    snr_db = payload["snr_db"]
    base = Path(raw_dir)

    def _recording(relative: str, label: str) -> Recording:
        return Recording(
            path=base / relative,
            machine_type=machine_type,
            machine_id=machine_id,
            snr_db=snr_db,
            label=label,
        )

    files = payload["files"]
    return Split(
        machine_type=machine_type,
        machine_id=machine_id,
        snr_db=snr_db,
        seed=payload["seed"],
        val_fraction=payload["val_fraction"],
        train=tuple(_recording(rel, "normal") for rel in files["train"]),
        val=tuple(_recording(rel, "normal") for rel in files["val"]),
        test_normal=tuple(_recording(rel, "normal") for rel in files["test_normal"]),
        test_abnormal=tuple(_recording(rel, "abnormal") for rel in files["test_abnormal"]),
    )
