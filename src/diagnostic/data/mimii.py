"""Discovery of MIMII recordings on disk.

The MIMII zips extract to a hierarchy that may include extra nesting; this
module locates every ``.wav`` file that sits under an ``id_XX/{normal,abnormal}``
folder for a given machine type and SNR.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

_ID_PATTERN = re.compile(r"^id_\d+$")
_LABELS: tuple[str, ...] = ("normal", "abnormal")


@dataclass(frozen=True)
class Recording:
    """A single MIMII recording file.

    Attributes:
        path: Path to the ``.wav`` file.
        machine_type: Machine class (fan/pump/slider/valve).
        machine_id: Machine identifier parsed from the ``id_XX`` folder.
        snr_db: Signal-to-noise ratio of the recording in dB.
        label: Either ``"normal"`` or ``"abnormal"``.
    """

    path: Path
    machine_type: str
    machine_id: str
    snr_db: int
    label: str


def machine_type_dir(raw_dir: str | Path, snr_db: int, machine_type: str) -> Path:
    """Return the directory holding one machine type at one SNR.

    Args:
        raw_dir: Root directory of the extracted MIMII recordings.
        snr_db: Signal-to-noise ratio in dB (e.g. ``6``).
        machine_type: Machine class (e.g. ``"fan"``).

    Returns:
        ``<raw_dir>/<snr_db>_dB_<machine_type>``.
    """
    return Path(raw_dir) / f"{snr_db}_dB_{machine_type}"


def _extract_machine_id(path: Path) -> str:
    """Return the ``id_XX`` component of ``path`` or raise ``ValueError``."""
    for part in path.parts:
        if _ID_PATTERN.match(part):
            return part
    raise ValueError(f"could not parse an id_XX folder from path: {path}")


def discover_recordings(raw_dir: str | Path, machine_type: str, snr_db: int) -> list[Recording]:
    """Discover all recordings for one machine type at one SNR.

    Files are matched with ``**/id_*/{normal,abnormal}/*.wav`` so that any
    internal zip nesting is tolerated. The result is deterministically sorted
    by ``(machine_id, label, path)`` and never truncated.

    Args:
        raw_dir: Root directory of the extracted MIMII recordings.
        machine_type: Machine class (e.g. ``"fan"``).
        snr_db: Signal-to-noise ratio in dB (e.g. ``6``).

    Returns:
        A sorted list of :class:`Recording`.

    Raises:
        FileNotFoundError: If the machine type directory does not exist.
        ValueError: If no recordings are found or a path lacks an ``id_XX`` folder.
    """
    base = machine_type_dir(raw_dir, snr_db, machine_type)
    if not base.is_dir():
        raise FileNotFoundError(f"machine type directory not found: {base}")

    recordings: list[Recording] = []
    for label in _LABELS:
        matches = sorted(base.glob(f"**/id_*/{label}/*.wav"), key=lambda p: str(p).lower())
        for wav in matches:
            recordings.append(
                Recording(
                    path=wav,
                    machine_type=machine_type,
                    machine_id=_extract_machine_id(wav),
                    snr_db=snr_db,
                    label=label,
                )
            )

    if not recordings:
        raise ValueError(f"no .wav recordings found under {base}")

    return sorted(recordings, key=lambda r: (r.machine_id, r.label, str(r.path).lower()))
