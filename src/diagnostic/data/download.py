"""Opt-in, resumable downloader for the MIMII dataset from Zenodo (decision D12).

Nothing is downloaded unless ``--download`` is passed. Files are resumed with
HTTP ``Range`` requests and verified against the MD5 checksum from the Zenodo
record metadata.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path

from diagnostic.config import load_data_config

ZENODO_RECORD_ID = 3384388
ZENODO_API = "https://zenodo.org/api/records/{record_id}"
_CHUNK_SIZE = 1 << 20


@dataclass(frozen=True)
class RemoteFile:
    """A downloadable file described by the Zenodo record metadata.

    Attributes:
        key: File name (e.g. ``6_dB_fan.zip``).
        size: Size in bytes.
        checksum: Zenodo checksum string (e.g. ``md5:abc...``).
        url: HTTPS URL to download the file from.
    """

    key: str
    size: int
    checksum: str
    url: str

    @property
    def md5(self) -> str:
        """Return the MD5 hex digest parsed from :attr:`checksum`."""
        return self.checksum.split(":", 1)[-1].strip()


def parse_record(payload: dict, record_id: int = ZENODO_RECORD_ID) -> list[RemoteFile]:
    """Parse a Zenodo record payload into a sorted list of :class:`RemoteFile`.

    Args:
        payload: Parsed JSON body of ``https://zenodo.org/api/records/<id>``.
        record_id: Record identifier (kept for context in error messages).

    Returns:
        Files sorted by name.

    Raises:
        ValueError: If the payload has no ``files`` list.
    """
    entries = payload.get("files")
    if entries is None:
        raise ValueError(f"record {record_id} payload has no 'files' list")

    files: list[RemoteFile] = []
    for entry in entries:
        links = entry.get("links", {})
        url = links.get("self") or links.get("content") or links.get("download") or ""
        files.append(
            RemoteFile(
                key=entry["key"],
                size=int(entry.get("size", 0)),
                checksum=entry.get("checksum", ""),
                url=url,
            )
        )
    return sorted(files, key=lambda remote: remote.key)


def fetch_record(record_id: int = ZENODO_RECORD_ID, timeout: float = 60.0) -> dict:
    """Fetch and parse the Zenodo record metadata (performs a network request)."""
    url = ZENODO_API.format(record_id=record_id)
    with urllib.request.urlopen(url, timeout=timeout) as response:  # noqa: S310
        return json.loads(response.read().decode("utf-8"))


def md5_of(path: str | Path) -> str:
    """Return the MD5 hex digest of a file, read in chunks."""
    digest = hashlib.md5()  # noqa: S324
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(_CHUNK_SIZE), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resume_offset(dest: str | Path) -> int:
    """Return the number of bytes already present at ``dest`` (0 if absent)."""
    path = Path(dest)
    return path.stat().st_size if path.is_file() else 0


def needs_download(dest: str | Path, remote: RemoteFile, force: bool = False) -> bool:
    """Return ``True`` if ``dest`` is missing, empty, forced, or checksum-mismatched."""
    path = Path(dest)
    if force or not path.is_file() or path.stat().st_size == 0:
        return True
    return md5_of(path) != remote.md5


def download_file(
    remote: RemoteFile, dest: str | Path, force: bool = False, timeout: float = 120.0
) -> Path:
    """Download ``remote`` to ``dest``, resuming and verifying the checksum.

    Args:
        remote: Remote file description.
        dest: Destination path.
        force: If ``True``, download from scratch even if ``dest`` is valid.
        timeout: Socket timeout in seconds.

    Returns:
        The destination path.

    Raises:
        ValueError: If the downloaded file fails MD5 verification.
    """
    destination = Path(dest)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not needs_download(destination, remote, force):
        return destination

    offset = 0 if force else resume_offset(destination)
    request = urllib.request.Request(remote.url)  # noqa: S310
    if offset:
        request.add_header("Range", f"bytes={offset}-")

    with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310
        status = getattr(response, "status", 200)
        mode = "ab" if offset and status == 206 else "wb"
        with destination.open(mode) as handle:
            while True:
                chunk = response.read(_CHUNK_SIZE)
                if not chunk:
                    break
                handle.write(chunk)

    actual = md5_of(destination)
    if actual != remote.md5:
        raise ValueError(
            f"checksum mismatch for {destination}: expected {remote.md5}, got {actual}"
        )
    return destination


def select_files(files: list[RemoteFile], machine_type: str, snr_db: int) -> list[RemoteFile]:
    """Return the ``<snr>_dB_<type>.zip`` entry, or raise ``ValueError``."""
    wanted = f"{snr_db}_dB_{machine_type}.zip"
    matches = [remote for remote in files if remote.key == wanted]
    if not matches:
        available = ", ".join(remote.key for remote in files) or "<none>"
        raise ValueError(f"no file {wanted!r} in record; available: {available}")
    return matches


def extract_zip(zip_path: str | Path, raw_dir: str | Path) -> list[str]:
    """Extract ``zip_path`` into ``raw_dir`` and return the member names."""
    destination = Path(raw_dir)
    destination.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as archive:
        archive.extractall(destination)
        return archive.namelist()


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser for the download CLI."""
    parser = argparse.ArgumentParser(
        prog="diagnostic.data.download",
        description="List and (opt-in) download MIMII zip files from Zenodo.",
    )
    parser.add_argument("--config", required=True, help="Path to a data config YAML file.")
    parser.add_argument("--record-id", type=int, default=ZENODO_RECORD_ID, help="Zenodo record id.")
    parser.add_argument("--list", action="store_true", help="List remote files and exit.")
    parser.add_argument(
        "--download", action="store_true", help="Download files (opt-in; off by default)."
    )
    parser.add_argument(
        "--all", action="store_true", help="Download every zip, not just the config."
    )
    parser.add_argument("--extract", action="store_true", help="Extract zips into raw_dir.")
    parser.add_argument("--force", action="store_true", help="Re-download even if valid.")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Entry point for ``python -m diagnostic.data.download``."""
    args = build_parser().parse_args(argv)
    config = load_data_config(args.config)
    files = parse_record(fetch_record(args.record_id), args.record_id)

    if args.all:
        selected = [remote for remote in files if remote.key.endswith(".zip")]
    else:
        selected = select_files(files, config.machine_type, config.snr_db)

    print(f"record {args.record_id}: {len(files)} file(s); selected {len(selected)}")
    for remote in selected:
        print(f"  {remote.key}\t{remote.size} bytes\tmd5:{remote.md5}")

    if not args.download:
        print("dry run (pass --download to actually fetch)")
        return 0

    config.raw_dir.mkdir(parents=True, exist_ok=True)
    for remote in selected:
        destination = download_file(remote, config.raw_dir / remote.key, force=args.force)
        print(f"downloaded {destination}")
        if args.extract:
            extract_zip(destination, config.raw_dir)
            print(f"extracted {destination.name} into {config.raw_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
