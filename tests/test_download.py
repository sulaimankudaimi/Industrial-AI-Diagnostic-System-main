"""Tests for diagnostic.data.download (Zenodo listing and file transport)."""

from __future__ import annotations

import hashlib
import zipfile
from pathlib import Path

import pytest

from diagnostic.data.download import (
    RemoteFile,
    build_parser,
    download_file,
    extract_zip,
    md5_of,
    needs_download,
    parse_record,
    resume_offset,
    select_files,
)


def _record_payload() -> dict:
    return {
        "files": [
            {
                "key": "6_dB_fan.zip",
                "size": 123,
                "checksum": "md5:aaaa",
                "links": {"self": "https://example.test/6_dB_fan.zip"},
            },
            {
                "key": "-6_dB_pump.zip",
                "size": "456",
                "checksum": "md5:bbbb",
                "links": {"content": "https://example.test/-6_dB_pump.zip"},
            },
        ]
    }


def test_parse_record_sorted_and_parsed() -> None:
    files = parse_record(_record_payload())
    assert [f.key for f in files] == ["-6_dB_pump.zip", "6_dB_fan.zip"]
    fan = files[1]
    assert fan.size == 123
    assert fan.md5 == "aaaa"
    assert fan.url == "https://example.test/6_dB_fan.zip"
    pump = files[0]
    assert pump.size == 456
    assert pump.url == "https://example.test/-6_dB_pump.zip"


def test_parse_record_without_files_raises() -> None:
    with pytest.raises(ValueError, match="no 'files' list"):
        parse_record({"metadata": {}})


def test_md5_of_known_content(tmp_path: Path) -> None:
    path = tmp_path / "x.bin"
    path.write_bytes(b"hello")
    assert md5_of(path) == hashlib.md5(b"hello").hexdigest()


def test_resume_offset(tmp_path: Path) -> None:
    assert resume_offset(tmp_path / "missing.bin") == 0
    path = tmp_path / "partial.bin"
    path.write_bytes(b"12345")
    assert resume_offset(path) == 5


def test_needs_download_states(tmp_path: Path) -> None:
    path = tmp_path / "f.bin"
    remote = RemoteFile("f.bin", 5, "md5:" + hashlib.md5(b"hello").hexdigest(), "u")
    assert needs_download(path, remote) is True  # missing
    path.write_bytes(b"")
    assert needs_download(path, remote) is True  # empty
    path.write_bytes(b"hello")
    assert needs_download(path, remote) is False  # valid
    assert needs_download(path, remote, force=True) is True  # forced
    path.write_bytes(b"other")
    assert needs_download(path, remote) is True  # mismatch


def test_select_files() -> None:
    files = parse_record(_record_payload())
    assert [f.key for f in select_files(files, "pump", -6)] == ["-6_dB_pump.zip"]
    with pytest.raises(ValueError, match="no file"):
        select_files(files, "fan", 0)


def _remote_for(path: Path) -> RemoteFile:
    return RemoteFile(path.name, path.stat().st_size, f"md5:{md5_of(path)}", path.as_uri())


def test_download_file_full_and_skip_and_force(tmp_path: Path) -> None:
    source = tmp_path / "src.bin"
    source.write_bytes(b"payload-data" * 100)
    remote = _remote_for(source)

    dest = tmp_path / "nested" / "dst.bin"
    download_file(remote, dest)
    assert dest.read_bytes() == source.read_bytes()

    stamp = dest.stat().st_mtime_ns
    download_file(remote, dest)  # valid -> no-op
    assert dest.stat().st_mtime_ns == stamp

    download_file(remote, dest, force=True)
    assert dest.read_bytes() == source.read_bytes()


def test_download_file_checksum_mismatch_raises(tmp_path: Path) -> None:
    source = tmp_path / "src.bin"
    source.write_bytes(b"payload")
    remote = RemoteFile("dst.bin", source.stat().st_size, "md5:" + "0" * 32, source.as_uri())
    with pytest.raises(ValueError, match="checksum mismatch"):
        download_file(remote, tmp_path / "dst.bin")


def test_extract_zip(tmp_path: Path) -> None:
    archive = tmp_path / "6_dB_fan.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("6_dB_fan/id_00/normal/a.wav", b"RIFF")

    raw_dir = tmp_path / "mimii"
    names = extract_zip(archive, raw_dir)

    assert "6_dB_fan/id_00/normal/a.wav" in names
    assert (raw_dir / "6_dB_fan" / "id_00" / "normal" / "a.wav").is_file()


def test_build_parser_defaults() -> None:
    args = build_parser().parse_args(["--config", "c.yaml"])
    assert args.config == "c.yaml"
    assert args.download is False
    assert args.list is False
    assert args.extract is False
    assert args.all is False
    assert args.record_id == 3384388
