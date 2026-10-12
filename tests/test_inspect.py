"""Tests for diagnostic.data.inspect (summary and split CLI)."""

from __future__ import annotations

from pathlib import Path

from diagnostic.data.inspect import build_parser, main, summarise
from diagnostic.data.mimii import Recording


def _make_wavs(base: Path, machine_id: str, n_normal: int, n_abnormal: int) -> None:
    for i in range(n_normal):
        path = base / "6_dB_fan" / machine_id / "normal" / f"n{i:04d}.wav"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"RIFF")
    for i in range(n_abnormal):
        path = base / "6_dB_fan" / machine_id / "abnormal" / f"a{i:04d}.wav"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"RIFF")


def _write_config(tmp_path: Path, raw_dir: Path, seeds: list[int]) -> Path:
    config = tmp_path / "config.yaml"
    config.write_text(
        "\n".join(
            [
                f"raw_dir: {raw_dir.as_posix()}",
                f"splits_dir: {(tmp_path / 'splits').as_posix()}",
                "machine_type: fan",
                "snr_db: 6",
                "val_fraction: 0.1",
                f"seeds: {seeds}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return config


def test_summarise() -> None:
    recordings = [
        Recording(Path("a.wav"), "fan", "id_00", 6, "normal"),
        Recording(Path("b.wav"), "fan", "id_00", 6, "normal"),
        Recording(Path("c.wav"), "fan", "id_00", 6, "abnormal"),
        Recording(Path("d.wav"), "fan", "id_02", 6, "normal"),
    ]
    summary = summarise(recordings)
    assert summary == {"id_00": {"normal": 2, "abnormal": 1}, "id_02": {"normal": 1, "abnormal": 0}}


def test_build_parser() -> None:
    args = build_parser().parse_args(["--config", "c.yaml", "--save-splits"])
    assert args.config == "c.yaml"
    assert args.save_splits is True


def test_main_summary_only(tmp_path: Path, capsys) -> None:
    raw_dir = tmp_path / "mimii"
    _make_wavs(raw_dir, "id_00", n_normal=20, n_abnormal=5)
    config = _write_config(tmp_path, raw_dir, seeds=[0])

    assert main(["--config", str(config)]) == 0
    out = capsys.readouterr().out
    assert "id_00: normal=20 abnormal=5" in out
    assert "total: 25" in out
    assert not (tmp_path / "splits").exists()


def test_main_save_splits(tmp_path: Path, capsys) -> None:
    raw_dir = tmp_path / "mimii"
    _make_wavs(raw_dir, "id_00", n_normal=20, n_abnormal=5)
    config = _write_config(tmp_path, raw_dir, seeds=[0, 1])

    assert main(["--config", str(config), "--save-splits"]) == 0
    splits = tmp_path / "splits"
    assert (splits / "fan_6dB_id_00_seed0.json").is_file()
    assert (splits / "fan_6dB_id_00_seed1.json").is_file()
    assert "train=" in capsys.readouterr().out


def test_main_reports_insufficient_data(tmp_path: Path, capsys) -> None:
    raw_dir = tmp_path / "mimii"
    _make_wavs(raw_dir, "id_00", n_normal=5, n_abnormal=5)
    config = _write_config(tmp_path, raw_dir, seeds=[0])

    assert main(["--config", str(config), "--save-splits"]) == 1
    err = capsys.readouterr().err
    assert "skipped id_00 seed 0" in err
