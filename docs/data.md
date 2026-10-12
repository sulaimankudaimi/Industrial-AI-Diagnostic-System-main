# Data layer (MIMII)

This document describes the F2 data pipeline: where recordings live, how they
are discovered, loaded, split, and how to obtain the dataset. Design decisions
D11-D13 in `docs/design.md` are the source of truth.

## Raw layout (D11)

MIMII zips are extracted under a single root (default `data/raw/mimii`):

```
data/raw/mimii/
  6_dB_fan/
    id_00/
      normal/*.wav
      abnormal/*.wav
    id_02/ ...
  -6_dB_fan/ ...
  0_dB_fan/ ...
```

Recordings are 10 s, 16 kHz, 8 channels. The pipeline selects channel 0. The
zip's internal nesting is tolerated: discovery matches
`**/id_*/{normal,abnormal}/*.wav`, always sorts the result, and never truncates.

## Configuration

All tunables live in YAML under `configs/data/` (example:
`configs/data/mimii_fan_6db.yaml`) and load into a frozen `diagnostic.config.DataConfig`:

| Key | Meaning |
|-----|---------|
| `raw_dir` | Root of extracted recordings. |
| `splits_dir` | Where split manifests are written. |
| `machine_type` | `fan` \| `pump` \| `slider` \| `valve`. |
| `snr_db` | `-6` \| `0` \| `6`. |
| `sample_rate` | Expected rate in Hz (default 16000). |
| `channel` | Channel index to select (default 0). |
| `val_fraction` | Fraction of remaining normals for validation, in (0, 0.5). |
| `seeds` | Explicit, non-negative, unique seeds. |

## Modules

- `diagnostic.data.mimii` - `Recording`, `machine_type_dir`, `discover_recordings`.
- `diagnostic.data.audio` - `load_wav(path, channel, expected_sr)` -> mono `float32`.
- `diagnostic.data.splits` - `make_split`, `save_split`, `load_split`, `content_hash`.
- `diagnostic.data.download` - Zenodo listing and resumable, MD5-verified downloads.
- `diagnostic.data.inspect` - inventory and split-manifest CLI.

## Splits (D3, D13)

For one machine ID and one seed: the test set is every abnormal recording plus an
equal number of seeded random normal recordings; validation takes at least one
normal recording (about `val_fraction` of the remainder); training gets the rest.
Splits are strictly file-level and never overlap. A split is refused when fewer
than 11 normals remain after the test sample.

Manifests are JSON files in `splits_dir` named
`<type>_<snr>dB_<id>_seed<seed>.json`, storing paths relative to `raw_dir`,
per-group counts, the seed, and a 12-char SHA-256 content hash.

## Commands

```powershell
# Inventory (no network, needs extracted data)
.\.venv\Scripts\python.exe -m diagnostic.data.inspect --config configs/data/mimii_fan_6db.yaml

# Inventory + write split manifests for every configured seed
.\.venv\Scripts\python.exe -m diagnostic.data.inspect --config configs/data/mimii_fan_6db.yaml --save-splits

# List remote files for the configured machine_type, all SNRs (one HTTPS
# metadata request; downloads nothing)
.\.venv\Scripts\python.exe -m diagnostic.data.download --config configs/data/mimii_fan_6db.yaml --list

# Opt-in download for the configured type/SNR, then extract
.\.venv\Scripts\python.exe -m diagnostic.data.download --config configs/data/mimii_fan_6db.yaml --download --extract

# Same, but delete the zip after a successful extraction
.\.venv\Scripts\python.exe -m diagnostic.data.download --config configs/data/mimii_fan_6db.yaml --download --extract --delete-zip

# Download every zip in the record
.\.venv\Scripts\python.exe -m diagnostic.data.download --config configs/data/mimii_fan_6db.yaml --download --all
```

`--list` runs on its own and always exits without downloading; `--download` is
the only flag that fetches data. Downloads are resumable (HTTP `Range`) and
verified against the MD5 from the Zenodo metadata. `--delete-zip` is refused
unless `--extract` is also given, and only deletes after extraction succeeds.

## Tests

Tests use synthetic audio fixtures created in `tmp_path`; they never touch the
network or the real dataset. Download transport is exercised over `file://`
URLs.
