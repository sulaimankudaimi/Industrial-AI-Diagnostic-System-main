# Features (log-mel + context)

This document describes the F3 feature pipeline. Decisions D5, D7, D14, D15 and
D16 in `docs/design.md` are the source of truth. There is exactly one
preprocessing path, shared by training and inference; audio is never converted
to image files.

## Pipeline

```
waveform (mono, 16 kHz, float32)
  -> log-mel        (n_mels, T)          compute_log_mel
  -> context stack  (T - C + 1, n_mels*C) stack_context
  -> standardize     (rows, n_mels*C)     Standardizer (training stats only)
```

### Log-mel (`diagnostic.features.logmel`)

`mel = |STFT(wave, n_fft, hop_length, center=True)| ** power`, then

```
log_mel = (20 / power) * log10(mel + eps)      # units: dBFS-like, absolute level kept
```

No per-file normalization, no `top_db` clipping, no `ref=`-based dB conversion
(D14). A uniform amplitude change of the input therefore shifts every bin by a
constant offset: doubling a non-silent signal raises the mean by `20*log10(2)`
≈ 6.02 dB. Silence maps to `10*log10(eps)` (finite).

Input validation: 1-D, finite, at least `n_fft` samples, otherwise `ValueError`.

Output `float32`, shape `(n_mels, T)` with `T = 1 + len(wave) // hop_length`.

### Context stacking (`diagnostic.features.context`)

For input `(n_mels, T)` and context `C`, the output is
`(T - C + 1, n_mels * C)`. Row `i` is the time-ordered concatenation

```
[frame_i, frame_{i+1}, ..., frame_{i+C-1}]
```

where each frame contributes its `n_mels` values in mel-bin order. `T < C`
raises `ValueError`.

### Standardization (`diagnostic.features.normalize`)

`Standardizer` stores per-dimension mean and std (std floored at `1e-6`). Fit it
**on the training split only** — never on validation or test (D15). It is
persisted with the feature `config_hash` so stored statistics are tied to the
exact pipeline that produced them.

## Cache (D16)

`cached_log_mel` writes `float32` `.npy` files at

```
<cache_dir>/<config_hash>/<path relative to raw_dir>.npy
```

Writes are atomic (temp file + rename) and a cache hit never recomputes. A
different feature config has a different `config_hash`, hence a different
folder. The cache is never committed (`data/` is git-ignored).

## Configuration

Feature settings live in YAML under `configs/features/` (example:
`configs/features/logmel_default.yaml`) and load into `diagnostic.config.FeatureConfig`:

| Key | Default | Meaning |
|-----|---------|---------|
| `sample_rate` | 16000 | Sampling rate (Hz). |
| `n_fft` | 1024 | STFT window (samples). |
| `hop_length` | 512 | STFT hop (samples). |
| `n_mels` | 64 | Mel channels. |
| `fmin` / `fmax` | 0.0 / 8000.0 | Mel band edges (Hz, `fmax <= sample_rate/2`). |
| `power` | 2.0 | Spectrogram exponent (1.0 or 2.0). |
| `eps` | 1e-10 | Additive floor inside the log (> 0). |
| `context_frames` | 5 | Frames stacked per vector (>= 1). |

`FeatureConfig.config_hash()` is the first 10 hex chars of the SHA-1 of the
canonical sorted JSON.

## CLI

```powershell
# Extract + cache features for one machine ID and seed
.\.venv\Scripts\python.exe -m diagnostic.features.extract `
  --data-config configs/data/mimii_fan_6db.yaml `
  --features-config configs/features/logmel_default.yaml `
  --machine-id id_00 --seed 0

# Quick smoke run on the first N files per split (printed prominently)
.\.venv\Scripts\python.exe -m diagnostic.features.extract `
  --data-config configs/data/mimii_fan_6db.yaml `
  --features-config configs/features/logmel_default.yaml `
  --machine-id id_00 --seed 0 --limit 5

# Pack the cache for this config hash into one file / restore it elsewhere
.\.venv\Scripts\python.exe -m diagnostic.features.extract ... --pack D:\cache\features_fan.tar.gz
.\.venv\Scripts\python.exe -m diagnostic.features.extract --unpack D:\cache\features_fan.tar.gz --cache-dir data/cache/features
```

Options: `--cache-dir` (default `data/cache/features`), `--out-dir` (default
`runs/features`, holds the JSON summary), `--limit`, and the mutually exclusive
`--pack` / `--unpack`. All paths may be absolute; defaults are overridable.

The CLI reads the F2 split manifest for `(machine_type, snr_db, machine_id,
seed)` and prints, per split, the number of files, matrix shape, extraction time,
memory in MB, and the mean/std of pooled log-mel values.
