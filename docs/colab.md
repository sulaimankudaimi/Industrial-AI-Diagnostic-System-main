# Colab runner (Google Colab, Python 3.13)

## Purpose and architecture

The laptop (Windows, Python 3.11) is for development, light tests and later
inference/hardware work. Heavy, long-running jobs run as commands in a Colab
notebook on Google's free CPU/GPU instances, which ship Python 3.13.

What runs where:

- Colab: download, extract, inspect, split creation, feature extraction,
  training, evaluation.
- Google Drive (only as external storage): split manifests, packed feature
  cache, models, metrics.
- Laptop: development, unit tests, `ruff`, and future inference/hardware work.

Sanitised copy of the data/features configs live under `configs/` and are
referenced by path inside the repo; the notebook only points the tool commands
at Colab-local and Drive paths.

## Rules

- The notebook holds commands only; it never contains pipeline logic. Logic
  lives in the repo under `src/diagnostic` and is exercised through the CLI.
- Work on the local Colab disk (`/content`) for speed; copy results to Google
  Drive as single `tar.gz` files. Never copy raw audio to Drive.
- Record the library versions for every run, together with the repo commit,
  the tool versions and the config file contents used.
- Reuse the clone with `git -C /content/repo pull` on later sessions instead
  of cloning again.
- The first run is on seed 0; other seeds reuse the same downloaded data.

## Runner cells

Copy the cells below, in order, into a Colab notebook.

### Cell 1 - mount Drive and check environments

```python
from google.colab import drive

drive.mount("/content/drive")
```

```python
!python --version
!df -h /content | tail -1
!df -h /content/drive | tail -1
```

### Cell 2 - clone the repo and install

```text
!git clone https://github.com/sulaimankudaimi/Industrial-AI-Diagnostic-System-main.git /content/repo
%cd /content/repo
!pip install -q -e . pytest
```

Note: on a later session use `!git -C /content/repo pull` instead of cloning
again.

### Cell 3 - smoke test

```python
!python -m pytest -q -m "not ml"
```

### Cell 4 - point the data config at Colab disk and Drive

```python
import yaml
from pathlib import Path

cfg = yaml.safe_load(Path("configs/data/mimii_fan_6db.yaml").read_text())
cfg["raw_dir"] = "/content/data/raw/mimii"
cfg["splits_dir"] = "/content/drive/MyDrive/industrial_ai/splits"
Path("/content/colab_data.yaml").write_text(yaml.safe_dump(cfg))
print(Path("/content/colab_data.yaml").read_text())
```

### Cell 5 - list and fetch the MIMII fan zips

```python
!python -m diagnostic.data.download --config /content/colab_data.yaml --list
```

```python
!python -m diagnostic.data.download --config /content/colab_data.yaml --download --extract --delete-zip
```

### Cell 6 - verify the harvest and build split manifests

```python
!find /content/data/raw/mimii -name "*.wav" | wc -l
!python -m diagnostic.data.inspect --config /content/colab_data.yaml --save-splits
!ls /content/drive/MyDrive/industrial_ai/splits | wc -l
```

### Cell 7 - extract log-mel features and pack the cache for Drive

```python
!python -m diagnostic.features.extract \
  --data-config /content/colab_data.yaml \
  --features-config configs/features/logmel_default.yaml \
  --machine-id id_00 --seed 0 \
  --cache-dir /content/cache/features \
  --out-dir /content/runs/features
```

```python
!python -m diagnostic.features.extract \
  --data-config /content/colab_data.yaml \
  --features-config configs/features/logmel_default.yaml \
  --machine-id id_00 --seed 0 \
  --cache-dir /content/cache/features \
  --out-dir /content/runs/features \
  --pack /content/drive/MyDrive/industrial_ai/features_<config_hash>.tar.gz
```

The second run reuses the file cache and only writes the archive; replace
`<config_hash>` with the hash printed by the first run. Training and evaluation
cells (F4+) will be added when their CLIs exist.
