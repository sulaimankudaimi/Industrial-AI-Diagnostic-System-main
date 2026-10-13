# AGENTS.md - Industrial AI Diagnostic System

## Mission
Acoustic anomaly detection for rotating industrial machines. A reproducible, tested system evaluated first on the MIMII dataset, then on a real machine (first target: a fan) using our own recordings.

## Source of truth
- docs/design.md holds all design decisions. Do not change a decision without an explicit instruction from the lead engineer; if you think one is wrong, say so and stop.
- legacy/ is frozen reference code. Never edit it, never import from it.

## Environment
- Windows 10, PowerShell, no admin rights, CPU only. Python 3.11 venv at .venv; always call .\.venv\Scripts\python.exe.
- Supported Python 3.11-3.13; Colab runs 3.13.
- Windows username is non-ASCII. Before running tools set: PYTHONUTF8=1, PIP_CACHE_DIR, NUMBA_CACHE_DIR, KERAS_HOME, MPLCONFIGDIR, PRE_COMMIT_HOME (all under D:\cache).

## Commands
- Install:  python -m pip install -e ".[dev,ml,app]"
- Lint:     python -m ruff check .   and   python -m ruff format --check .
- Test:     python -m pytest -q   (CI runs: pytest -m "not ml")
- Hooks:    python -m pre_commit run --all-files
- Data:     python -m diagnostic.data.inspect  --config configs/data/mimii_fan_6db.yaml [--save-splits]
- Fetch:    python -m diagnostic.data.download --config configs/data/mimii_fan_6db.yaml [--list | --download [--all] [--extract [--delete-zip]]]
- Features: python -m diagnostic.features.extract --data-config configs/data/mimii_fan_6db.yaml --features-config configs/features/logmel_default.yaml --machine-id id_00 --seed 0 [--limit N] [--cache-dir DIR] [--out-dir DIR] [--pack PATH | --unpack PATH]

## Architecture rules
- src layout, package "diagnostic": data, features, models, training, inference.
- ONE preprocessing code path (diagnostic.features) shared by training and inference. Never convert audio to image files.
- All tunables live in YAML files under configs/, loaded into typed objects. No magic numbers in code.
- Notebooks are for exploration only; no logic that the pipeline depends on.
- Splits are done at recording-file level, never at frame/patch level. Seeds are always explicit.
- Data, models and run outputs (data/, models/, runs/) are never committed.
- Core package must import without tensorflow; TensorFlow/Keras is confined to diagnostic.models, training and inference.

## Code standards
- Type hints on public functions, docstrings (what/why, units), small functions, no dead code.
- Every module gets tests; synthetic audio fixtures only (no dataset needed in tests).
- Dependencies are added to pyproject.toml, never installed ad hoc.

## Workflow rules
- One task per branch, small conventional commits, never push without instruction.
- Before every commit: ruff check, ruff format, pytest must pass.
- Ask before: downloads over 100 MB, deleting files, adding dependencies not in the task.
- Keep context small: read files selectively, never dump large files or notebooks; summarize command output.

## Definition of done
Tests pass, lint clean, docs updated if a decision or interface changed, final report in the format requested by the task.

## Domain facts (MIMII)
Machine types: fan, pump, slider, valve; 4 machine IDs each (id_00, id_02, id_04, id_06). Recordings: 10 s, 16 kHz, 8 channels, labels normal/abnormal, noise levels 6, 0 and -6 dB (SNR). Zenodo record 3384388.

## Lessons learned
- Anchor .gitignore patterns for top-level folders (/data/, /models/, /runs/); unanchored patterns also hide src/diagnostic/data and src/diagnostic/models.
- The pre-commit module is invoked as: python -m pre_commit (underscore).
- ruff and pre-commit exclude legacy/ on purpose; never remove that exclusion.
- Verify renames with: git diff -M --stat main HEAD (do not restrict the pathspec).
