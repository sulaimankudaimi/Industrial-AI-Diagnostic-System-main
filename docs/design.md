# Design decisions (v1)

## Problem
Detect abnormal machine sounds early, on a single physical machine, with very few or no examples of faults. Evidence must come from rigorous evaluation before any field deployment.

## Decisions
| # | Decision | Rationale |
|---|----------|-----------|
| D1 | Primary approach: unsupervised anomaly detection (train on normal sound only; anomaly score = reconstruction error). A supervised classifier is only a comparison baseline. | Real plants rarely have labeled faults; a classifier cannot flag sounds it has never seen. |
| D2 | First target machine: fan. Development noise level 6 dB; robustness checked at 0 and -6 dB. | A fan is cheap, safe and easy to acquire for the field pilot. |
| D3 | Protocol: one model per machine ID. Per ID: all abnormal recordings plus an equal number of seeded random normal recordings form the test set; 10% of the remaining normal recordings are validation (threshold, early stopping); the rest are training. Splits are by file. Cross-ID generalization is a secondary experiment. | Matches deployment (one model per physical machine) and avoids leakage. |
| D4 | Metrics: AUC and pAUC (max_fpr=0.1) per ID, mean and std over seeds; plus recall at a fixed false-positive rate. Accuracy is never the headline metric. | Classes are imbalanced; false alarms are costly. |
| D5 | Features (initial, tunable in configs): mono = channel 0, 16 kHz, log-mel with n_fft=1024, hop=512, n_mels=64, context of 5 consecutive frames per input vector. | Standard, reproducible baseline from the literature; revisit after baseline results. |
| D6 | Framework: Keras 3 on TensorFlow-CPU; later export to TFLite for edge devices. | Already installed and verified; edge path is straightforward. |
| D7 | Single preprocessing path for training and inference (no PNG detour). | Eliminates train/serve skew found in legacy. |
| D8 | Reproducibility: seeds, YAML configs, each run writes config + metrics JSON to runs/<timestamp>/. | Comparable, auditable experiments. |
| D9 | Data and models are never in git; they live in data/ and models/ (ignored). | Repo stays small and clean. |
| D10 | Legacy code is frozen under legacy/ and tag v0-legacy; it is not fixed. | It is a documented baseline only. |
| D11 | MIMII zips extract under data/raw/mimii/<snr>_dB_<type>/id_XX/{normal,abnormal}/*.wav. Discovery tolerates extra nesting, always sorts, and never truncates. Channel 0 is selected. | Reproducible, stable file inventory; avoids the legacy os.listdir ordering bug. |
| D12 | Downloads are opt-in (`--download`), resumable via HTTP Range, and MD5-verified against the Zenodo record metadata; the default is a dry-run listing. | Large files, flaky links; never fetch implicitly and never trust a partial file. |
| D13 | Splits are persisted as JSON manifests in data/splits/ named <type>_<snr>dB_<id>_seed<seed>.json, storing paths relative to raw_dir plus counts and a short content hash. | Auditable, portable, and diffable split definitions. |

## Roadmap (phases)
F1 foundation -> F2 data pipeline (MIMII fan, 6 dB) -> F3 features -> F4 baseline autoencoder + evaluation -> F5 inference library, API and UI -> F6 packaging -> F7 hardware and own recordings -> F8 field pilot -> F9 monitoring.

## Open questions
- Channel handling: channel 0 vs mix vs multi-channel (revisit after F4).
- Threshold policy for the field (per-machine calibration on own normal recordings).
- Hardware selection (microphone, interface, edge device) - decided in F7.
