# Legacy findings (frozen at tag v0-legacy)

1. The trained model has 8 output classes (fan/pump/slider/valve x normal/abnormal); app.py defines 2 classes, so predictions with index >= 2 raise IndexError. Verified: model.output_shape == (None, 8).
2. The reported ~80% accuracy is on 8 classes; part of it is machine-type discrimination, not fault detection. Fault detection quality is unknown.
3. No held-out test set, no confusion matrix, no AUC/recall for abnormal sounds.
4. Data: only id_00 per machine type, first 400 files per folder (os.listdir order), -6 dB only, random 80/20 split via ImageDataGenerator validation_split (possible leakage across correlated frames/files).
5. Preprocessing goes through matplotlib PNG images (2x2 in, default colormap) resized to 128x128; training and serving resize paths differ; fixed temp file "temp_spec.png" causes a race condition for concurrent users.
6. A supervised classifier cannot flag unseen sounds as anomalous (random noise gets a confident class).
7. requirements.txt had no version pins; the exact working versions are in requirements-legacy-frozen.txt.
