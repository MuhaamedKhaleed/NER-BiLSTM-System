# Changelog

## Unreleased

### Fixed
- Replace isolated-word row classification with complete-sentence token labeling.
- Keep validation sentences out of training, fit vocabulary on training only and reject unseen validation tags instead of removing rows.
- Mask/weight padding and prevent output overwrite and implicit training.

### Added
- Explicit local check-data/train/predict CLI, optional CPU backend and trusted-local artifact round-trip.
- Offline tests including real synthetic TensorFlow wiring, packaging, quality checks, CI, data-only Docker and truthful docs.

### Changed
- Original script retained under legacy/, not the supported API. No widget migration, release or pretrained model.
