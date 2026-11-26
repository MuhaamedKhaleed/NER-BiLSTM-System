# Sentence-Level BiLSTM NER

A local token tagger that groups labeled CSV rows into sentences, splits whole sentences, and learns a training-only vocabulary. A bidirectional LSTM sees each full sentence and emits one label per token. This corrects the historical isolated-word classifier; it does not preserve its predictions.

## Results (honest)

- Fresh-clone contextual route (frozen DistilBERT head, WNUT17, CC BY 4.0): **test span F1 0.3620** (301 TP / 283 FP / 778 FN), versus 0.140 for a sparse-SGD baseline on the same annotated test. The test split was already exposed in earlier runs, so this is not an untouched benchmark, and WNUT17 scores are not comparable to CoNLL-style NER.
- The from-scratch tiny BiLSTM mostly predicts the majority tag `O` on small CPU runs; token accuracy is not entity-span F1 and was misleading historically.
- 38 offline tests (actual TensorFlow forward/loss, tiny synthetic epoch, Keras save/load, inference). Wiring evidence, not NER quality.

## Quickstart (fresh clone, about 1 minute)

Needs Python 3.11 or 3.12 (locked wheels have no 3.13+ build), internet for the first install/downloads, ~2 GB free RAM.

```sh
uv sync --python 3.11 --locked --extra dev --extra ml
uv run --no-sync python scripts/fetch_real_data.py --output ./real-source          # WNUT17, CC BY 4.0
uv run --no-sync python scripts/real_data.py --data ./real-source --epochs 2 --max-train 512
uv run --no-sync ner-bilstm predict ./real-source/model-real Mona visits Cairo
```

The stronger pretrained-encoder route: `scripts/fetch_encoder.py` (pinned, hash-checked `distilbert-base-uncased`, Apache-2.0) plus `scripts/contextual_cpu.py`; steps and measured F1 are in `docs/contextual-alternative.md`. Input CSV needs `Sentence #,Word,Tag` headers; `ner-bilstm check-data --data YOUR.csv` validates it.

## Limits

- Random sentence split is not document/source independence; small `--max-train` slices skip dev sentences with unseen tags, so subset scores are not comparable to the full dev set.
- No CRF/BIO-transition enforcement, calibrated uncertainty, early stopping or resume.
- Validation is reused for training feedback; a separate untouched test set is still needed for final quality evidence.
- Verified on Linux/Python 3.11; the container runs the data-check CLI only (no TensorFlow). Free hosted CI covers tests, lint, CLI and Docker build.
- `legacy/` keeps the original code for comparison only; it has misleading row-level evaluation. No deployment, release or data/model redistribution.

## License

MIT for the new project-owned code (see `LICENSE` and `LICENSE-SCOPE.md`). Historical code, the encoder and datasets keep their own terms. Details and evidence: `docs/verification.md`, `docs/contextual-alternative.md`, `docs/quality-cpu.md`, `CHANGELOG.md`.
