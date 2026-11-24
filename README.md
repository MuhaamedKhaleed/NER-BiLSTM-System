# Sentence-level BiLSTM NER

A local token tagger that groups labeled CSV rows into sentences, splits whole sentences and learns a training-only vocabulary. A bidirectional LSTM sees each complete sentence and emits one label per token. This corrects the historical isolated-word classifier, it does not preserve its predictions.

## Verified scope

38 offline tests pass locally, including actual TensorFlow CPU forward/loss, one tiny synthetic epoch, Keras save/load and inference checks. These verify wiring, not NER quality. No real dataset, pretrained model or full training run was used. Token accuracy is not entity-span F1. No benchmark score is claimed.

## Install and checks

Python 3.11 or 3.12. Verified locally on Linux/Python 3.11 only. Installation needs internet, processing supplied files does not.

```sh
uv sync --locked --extra dev --extra ml
. .venv/bin/activate
ner-bilstm --help
python -m ner_bilstm --help
python main.py --help
python -m pytest --cov=ner_bilstm --cov-report=term-missing
python -m ruff check src tests
python -m ruff format --check src tests
python -m mypy src
python -m pre_commit run --all-files
```

The lightweight `uv sync --locked --extra dev` installation can check data and show help without TensorFlow. ML dependencies are optional and imported only for explicit model operations. Tests skip the backend file if TensorFlow is absent; the full check above includes it.

## CSV and CLI

Supply a local CSV with unique `Sentence #`, `Word`, `Tag` headers. Extra columns such as POS are allowed. An ID starts a sentence; blank subsequent IDs continue it. IDs must be contiguous, first ID cannot be blank, and words/tags cannot be empty. UTF-8 is default; choose `--encoding latin-1` for older files. Words preserve case and punctuation. Unknown words keep their position using an OOV ID; unseen validation tags fail instead of being removed.

```csv
Sentence #,Word,Tag
s1,Mona,B-PER
,visits,O
,Cairo,B-LOC
s2,Cairo,B-LOC
```

The CLI expects shell-separated already-tokenized sentence words. It does not claim language-aware tokenization.

The tested synthetic five-token/four-sentence check reports:

```json
{"sentences": 4, "train_sentences": 3, "validation_sentences": 1, "training_words": 4, "training_tags": 1, "validation_oov_tokens": 1}
```

Data-only example (replace the path with your file):

```sh
ner-bilstm check-data --data /tmp/ner-demo.csv
```

Configuration is opt-in: `ner-bilstm check-data --config .env`. Only `NER_DATA_PATH` is read; explicit `--data` wins. See `.env.example`. No API keys, accounts or automatic downloads.

Explicit training interface: `ner-bilstm train DATA OUTPUT --epochs 10 --batch-size 32`. It writes a new output directory, never overwrites one, and evaluates the held-out validation split. No large training run has been performed. A separate untouched test set is required for final quality evidence. Prediction interface: `ner-bilstm predict ARTIFACT Mona visits Cairo`. Only load your own trusted local Keras artifacts; safe mode is not a security guarantee for untrusted model files. All CLI results are JSON; errors go to stderr and exit 2. Help, data check, synthetic training and prediction paths have been tested, not real-data results.

## Architecture

```text
CSV -> complete sentences -> sentence split -> train-only word/tag maps
  -> post-padded integer arrays + padding weights
  -> masked embedding -> BiLSTM(return_sequences=True) -> per-token softmax
  -> unpadded token accuracy / aligned local predictions
```

Runtime: Typer, Click, python-dotenv. Optional ML: TensorFlow CPU, NumPy. Dev: pytest/cov, Ruff, mypy, pre-commit. No remote API. `legacy/` preserves the original code for comparison, excluded from package/CI. It has known misleading row-level evaluation and import-time widget effects; do not run it as the supported interface. `main.py` now opens the explicit CLI and no longer starts training automatically.

## Container and CI

Multi-stage slim/non-root Docker runs the lightweight data-check CLI. It does not include TensorFlow, data, weights or dev tools. Optional ML runs in the local environment, not this image. Hosted CI checks the actual TensorFlow synthetic path and the container separately. No image publishing, caches, stored artifacts or paid runners are configured. Final hosted result must be checked before claiming a green build.

```sh
docker build --tag ner-bilstm:local .
docker run --rm ner-bilstm:local --help
```

Local Docker is unavailable; build/run are verified only by hosted CI when that job passes, not locally.

## Remaining work

- Train on licensed representative sentence-level data and evaluate an untouched test split with entity-span precision/recall/F1, per-tag support and OOV/error slices.
- Random sentence split does not guarantee document/person/source independence or remove duplicate sentence contents. Audit these before any quality claim.
- No BIO-transition enforcement or CRF, contextual pretrained embeddings, calibrated uncertainty, streaming/large-corpus batching, early stopping or training resume.
- Validation is reused for training feedback, never described as an untouched test set. No fairness, latency, memory or production-readiness claims.
- Optional widgets were not carried into the supported package. No deployment, dataset/model redistribution, release or license change.

See `docs/verification.md` for checks and scope, `docs/presentation.md` for unapplied GitHub copy. Demo screenshot/GIF: TODO after real-data evaluation. No fake badge or result.


## License scope

New project-owned package/engineering files use the owner's MIT default. See LICENSE and LICENSE-SCOPE.md. This is not permission to relicense historical notebooks, third-party code, datasets, models or weights. Their provenance and terms still need review. Earlier verification entries saying no license was added describe the initial pilot, not this follow-up.

## Checked synthetic wiring demo

See [commands, saved output and scope](docs/synthetic-demo.md). This is invented-input execution evidence, not model accuracy or a real-data gallery.

## Real-data CPU example

See [actual results, source terms and explicit download/preparation commands](docs/real-data.md). These are small development runs, not accuracy or production guarantees. No model/data is committed.

## Measured quality follow-up

See [span metrics and explicit linear alternative](docs/quality-cpu.md). Token accuracy alone was misleading; no high-quality NER claim.
