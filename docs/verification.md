# Verification and phase record

Local CPU, Linux, Python 3.11.18. Corrective behavior change: original code treated each word row as a separate sequence, split rows, dropped unknown test labels and created widgets at import. Original retained under legacy/. New CLI does not auto-train.

1. Package: src layout and installed Typer entry point; module/root help checked.
2. Cleanup: sentence parsing/split/vocabulary, optional model adapter and CLI separated. Public functions typed. Old widget API intentionally unsupported.
3. Validation: malformed CSV, first/repeated sentence IDs, empty tokens, unique headers, fraction/shape/vocabulary errors; stderr/exit 2 tested.
4. Config: explicit dotenv NER_DATA_PATH, argument precedence, no implicit secret files.
5. Dependencies: runtime/dev/optional ML separated, uv lock, fresh locked sync passes. TensorFlow CPU 2.18.0 and NumPy 1.26.4 locally exercised.
6. Tests: 38 tests, 97% statement coverage locally before final publication. Includes actual TensorFlow output shape, right-padding invariance, masked weighted loss, tiny synthetic epoch/save/load/infer. No real NER quality evidence.
7. Ruff check/format and strict mypy pass. Initial pre-commit command used system Python, then corrected PATH to the active environment and rerun on tracked files; corrected tracked-file run passed all hooks.
8. Public-only free CI uses standard runner, no caches, artifacts or model publishing. Lint/types/hooks/tests and Docker CLI smoke. Hosted result separate from local checks.
9. Docker: multi-stage nonroot lightweight data-only runtime, no TensorFlow. Local Docker absent, hosted check required.
10. Thin explicit check-data/train/predict commands. Missing-file/config and no-overwrite behavior tested. Complete sentence, not isolated words, is the prediction unit.
11. README describes behavior changes, real synthetic check output, architecture and limits.
12. Feature branch, original isolated in separate commit. No merge, release, tag, blanket license or main change.
13. Suggested description/topics/profile text prepared, not applied.

Commands: uv sync --locked --extra dev --extra ml; python -m ner_bilstm --help; python main.py --help; ner-bilstm --help; python -m pytest --cov=ner_bilstm --cov-report=term-missing; python -m ruff check src tests; python -m ruff format --check src tests; python -m mypy src; python -m pre_commit run --all-files. TensorFlow test environment limits CPU threads, not sample sizes or test assertions.

Synthetic training uses invented tiny sentences and a single epoch only to test execution and artifact round-trip. No model from it is published. Train and validation keys must differ. The synthetic sentences intentionally reuse content, so this is not a generalization experiment. Padding labels have zero loss weight and inference excludes the padding class. Unknown validation labels fail instead of lowering evaluation support silently. Error handling catches expected local/data failures, not every possible TensorFlow hardware failure.

Remaining: licensed corpus, document/source-aware and deduplicated splits, untouched test set, span F1, confidence intervals/error analysis, reproducibility across devices, full-data runtime/memory, trustworthy deployment. Lock does not prove all platforms work. Keras safe mode does not authorize loading unknown artifacts.


## Gap-closure follow-up (24 November 2026)

The CI source now uses uv 0.13.8 and the committed transitive lock, rather than a new pip resolution of direct pins. Dependency installer caching is keyed by uv.lock. The local matching command is `uv sync --locked --extra dev --extra ml`, followed by the existing tests/quality commands. This does not cache or publish private datasets, weights or model outputs. Public-repository-only guards remain. The changed workflow's hosted outcome must be checked after publication; prior green runs do not validate a changed workflow.

A complete copy-only project-focused profile README draft is appended to presentation.md. No profile account edits, main merge, PR, release or license change is implied by that text. Historical notebook/full-data/quality limitations remain open, not replaced by a passing synthetic test.

## Standalone synthetic demo and CPU container follow-up

`python scripts/demo.py NEW_DIRECTORY` ran successfully in the locked local environment, with actual backend training/artifact/reload/inference where a model backend exists. `docs/demo-output.json` records one actual run, labeled synthetic-only with no quality claim. A subprocess regression checks the report and no-overwrite behavior. Tests, lint/format and package mypy rerun locally. OCR demo is decoder-only, monitoring uses actual SciPy. `Dockerfile.ml` is opt-in where applicable; hosted build/run evidence must be checked after publication, not inferred from local Python.
