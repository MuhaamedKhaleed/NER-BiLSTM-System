# Frozen contextual NER alternative,24November2026

An explicit alternative to BiLSTM, not a repair of its trained weights. Frozen DistilBERT base uncased encoder, train-fitted ridge token head and dev-selected O-bias. No encoder fine-tuning. Base is not somebody else's WNUT-finetuned model. Source: https://huggingface.co/distilbert/distilbert-base-uncased . Apache2.0 license read: https://huggingface.co/distilbert/distilbert-base-uncased/raw/main/LICENSE . Third-party rights remain separate from this repository's engineering MIT scope. No model weights, raw social text or feature caches published. Pretraining on books/Wikipedia may introduce unknown source overlap and bias; no fairness guarantee.

## Reproducible local inputs

Encoder revision: 12040accade4e8a0f71eabdb258fecc2e7e948be. Download config.json,tokenizer.json,tokenizer_config.json,vocab.txt,model.safetensors and LICENSE from that pinned source into an explicit local directory, using the supported HuggingFace client. Keep the license. No credentials required. Artifact SHA256 recorded in contextual-provenance.json. The script never automatically downloads or runs remote code; local_files_only and safetensors required.

Use authoritative WNUT2017 source files from existing fetch recipe:wnut17train.conll,emerging.dev.conll,emerging.test.annotated. Source https://github.com/leondz/emerging_entities_17 ,CCBY4 attribution and social-text provenance caveats remain. An older parallel emerging.test.conll copy has changed gold labels, so is NOT used for the matched comparison. Test source SHA256 and token/span denominator must match before interpreting results.

Separate optional environment, does not change default dev/ml dependency group:

```sh
uv venv .venv-context --python 3.11
uv pip sync --python .venv-context/bin/python --extra-index-url https://download.pytorch.org/whl/cpu requirements-contextual.lock
.venv-context/bin/python scripts/contextual_cpu.py extract --data SOURCE --encoder LOCAL_ENCODER --split train --output FEATURES/train.npz
.venv-context/bin/python scripts/contextual_cpu.py extract --data SOURCE --encoder LOCAL_ENCODER --split dev --output FEATURES/dev.npz
OPENBLAS_NUM_THREADS=1 .venv-context/bin/python scripts/contextual_cpu.py fit --features FEATURES --output HEAD.npz
.venv-context/bin/python scripts/contextual_cpu.py extract --data SOURCE --encoder LOCAL_ENCODER --split test --output FEATURES/test.npz
.venv-context/bin/python scripts/contextual_cpu.py evaluate --features FEATURES/test.npz --head HEAD.npz --output evaluation.json
.venv-context/bin/python scripts/contextual_cpu.py predict --encoder LOCAL_ENCODER --head HEAD.npz --text Alice visited Cairo . --output prediction.json
```

Create FEATURES directory first. Every output must be new; extract/fit outputs must end in.npz. The script uses local source data and local encoder and does no work at import/help. Inference is optional, not the default BiLSTM CLI. Dense feature arrays have768dimensions, first subword per source word, overlapping512token windows/stride64, no missing token allowed. Empty-normalizing words use the tokenizer unknown token. No silent truncation. Fit standardization/intercept only on train, not dev/test. Zero-pickle NPZ loading.

## Actual measured results

3394train sentences/62730tokens,1009dev/15733tokens,1287test/23394tokens. Twelve dev candidates:ridgealpha1,10,100 and O additive bias-.3,-.1,0,.1. Selected first tie alpha1/bias-.3. Dev exact-span F1 .498909,343TP/196FP/493FN. Test .361996,301TP/283FP/778FN. Same annotated test as previous sparseSGD .140481,152TP/933FP/927FN. Both use exact spans with invalidI-asnew convention. Test was previously exposed in prior waves; not an untouched new benchmark or independent external proof, and was not used for this head selection. Much remains missed. Do not market token accuracy as entity quality or call this solved/production ready.

Portable extraction matched saved dev features exactly. A clean synced optional environment independently extracted all train features and matched exactly. Portable fit/evaluate repeated the metrics. Toy inference tagged Alice asperson and Cairo aslocation, just a smoke check, not quality evidence. Two logistic-head attempts exceeded bounded time; ridge solved quickly. CRF marginal calibration improved dev .3002 but was weak on the older parallel test copy; not used as comparison evidence.

All standard repository tests remain offline and use no downloads. New safety tests cover import/help and preserving an existing output. Encoder/license artifacts remain local.
