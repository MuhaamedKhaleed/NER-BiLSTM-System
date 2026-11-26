# Reproducible original-path BiLSTM probe

25 November 2026. Development comparison on an exposed test, not a benchmark.

The full-data BiLSTM result recorded in docs/quality-cpu.md (development span
F1 .0731, test .0480) had no reproducing script in the repository and came
from a single untuned default configuration. This follow-up (a) makes that
neural run reproducible end to end and (b) asks whether the weak result was
an artifact of that one default, using the same data, span convention and
evaluation file as the earlier alternatives.

## Protocol

Full WNUT17 sources: 3394 training, 1009 development, 1287 test sentences
(source SHA-256 pinned in the JSON). Exact BIO spans, invalid I- starts a new
span, same adjudicated `emerging.test.annotated` file as the linear and
contextual alternatives. Three fixed configurations of the original-path
Keras BiLSTM (train-only vocabulary, masked padded loss, padding class
excluded from predictions):

| Configuration | Best development span F1 | Epoch |
|---|---:|---:|
| default (embedding 100, units 64, Adam 1e-3, no dropout) | .0860 | 12 |
| + dropout 0.3 | .0771 | 12 |
| dropout 0.3, units 128, Adam 3e-4 | .0354 | 11 |

Epoch and winner selected on development only. The winner (the default
configuration) was then read once on the official test: entity span F1
.0549 (40 TP / 337 FP / 1039 FN), token accuracy .9168. This is close to the earlier recorded .0480, but no seed-variance estimate
or significance test was performed.

## Conclusion

In these three tested configurations, dropout and a larger lower-learning-rate
variant did not fix weak span quality. The default remains best on this dev
set. This does not rule out other training or architecture choices. High token
accuracy (~.92) still conceals weak entity recall (1039 missed entities).
Within these three fixed configurations the original neural path remains far
below the frozen contextual alternative (.362 test F1, same file and span
convention) - a same-source, same-protocol comparison, though the test is
exposed and neither result is production quality.

Exact per-epoch history, hashes and limits: `original-span-probe.json`.

## Reproduce

Python 3.11, TensorFlow CPU 2.18.0, NumPy 1.26.4, locked environment. Local
WNUT17 sources only (see docs/real-data.md for the pinned download helper and
CC BY 4.0 attribution), no download at import or test time, new output
directory required. Roughly 35 minutes on one CPU core. Interrupted runs may use `--resume` to skip completed configurations. An
incomplete configuration restarts from its beginning, not its last epoch.
The script does not enforce a one-time final-test read across repeated invocations.

```sh
uv sync --locked --extra dev --extra ml
uv run --no-sync python scripts/original_span_probe.py --data LOCAL_WNUT_SOURCE --output NEW_DIRECTORY
python -m pytest tests/test_original_span_probe.py
```

The official test has now been read by the earlier BiLSTM run, the sparse
linear baseline, the frozen contextual alternative and this probe. It is
exposed; never cite these numbers as an untouched benchmark. Three fixed
configurations are not exhaustive tuning. No model weights or raw text are
committed. Not production NER, no fairness claim.
