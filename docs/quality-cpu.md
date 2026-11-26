# Measured NER quality follow-up

The full3394training/1009development BiLSTM through11epochs gave devspanF1.0731 and testF1.0480. High tokenaccuracy.922 concealed weak entityrecall. The first attempt using emerging.test.conll was rejected because one annotation is a comma-separated ambiguous tag; the repository's emerging.test.annotated adjudicated file supplies B-location. No row was silently removed.

The explicit alternative is contextual word/shape features with DictVectorizer and balanced SGDlinear classifier, not a BiLSTM replacement. Dev-selected alpha.0001 from three trials: devF1.2193, officialtestF1.1405,6.9seconds/~285MiB. Convergence warnings at30iterations are retained, not claimed converged. The officialtest had been observed for the neural baseline but did not select linear hyperparameters; all choices used development only. Not a stateoftheart or fairness claim. BIO exact spans, invalid I starts a newspan.

```sh
uv sync --locked --extra dev --extra ml
uv run --with scikit-learn==1.6.1 python scripts/quality_cpu.py --data LOCAL_WNUT_SOURCE --output NEW_OUTPUT
```

The helper fetches train/dev only. For final evaluation get emerging.test.annotated from the same pinned upstream revision recorded in real-data.md, retain README/CCBY4 attribution. No rawtext/model committed. Future work must not call the observed test unseen. Neural result is a recorded experiment; this script reproduces the separately identified linear alternative.
