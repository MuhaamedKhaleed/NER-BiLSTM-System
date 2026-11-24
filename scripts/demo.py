"""Explicit synthetic wiring demo. No real-data accuracy or quality claim."""

import json
import os
import sys
from pathlib import Path


def main() -> None:
    for key, value in {
        "CUDA_VISIBLE_DEVICES": "-1",
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "TF_CPP_MIN_LOG_LEVEL": "3",
        "TF_NUM_INTRAOP_THREADS": "1",
        "TF_NUM_INTEROP_THREADS": "1",
        "OMP_NUM_THREADS": "1",
    }.items():
        os.environ.setdefault(key, value)

    out = Path(sys.argv[1] if len(sys.argv) == 2 else "synthetic-run")
    if len(sys.argv) > 2:
        raise SystemExit("Usage: python scripts/demo.py [NEW_OUTPUT_DIRECTORY]")
    out.mkdir(parents=True, exist_ok=False)
    from ner_bilstm.core import Sentence
    from ner_bilstm.model import backend, predict, train

    backend().keras.utils.set_random_seed(42)
    training = [
        Sentence("t1", ("Mona", "visits", "Cairo"), ("B-PER", "O", "B-LOC")),
        Sentence("t2", ("Cairo",), ("B-LOC",)),
    ]
    validation = [Sentence("v1", ("Mona", "Cairo"), ("B-PER", "B-LOC"))]
    scores = train(
        training,
        validation,
        out / "model",
        epochs=1,
        batch_size=2,
        embedding=4,
        units=3,
    )
    prediction = predict(out / "model", ("Mona", "unseen", "Cairo"))
    assert len(prediction) == 3 and scores["tokens"] == 2
    result = {"training": scores, "prediction": prediction}
    scope = "invented_sentences_tiny_random_network_not_NER_quality"
    report = {
        "synthetic_only": True,
        "scope": scope,
        "quality_claim": False,
        "result": result,
    }
    (out / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {"synthetic_only": True, "report": str(out / "report.json")},
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
