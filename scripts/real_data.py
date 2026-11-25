"""Explicit real-data CPU development recipe. Never runs on import."""

import argparse


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, help="Prepared local source directory")
    parser.add_argument(
        "--base-model", help="Trusted compatible local DistilBERT directory"
    )
    parser.add_argument(
        "--manifest", help="Reviewed selected Open Images source manifest"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=2,
        help="Training epochs (default: the small recorded run)",
    )
    parser.add_argument(
        "--max-train",
        type=int,
        default=512,
        help="Training examples to use (default: 512)",
    )
    args = parser.parse_args()
    import hashlib
    import json
    import os
    import pathlib
    import time

    for k, v in {
        "TF_NUM_INTRAOP_THREADS": "1",
        "TF_NUM_INTEROP_THREADS": "1",
        "OMP_NUM_THREADS": "1",
        "CUDA_VISIBLE_DEVICES": "-1",
        "TF_CPP_MIN_LOG_LEVEL": "3",
    }.items():
        os.environ[k] = v

    from ner_bilstm.core import Sentence, Vocabulary
    from ner_bilstm.model import train

    root = pathlib.Path(args.data)

    def read(file, prefix):
        out = []
        words = []
        tags = []
        for line in (root / file).read_text().splitlines() + [""]:
            if not line.strip():
                if words:
                    out.append(
                        Sentence(f"{prefix}{len(out)}", tuple(words), tuple(tags))
                    )
                    words = []
                    tags = []
            else:
                w, t = line.rsplit("\t", 1)
                words.append(w)
                tags.append(t)
        return out

    # Whole official partitions; small CPU development slice, no row-level random split.
    training = read("wnut17train.conll", "train")[: args.max_train]
    vocab = Vocabulary.fit(training)
    dev_all = read("emerging.dev.conll", "dev")[:128]
    # A small --max-train slice may miss a rare tag; skip dev sentences using it
    # (counted in the report) instead of failing.
    validation = [s for s in dev_all if set(s.tags) <= set(vocab.tags)]
    dropped = len(dev_all) - len(validation)
    if dropped:
        print(
            f"Skipped {dropped} dev sentences with tags absent from the training slice"
        )
    start = time.time()
    scores = train(
        training,
        validation,
        root / "model-real",
        epochs=args.epochs,
        batch_size=16,
        embedding=16,
        units=16,
    )
    report = {
        "real_data": True,
        "dataset": "WNUT 2017 Emerging Entities",
        "source": "https://github.com/leondz/emerging_entities_17",
        "license": "CC BY 4.0",
        "train_sentences": len(training),
        "development_sentences": len(validation),
        "development_skipped_unseen_tags": dropped,
        "selection": (
            f"first{args.max_train} official train / first128 official dev, complete "
            "sentences; final test untouched"
        ),
        "training_sha256": hashlib.sha256(
            (root / "wnut17train.conll").read_bytes()
        ).hexdigest(),
        "training": scores,
        "elapsed_seconds": round(time.time() - start, 2),
        "limits": (
            f"small {args.epochs}-epoch CPU development run; token accuracy, not "
            "entity-span F1; social-text provenance/privacy limits; not"
            " production NER"
        ),
    }
    (root / "real-report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
