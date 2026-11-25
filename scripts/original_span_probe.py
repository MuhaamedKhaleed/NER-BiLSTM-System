"""Reproducible original-path BiLSTM probe with development-only selection.

This script makes the recorded full-data BiLSTM result in docs/quality-cpu.md
reproducible and asks one bounded follow-up question: was the weak span F1 of
the original BiLSTM an artifact of one untuned default configuration? It trains
a small number of fixed BiLSTM configurations on the full WNUT17 training
sentences, selects epochs and the winning configuration by development exact
-span F1 only, and reads the official annotated test once for the winner.

The official test was already read by the earlier BiLSTM run, the sparse
linear baseline and the frozen contextual alternative. It is exposed; treat
every number here as a development comparison on an exposed test, never as an
untouched benchmark. No automatic download, no import-time work, no quality
or production claim.

Crash safety: the new output directory is created at start and per-epoch
development scores, per-config best weights and partial reports are written as
training progresses. Rerun with --resume into that directory to continue an
interrupted probe without repeating completed configurations; the single test
read still happens once, after all configurations finish.
"""


def main():
    import argparse
    from pathlib import Path

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Continue an interrupted probe in its existing output directory.",
    )
    args = parser.parse_args()
    if args.output.exists() and not args.resume:
        parser.error("Output exists; use a new directory, or --resume it.")
    if args.epochs < 1:
        parser.error("Epochs must be positive.")
    import os

    for key in ["TF_NUM_INTRAOP_THREADS", "TF_NUM_INTEROP_THREADS", "OMP_NUM_THREADS"]:
        os.environ[key] = "1"
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    import hashlib
    import json
    import resource
    import time

    from ner_bilstm.core import Sentence, Vocabulary
    from ner_bilstm.model import arrays, backend

    def read(name, prefix):
        out, words, tags = [], [], []
        for line in (args.data / name).read_text().splitlines() + [""]:
            if line.strip():
                word, tag = line.rsplit("\t", 1)
                words.append(word)
                tags.append(tag)
            elif words:
                out.append(Sentence(prefix + str(len(out)), tuple(words), tuple(tags)))
                words, tags = [], []
        return out

    def spans(tags):
        result, start, label = set(), None, None
        for i, tag in enumerate(list(tags) + ["O"]):
            kind, typ = tag.split("-", 1) if "-" in tag else ("O", "")
            if start is not None and (kind != "I" or typ != label):
                result.add((start, i, label))
                start = None
            if kind == "B" or (kind == "I" and start is None):
                start, label = i, typ
        return result

    def evaluate(model, rows, vocab):
        reverse = {value: key for key, value in vocab.tags.items()}
        tp = fp = fn = correct = total = 0
        for start in range(0, len(rows), 64):
            batch = rows[start : start + 64]
            x, y, weights = arrays(batch, vocab)
            predictions = model(x, training=False).numpy()
            ids = predictions[:, :, 1:].argmax(axis=-1) + 1
            for row, guess, gold, mask in zip(batch, ids, y, weights, strict=True):
                length = len(row.words)
                predicted = [reverse[int(g)] for g in guess[:length]]
                tp += len(spans(row.tags) & spans(predicted))
                fp += len(spans(predicted) - spans(row.tags))
                fn += len(spans(row.tags) - spans(predicted))
                correct += int(((guess == gold) * mask).sum())
                total += int(mask.sum())
        return {
            "entity_micro_f1": 2 * tp / max(1, 2 * tp + fp + fn),
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "token_accuracy": correct / max(1, total),
        }

    start_time = time.time()
    train = read("wnut17train.conll", "train")
    dev = read("emerging.dev.conll", "dev")
    test = read("emerging.test.annotated", "test")
    tf = backend()
    vocab = Vocabulary.fit(train)
    x, y, weights = arrays(train, vocab)
    out = args.output
    out.mkdir(parents=True, exist_ok=args.resume)
    state_path = out / "state.jsonl"
    completed = set()
    trials = []
    if args.resume and state_path.exists():
        for line in state_path.read_text(encoding="utf-8").splitlines():
            record = json.loads(line)
            if record["kind"] == "config":
                completed.add(record["config"]["name"])
                trials.append(record["config"])
    if not args.resume:
        state_path.write_text("", encoding="utf-8")

    def build(embedding, units, dropout, lr):
        inputs = tf.keras.Input(shape=(None,), dtype="int32")
        z = tf.keras.layers.Embedding(len(vocab.words) + 2, embedding, mask_zero=True)(
            inputs
        )
        if dropout:
            z = tf.keras.layers.Dropout(dropout)(z)
        z = tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(units, return_sequences=True)
        )(z)
        if dropout:
            z = tf.keras.layers.Dropout(dropout)(z)
        outputs = tf.keras.layers.Dense(len(vocab.tags) + 1, activation="softmax")(z)
        model = tf.keras.Model(inputs, outputs)
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
            loss="sparse_categorical_crossentropy",
        )
        return model

    configs = [
        {
            "name": "default-e100-u64-lr1e-3",
            "embedding": 100,
            "units": 64,
            "dropout": 0.0,
            "lr": 1e-3,
        },
        {
            "name": "dropout0.3-e100-u64-lr1e-3",
            "embedding": 100,
            "units": 64,
            "dropout": 0.3,
            "lr": 1e-3,
        },
        {
            "name": "dropout0.3-e100-u128-lr3e-4",
            "embedding": 100,
            "units": 128,
            "dropout": 0.3,
            "lr": 3e-4,
        },
    ]

    def weights_file(name):
        return out / (name + ".weights.h5")

    for config in configs:
        if config["name"] in completed:
            continue
        tf.keras.utils.set_random_seed(args.seed)
        model = build(
            config["embedding"], config["units"], config["dropout"], config["lr"]
        )
        history = []
        best_dev, best_epoch = -1.0, 0
        for epoch in range(1, args.epochs + 1):
            model.fit(x, y, sample_weight=weights, epochs=1, batch_size=32, verbose=0)
            score = evaluate(model, dev, vocab)
            row = {"epoch": epoch, "development": score}
            history.append(row)
            print(config["name"], row, flush=True)
            with state_path.open("a", encoding="utf-8") as stream:
                stream.write(
                    json.dumps({"kind": "epoch", "config": config["name"], **row})
                    + "\n"
                )
            if score["entity_micro_f1"] > best_dev:
                best_dev = score["entity_micro_f1"]
                best_epoch = epoch
                model.save_weights(weights_file(config["name"]))
        trial = {
            **config,
            "history": history,
            "best_development_f1": best_dev,
            "best_epoch": best_epoch,
        }
        trials.append(trial)
        with state_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps({"kind": "config", "config": trial}) + "\n")
    winner = max(trials, key=lambda t: t["best_development_f1"])
    model = build(winner["embedding"], winner["units"], winner["dropout"], winner["lr"])
    model.load_weights(weights_file(winner["name"]))
    test_score = evaluate(model, test, vocab)
    report = {
        "source": "https://github.com/leondz/emerging_entities_17",
        "source_sha256": {
            name: hashlib.sha256((args.data / name).read_bytes()).hexdigest()
            for name in [
                "wnut17train.conll",
                "emerging.dev.conll",
                "emerging.test.annotated",
            ]
        },
        "model": (
            "Original-path Keras BiLSTM over train vocabulary, masked padded "
            "loss; probe adds optional dropout and learning-rate variants."
        ),
        "train_sentences": len(train),
        "dev_sentences": len(dev),
        "test_sentences": len(test),
        "selection": (
            "Epoch and configuration selected by development exact-span F1 "
            "only; the single test read below happened after selection."
        ),
        "trials": trials,
        "winner": winner["name"],
        "winner_best_epoch": winner["best_epoch"],
        "test": test_score,
        "elapsed_seconds": time.time() - start_time,
        "peak_rss_kb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "limits": (
            "Official test already exposed to the earlier BiLSTM run, the "
            "sparse linear baseline and the contextual alternative; this is a "
            "development comparison on an exposed test, not an untouched "
            "benchmark. Three fixed configurations, not exhaustive tuning. "
            "BIO exact spans, invalid I starts a new span. Token accuracy is "
            "not entity quality. Not production NER, no fairness claim."
        ),
    }
    (out / "report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({"winner": winner["name"], "test": test_score}))


if __name__ == "__main__":
    main()
