"""Explicit contextual sparse linear NER baseline, not the BiLSTM."""


def main():
    import argparse
    from pathlib import Path

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output exists; use a new directory.")
    import json
    import resource
    import time
    from pathlib import Path

    from sklearn.feature_extraction import DictVectorizer
    from sklearn.linear_model import SGDClassifier

    # Reuse only parser and strict span evaluator definitions, never neural training.
    def read(name, prefix):
        out = []
        words = []
        tags = []
        for line in (source / name).read_text().splitlines() + [""]:
            if line.strip():
                w, t = line.rsplit("\t", 1)
                words.append(w)
                tags.append(t)
            elif words:
                out.append(Sentence(prefix + str(len(out)), tuple(words), tuple(tags)))
                words = []
                tags = []
        return out

    def spans(tags):
        result = set()
        start = None
        label = None
        for i, t in enumerate(list(tags) + ["O"]):
            kind, typ = t.split("-", 1) if "-" in t else ("O", "")
            if start is not None and (kind != "I" or typ != label):
                result.add((start, i, label))
                start = None
            if kind == "B" or (kind == "I" and start is None):
                start = i
                label = typ
        return result

    source = args.data
    root = args.output
    root.mkdir(exist_ok=False)

    def features(s, i):
        w = s.words[i]
        f = {
            "word": w.lower(),
            "suffix2": w[-2:],
            "suffix3": w[-3:],
            "prefix2": w[:2],
            "istitle": w.istitle(),
            "isupper": w.isupper(),
            "isdigit": w.isdigit(),
            "hasdigit": any(c.isdigit() for c in w),
        }
        for d in [-2, -1, 1, 2]:
            j = i + d
            if 0 <= j < len(s.words):
                t = s.words[j]
                f[str(d) + "word"] = t.lower()
                f[str(d) + "title"] = t.istitle()
            else:
                f[str(d) + "boundary"] = True
        return f

    def encode(rows):
        return [features(s, i) for s in rows for i in range(len(s.words))]

    def evaluate(rows, pred):
        tp = fp = fn = offset = 0
        for s in rows:
            tags = pred[offset : offset + len(s.tags)]
            offset += len(s.tags)
            g = spans(s.tags)
            p = spans(tags)
            tp += len(g & p)
            fp += len(p - g)
            fn += len(g - p)
        return {
            "entity_micro_f1": 2 * tp / max(1, 2 * tp + fp + fn),
            "tp": tp,
            "fp": fp,
            "fn": fn,
        }

    from ner_bilstm.core import Sentence

    train = read("wnut17train.conll", "train")
    dev = read("emerging.dev.conll", "dev")
    vector = DictVectorizer()
    x = vector.fit_transform(encode(train))
    vx = vector.transform(encode(dev))
    y = [t for s in train for t in s.tags]
    history = []
    best = -1
    start = time.time()
    for alpha in [1e-5, 1e-4, 1e-3]:
        model = SGDClassifier(
            loss="log_loss",
            alpha=alpha,
            max_iter=30,
            tol=1e-4,
            random_state=42,
            class_weight="balanced",
        )
        model.fit(x, y)
        score = evaluate(dev, model.predict(vx))
        history.append({"alpha": alpha, **score})
        print(history[-1], flush=True)
        if score["entity_micro_f1"] > best:
            best = score["entity_micro_f1"]
            chosen = model
            chosen_alpha = alpha
    test = read("emerging.test.annotated", "test")
    score = evaluate(test, chosen.predict(vector.transform(encode(test))))
    r = {
        "source": "https://github.com/leondz/emerging_entities_17",
        "model": (
            "Contextual word/shape sparse balanced SGDlinear classifier, not BiLSTM"
        ),
        "development_trials": history,
        "selected_alpha": chosen_alpha,
        "test": score,
        "elapsed_seconds": time.time() - start,
        "peak_rss_kb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "limits": (
            "Officialtest also evaluated earlier neural baseline; "
            "dev-only hyperparameters. Word tags may violate BIO; "
            "spans apply invalidI-asnew rule. No fairness guarantee."
        ),
    }
    (root / "report.json").write_text(json.dumps(r, indent=2))
    print(json.dumps(r))


if __name__ == "__main__":
    main()
