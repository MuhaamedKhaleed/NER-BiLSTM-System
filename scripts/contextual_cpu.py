"""Optional frozen-context NER alternative, not the BiLSTM.

Explicit local artifacts only. Run extract for train/dev, fit, then extract test
and evaluate once. Test is already exposed, never tune on its results.
"""


def main():
    import argparse
    from pathlib import Path

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage", choices=["extract", "fit", "evaluate", "predict"])
    p.add_argument("--data", type=Path)
    p.add_argument("--encoder", type=Path)
    p.add_argument("--features", type=Path)
    p.add_argument("--head", type=Path)
    p.add_argument("--split", choices=["train", "dev", "test"], default="dev")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--text", nargs="+")
    args = p.parse_args()
    if args.output.exists():
        p.error("Output exists; choose a fresh file.")
    if args.stage in {"extract", "fit"} and args.output.suffix != ".npz":
        p.error("Extract/fit output must end in .npz.")
    import os

    os.environ["USE_TF"] = "0"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["OMP_NUM_THREADS"] = "1"
    import hashlib
    import json

    import numpy as np

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

    def score(gold, pred, lengths):
        tp = fp = fn = offset = 0
        for length in lengths:
            g = spans(gold[offset : offset + length])
            guess = spans(pred[offset : offset + length])
            offset += length
            tp += len(g & guess)
            fp += len(guess - g)
            fn += len(g - guess)
        if offset != len(gold) or len(pred) != len(gold):
            raise ValueError("Token alignment mismatch.")
        return dict(span_f1=2 * tp / max(1, 2 * tp + fp + fn), tp=tp, fp=fp, fn=fn)

    def contextual(rows):
        if args.encoder is None or not (args.encoder / "model.safetensors").is_file():
            p.error("Local safetensors encoder required, no automatic download.")
        import torch
        from transformers import AutoModel, AutoTokenizer

        torch.set_num_threads(1)
        tok = AutoTokenizer.from_pretrained(
            args.encoder, local_files_only=True, trust_remote_code=False
        )
        model = AutoModel.from_pretrained(
            args.encoder,
            local_files_only=True,
            trust_remote_code=False,
            use_safetensors=True,
        ).eval()
        xs = []
        for i in range(0, len(rows), 4):
            batch = rows[i : i + 4]
            words = [
                [word if tok.tokenize(word) else tok.unk_token for word in row]
                for row in batch
            ]
            enc = tok(
                words,
                is_split_into_words=True,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512,
                return_overflowing_tokens=True,
                stride=64,
            )
            mapping = enc.pop("overflow_to_sample_mapping").tolist()
            with torch.inference_mode():
                hidden = model(**enc).last_hidden_state.numpy()
            feat = [
                np.zeros((len(row), hidden.shape[-1]), dtype=np.float32)
                for row in batch
            ]
            seen = [set() for _ in batch]
            for chunk, j in enumerate(mapping):
                for k, word in enumerate(enc.word_ids(chunk)):
                    if word is not None and word not in seen[j]:
                        feat[j][word] = hidden[chunk, k]
                        seen[j].add(word)
            for j, row in enumerate(batch):
                if len(seen[j]) != len(row):
                    raise ValueError("Missing token representation.")
                xs.extend(feat[j])
        return np.asarray(xs)

    if args.stage == "extract":
        if args.data is None:
            p.error("Local source data directory required.")
        name = {
            "train": "wnut17train.conll",
            "dev": "emerging.dev.conll",
            "test": "emerging.test.annotated",
        }[args.split]
        raw = (args.data / name).read_bytes()
        rows, tags, words, labels = [], [], [], []
        for line in raw.decode().splitlines() + [""]:
            if line.strip():
                word, label = line.rsplit("\t", 1)
                words.append(word)
                labels.append(label)
            elif words:
                rows.append(words)
                tags.extend(labels)
                words, labels = [], []
        x = contextual(rows)
        np.savez_compressed(
            args.output,
            x=x,
            y=np.asarray(tags),
            lengths=[len(row) for row in rows],
            source_sha256=hashlib.sha256(raw).hexdigest(),
        )
        print(json.dumps(dict(sentences=len(rows), tokens=len(tags), split=args.split)))
    elif args.stage == "fit":
        if args.features is None:
            p.error("Features directory required with train.npz and dev.npz.")
        train = np.load(args.features / "train.npz", allow_pickle=False)
        dev = np.load(args.features / "dev.npz", allow_pickle=False)
        x, y = train["x"], train["y"]
        mean, scale = x.mean(0), x.std(0) + 1e-5
        x = (x - mean) / scale
        vx = (dev["x"] - mean) / scale
        classes = np.unique(y)
        targets = np.zeros((len(y), len(classes)), dtype=np.float32)
        for j, tag in enumerate(classes):
            targets[:, j] = y == tag
        intercept = targets.mean(0)
        a, b = x.T @ x, x.T @ (targets - intercept)
        trials, best = [], None
        oi = list(classes).index("O")
        for alpha in [1.0, 10.0, 100.0]:
            weights = np.linalg.solve(a + alpha * np.eye(a.shape[0]), b)
            prob = vx @ weights + intercept
            for bias in [-0.3, -0.1, 0.0, 0.1]:
                adjusted = prob.copy()
                adjusted[:, oi] += bias
                result = dict(
                    alpha=alpha,
                    o_bias=bias,
                    **score(dev["y"], classes[adjusted.argmax(1)], dev["lengths"]),
                )
                trials.append(result)
                if best is None or result["span_f1"] > best["span_f1"]:
                    best = result
                    selected = weights.copy()
        np.savez_compressed(
            args.output,
            weights=selected,
            mean=mean,
            scale=scale,
            intercept=intercept,
            classes=classes,
            o_bias=best["o_bias"],
            alpha=best["alpha"],
        )
        print(json.dumps(dict(trials=trials, selected=best, test_read=False), indent=2))
    else:
        if args.head is None:
            p.error("Local head .npz required.")
        head = np.load(args.head, allow_pickle=False)
        if args.stage == "predict":
            if not args.text:
                p.error("Supply pretokenized words with --text.")
            x = contextual([args.text])
        else:
            if args.features is None:
                p.error("Selected feature .npz required.")
            data = np.load(args.features, allow_pickle=False)
            x = data["x"]
        prob = ((x - head["mean"]) / head["scale"]) @ head["weights"] + head[
            "intercept"
        ]
        prob[:, list(head["classes"]).index("O")] += float(head["o_bias"])
        pred = head["classes"][prob.argmax(1)]
        if args.stage == "predict":
            result = [
                dict(word=word, tag=tag)
                for word, tag in zip(args.text, pred, strict=True)
            ]
        else:
            result = dict(
                **score(data["y"], pred, data["lengths"]),
                limits="Previously exposed test; no new untouched claim.",
            )
        args.output.write_text(json.dumps(result, indent=2))
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
