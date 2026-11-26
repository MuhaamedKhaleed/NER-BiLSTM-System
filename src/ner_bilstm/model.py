"""Optional TensorFlow backend. Import and compute occur only on explicit calls."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .core import Sentence, Vocabulary


def backend() -> Any:
    """Import the optional CPU backend with an actionable installation error."""
    try:
        import tensorflow as tf
    except ImportError as exc:
        raise ValueError(
            "ML backend missing. Install with uv sync --extra ml."
        ) from exc
    return tf


def build_model(vocab: Vocabulary, embedding: int = 100, units: int = 64) -> Any:
    """Build a dynamic-length BiLSTM returning one tag distribution per token."""
    if embedding < 1 or units < 1:
        raise ValueError("Embedding size and LSTM units must be positive.")
    tf = backend()
    inputs = tf.keras.Input(shape=(None,), dtype="int32")
    x = tf.keras.layers.Embedding(len(vocab.words) + 2, embedding, mask_zero=True)(
        inputs
    )
    x = tf.keras.layers.Bidirectional(
        tf.keras.layers.LSTM(units, return_sequences=True)
    )(x)
    outputs = tf.keras.layers.Dense(len(vocab.tags) + 1, activation="softmax")(x)
    model = tf.keras.Model(inputs, outputs)
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy")
    return model


def arrays(sentences: list[Sentence], vocab: Vocabulary) -> tuple[Any, Any, Any]:
    """Post-pad locally to batch length. Weights exclude padding from the loss."""
    if not sentences:
        raise ValueError("Cannot encode an empty batch.")
    import numpy as np

    width = max(len(s.words) for s in sentences)
    x = np.zeros((len(sentences), width), dtype=np.int32)
    y = np.zeros_like(x)
    weights = np.zeros_like(x, dtype=np.float32)
    for i, sentence in enumerate(sentences):
        length = len(sentence.words)
        x[i, :length] = vocab.encode_words(sentence.words)
        y[i, :length] = vocab.encode_tags(sentence.tags)
        weights[i, :length] = 1
    return x, y, weights


def token_scores(
    model: Any, sentences: list[Sentence], vocab: Vocabulary
) -> dict[str, int | float]:
    """Measure token accuracy excluding padding; this is not entity span F1."""
    x, y, weights = arrays(sentences, vocab)
    predictions = model(x, training=False).numpy()
    # Padding class must never be emitted for real tokens.
    indices = predictions[:, :, 1:].argmax(axis=-1) + 1
    correct = int(((indices == y) * weights).sum())
    total = int(weights.sum())
    return {
        "correct_tokens": correct,
        "tokens": total,
        "token_accuracy": correct / total,
    }


def train(
    train_data: list[Sentence],
    validation: list[Sentence],
    output: Path,
    epochs: int = 10,
    batch_size: int = 32,
    seed: int = 42,
    embedding: int = 100,
    units: int = 64,
) -> dict[str, int | float]:
    """Explicit local training, never invoked by import/help/data validation.

    A separate untouched test set is needed for any final quality claim.
    """
    if epochs < 1 or batch_size < 1:
        raise ValueError("Epochs and batch size must be positive.")
    if {s.key for s in train_data} & {s.key for s in validation}:
        raise ValueError("Training and validation sentence IDs must be disjoint.")
    if output.exists():
        raise ValueError(
            "Output already exists. Choose a new directory, no overwrite allowed."
        )
    tf = backend()
    vocab = Vocabulary.fit(train_data)
    x, y, weights = arrays(train_data, vocab)
    vx, vy, vw = arrays(validation, vocab)
    tf.keras.utils.set_random_seed(seed)
    model = build_model(vocab, embedding, units)
    model.fit(
        x,
        y,
        sample_weight=weights,
        validation_data=(vx, vy, vw),
        epochs=epochs,
        batch_size=batch_size,
        verbose=0,
    )
    scores = token_scores(model, validation, vocab)
    output.mkdir(parents=True)
    model.save(output / "model.keras")
    (output / "vocabulary.json").write_text(
        json.dumps({"words": vocab.words, "tags": vocab.tags}), encoding="utf-8"
    )
    (output / "validation.json").write_text(json.dumps(scores), encoding="utf-8")
    return scores


def predict(directory: Path, words: tuple[str, ...]) -> list[dict[str, str]]:
    """Predict a complete pretokenized sentence using a trusted local artifact.

    Never load artifacts from unknown sources, even with Keras safe mode.
    """
    if not words or any(not w.strip() for w in words):
        raise ValueError("Supply at least one nonempty token.")
    tf = backend()
    import numpy as np

    payload = json.loads((directory / "vocabulary.json").read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or set(payload) != {"words", "tags"}:
        raise ValueError("Vocabulary JSON needs exactly words and tags maps.")
    vocab = Vocabulary(payload["words"], payload["tags"])
    model = tf.keras.models.load_model(
        directory / "model.keras", compile=False, safe_mode=True
    )
    if model.output_shape[-1] != len(vocab.tags) + 1:
        raise ValueError("Model output does not match vocabulary tag count.")
    if model.layers[1].input_dim != len(vocab.words) + 2:
        raise ValueError("Model embedding does not match vocabulary word count.")
    output = model(
        np.array([vocab.encode_words(words)], dtype=np.int32), training=False
    ).numpy()[0]
    ids = output[:, 1:].argmax(axis=-1) + 1
    reverse = {value: key for key, value in vocab.tags.items()}
    return [
        {"word": word, "tag": reverse[int(label)]}
        for word, label in zip(words, ids, strict=True)
    ]
