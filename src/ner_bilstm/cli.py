"""Thin local command line. No implicit training or model downloads."""

from __future__ import annotations

import json
import logging
from collections.abc import Callable
from pathlib import Path
from typing import Annotated, Any

import typer
from dotenv import dotenv_values

from .core import Vocabulary, read_sentences, split_sentences

app = typer.Typer(help="Sentence-grouped local BiLSTM NER tools.")
logger = logging.getLogger(__name__)


def run(action: Callable[[], Any]) -> None:
    """Keep machine output on stdout and actionable failures on stderr."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    try:
        typer.echo(json.dumps(action()))
    except (OSError, ValueError, KeyError, TypeError) as exc:
        logger.error("%s", exc)
        raise typer.Exit(2) from exc


def resolve(data: Path | None, config: Path | None) -> Path:
    """Use explicit data first, otherwise opt-in dotenv NER_DATA_PATH."""
    if data:
        return data
    if config:
        if not config.is_file():
            raise ValueError("Config not found. Supply an existing dotenv file.")
        value = dotenv_values(config).get("NER_DATA_PATH")
        if value:
            return Path(value)
    raise ValueError("Supply --data or --config containing NER_DATA_PATH.")


@app.command()
def check_data(
    data: Annotated[Path | None, typer.Option(help="Local sentence CSV.")] = None,
    config: Annotated[
        Path | None, typer.Option(help="Optional explicit dotenv.")
    ] = None,
    encoding: str = "utf-8",
    validation_fraction: float = 0.2,
    seed: int = 42,
) -> None:
    """Validate whole-sentence splits and unseen tags without importing TensorFlow."""

    def action() -> dict[str, int]:
        sentences = read_sentences(resolve(data, config), encoding)
        train, validation = split_sentences(sentences, validation_fraction, seed)
        vocab = Vocabulary.fit(train)
        for sentence in validation:
            vocab.encode_tags(sentence.tags)
        return {
            "sentences": len(sentences),
            "train_sentences": len(train),
            "validation_sentences": len(validation),
            "training_words": len(vocab.words),
            "training_tags": len(vocab.tags),
            "validation_oov_tokens": sum(
                w not in vocab.words for s in validation for w in s.words
            ),
        }

    run(action)


@app.command()
def train(
    data: Path,
    output: Path,
    encoding: str = "utf-8",
    validation_fraction: float = 0.2,
    seed: int = 42,
    epochs: int = 10,
    batch_size: int = 32,
    embedding: int = 100,
    units: int = 64,
) -> None:
    """Start local CPU training deliberately; requires optional ML dependencies."""

    def action() -> dict[str, int | float]:
        from .model import train as fit

        train_data, validation = split_sentences(
            read_sentences(data, encoding), validation_fraction, seed
        )
        return fit(
            train_data, validation, output, epochs, batch_size, seed, embedding, units
        )

    run(action)


@app.command()
def predict(
    artifact: Path,
    words: Annotated[list[str], typer.Argument(help="Pretokenized complete sentence.")],
) -> None:
    """Load a trusted local Keras artifact, never remote code or a pickle."""

    def action() -> list[dict[str, str]]:
        from .model import predict as infer

        return infer(artifact, tuple(words))

    run(action)
