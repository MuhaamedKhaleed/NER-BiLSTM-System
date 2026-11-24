"""Validated sentence data and train-only vocabularies, without ML imports."""

from __future__ import annotations

import csv
import math
import random
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Sentence:
    """One complete sentence with aligned token labels."""

    key: str
    words: tuple[str, ...]
    tags: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.key or not self.words or len(self.words) != len(self.tags):
            raise ValueError("Sentence needs a key and equal nonempty words/tags.")
        if any(not x.strip() for x in (*self.words, *self.tags)):
            raise ValueError("Words and tags must be nonempty strings.")


def read_sentences(path: Path, encoding: str = "utf-8") -> list[Sentence]:
    """Read Kaggle-style Sentence #/Word/Tag CSV, forward-filling sentence IDs.

    Blank sentence IDs continue the preceding sentence. Reappearing IDs after
    another sentence are rejected to prevent accidental split overlap.
    """
    groups: list[Sentence] = []
    key = ""
    words: list[str] = []
    tags: list[str] = []
    seen: set[str] = set()
    with path.open(encoding=encoding, newline="") as stream:
        reader = csv.DictReader(stream)
        fields = reader.fieldnames or []
        if len(set(fields)) != len(fields) or not {"Sentence #", "Word", "Tag"} <= set(
            fields
        ):
            raise ValueError(
                "CSV needs unique headers including Sentence #, Word, Tag."
            )
        for number, row in enumerate(reader, 2):
            if None in row or any(v is None for v in row.values()):
                raise ValueError(f"CSV row {number} has wrong field count.")
            new_key = row["Sentence #"].strip()
            if new_key and new_key != key:
                if new_key in seen:
                    raise ValueError(
                        f"Sentence ID repeats non-contiguously at row {number}."
                    )
                if key:
                    groups.append(Sentence(key, tuple(words), tuple(tags)))
                key, words, tags = new_key, [], []
                seen.add(key)
            if not key:
                raise ValueError("First token needs a sentence ID, not a blank value.")
            word, tag = row["Word"].strip(), row["Tag"].strip()
            if not word or not tag:
                raise ValueError(f"CSV row {number} needs a nonempty Word and Tag.")
            words.append(word)
            tags.append(tag)
    if key:
        groups.append(Sentence(key, tuple(words), tuple(tags)))
    if not groups:
        raise ValueError("CSV contains no sentences.")
    return groups


def split_sentences(
    sentences: list[Sentence], fraction: float = 0.2, seed: int = 42
) -> tuple[list[Sentence], list[Sentence]]:
    """Split whole sentences deterministically, never individual token rows."""
    if len(sentences) < 2 or not math.isfinite(fraction) or not 0 < fraction < 1:
        raise ValueError("Need at least 2 sentences and validation fraction in (0, 1).")
    if len({s.key for s in sentences}) != len(sentences):
        raise ValueError("Sentence keys must be unique before splitting.")
    shuffled = list(sentences)
    random.Random(seed).shuffle(shuffled)
    count = max(1, min(len(shuffled) - 1, math.ceil(len(shuffled) * fraction)))
    return shuffled[count:], shuffled[:count]


@dataclass(frozen=True)
class Vocabulary:
    """Train-only word/tag maps. Word 0 pads, word 1 is OOV; tag 0 pads."""

    words: dict[str, int]
    tags: dict[str, int]

    def __post_init__(self) -> None:
        if not isinstance(self.words, dict) or not isinstance(self.tags, dict):
            raise ValueError("Vocabulary maps must be JSON objects.")
        for mapping in (self.words, self.tags):
            if any(
                not isinstance(k, str) or not k.strip() or type(v) is not int
                for k, v in mapping.items()
            ):
                raise ValueError(
                    "Vocabulary needs nonempty string keys and integer IDs."
                )
        if not self.words or not self.tags:
            raise ValueError("Vocabulary must have training words and tags.")
        if sorted(self.words.values()) != list(range(2, len(self.words) + 2)):
            raise ValueError("Word IDs must be contiguous from 2.")
        if sorted(self.tags.values()) != list(range(1, len(self.tags) + 1)):
            raise ValueError("Tag IDs must be contiguous from 1.")

    @classmethod
    def fit(cls, train: list[Sentence]) -> Vocabulary:
        """Fit only supplied training sentences, preserving case and punctuation."""
        words = sorted({w for s in train for w in s.words})
        tags = sorted({t for s in train for t in s.tags})
        return cls(
            {w: i + 2 for i, w in enumerate(words)},
            {t: i + 1 for i, t in enumerate(tags)},
        )

    def encode_words(self, words: tuple[str, ...]) -> list[int]:
        """Encode every token; unknowns retain position using ID 1."""
        return [self.words.get(w, 1) for w in words]

    def encode_tags(self, tags: tuple[str, ...]) -> list[int]:
        """Reject unseen validation tags instead of dropping evaluation rows."""
        unknown = set(tags) - self.tags.keys()
        if unknown:
            raise ValueError(
                f"Tags absent from training: {sorted(unknown)}. No rows were removed."
            )
        return [self.tags[t] for t in tags]
