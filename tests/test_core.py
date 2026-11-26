from pathlib import Path

import pytest

from ner_bilstm.core import Sentence, Vocabulary, read_sentences, split_sentences


def test_grouped_csv_keeps_boundaries_and_punctuation(tmp_path: Path) -> None:
    path = tmp_path / "data.csv"
    path.write_text("Sentence #,Word,Tag,POS\ns1,Hello,O,N\n,!,O,P\ns2,Cairo,B-LOC,N\n")
    sentences = read_sentences(path)
    assert sentences == [
        Sentence("s1", ("Hello", "!"), ("O", "O")),
        Sentence("s2", ("Cairo",), ("B-LOC",)),
    ]


@pytest.mark.parametrize(
    "text",
    [
        "",
        "Word,Tag\na,O\n",
        "Sentence #,Word,Tag,Word\ns,a,O,b\n",
        "Sentence #,Word,Tag\n,a,O\n",
        "Sentence #,Word,Tag\ns,,O\n",
        "Sentence #,Word,Tag\ns,a,\n",
        "Sentence #,Word,Tag\ns,a\n",
        "Sentence #,Word,Tag\ns,a,O,extra\n",
        "Sentence #,Word,Tag\ns1,a,O\ns2,b,O\ns1,c,O\n",
    ],
)
def test_invalid_csv(tmp_path: Path, text: str) -> None:
    path = tmp_path / "data.csv"
    path.write_text(text)
    with pytest.raises(ValueError):
        read_sentences(path)


def test_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        read_sentences(tmp_path / "missing")


def test_explicit_encoding(tmp_path: Path) -> None:
    path = tmp_path / "data.csv"
    path.write_bytes("Sentence #,Word,Tag\ns,café,O\n".encode("latin-1"))
    assert read_sentences(path, "latin-1")[0].words == ("café",)
    with pytest.raises(UnicodeError):
        read_sentences(path)


def test_whole_sentence_split_no_mutation() -> None:
    original = [Sentence(str(i), ("hello", str(i)), ("O", "O")) for i in range(10)]
    train, val = split_sentences(original, 0.2, 42)
    assert len(train) == 8 and len(val) == 2
    assert {s.key for s in train}.isdisjoint(s.key for s in val)
    assert train + val != original
    assert [s.key for s in original] == list(map(str, range(10)))
    assert (train, val) == split_sentences(original, 0.2, 42)


@pytest.mark.parametrize("fraction", [0, 1, -0.1, float("nan"), float("inf")])
def test_invalid_fraction(fraction: float) -> None:
    with pytest.raises(ValueError):
        split_sentences([Sentence("a", ("x",), ("O",))] * 2, fraction)


def test_too_few_and_duplicate_keys() -> None:
    sentence = Sentence("a", ("x",), ("O",))
    with pytest.raises(ValueError):
        split_sentences([sentence])
    with pytest.raises(ValueError, match="unique"):
        split_sentences([sentence, sentence])


def test_train_only_vocab_oov_alignment_unknown_tags() -> None:
    vocab = Vocabulary.fit([Sentence("a", ("Cairo", "!"), ("B-LOC", "O"))])
    assert vocab.encode_words(("Cairo", "unseen", "!")) == [3, 1, 2]
    assert "unseen" not in vocab.words
    assert vocab.encode_tags(("O", "B-LOC")) == [2, 1]
    with pytest.raises(ValueError, match="No rows were removed"):
        vocab.encode_tags(("B-PER",))


@pytest.mark.parametrize(
    "words,tags", [({}, {"O": 1}), ({"x": 1}, {"O": 1}), ({"x": 2}, {"O": 0})]
)
def test_bad_vocab(words: dict[str, int], tags: dict[str, int]) -> None:
    with pytest.raises(ValueError):
        Vocabulary(words, tags)


@pytest.mark.parametrize(
    "key,words,tags",
    [("", ("x",), ("O",)), ("a", (), ()), ("a", ("x",), ()), ("a", (" ",), ("O",))],
)
def test_invalid_sentence(
    key: str, words: tuple[str, ...], tags: tuple[str, ...]
) -> None:
    with pytest.raises(ValueError):
        Sentence(key, words, tags)


@pytest.mark.parametrize(
    "words,tags",
    [
        ({"x": 2.0}, {"O": 1}),
        ({"x": 2}, {"O": True}),
        ({"": 2}, {"O": 1}),
        ([], {"O": 1}),
    ],
)
def test_vocabulary_rejects_untyped_json(words, tags) -> None:
    with pytest.raises(ValueError):
        Vocabulary(words, tags)
