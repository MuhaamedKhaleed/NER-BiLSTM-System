import json
from pathlib import Path

import pytest

from ner_bilstm.core import Sentence, Vocabulary
from ner_bilstm.model import arrays, backend, build_model, predict, token_scores, train

np = pytest.importorskip("numpy")
pytest.importorskip("tensorflow")


@pytest.fixture()
def batch() -> list[Sentence]:
    return [
        Sentence("a", ("Mona", "visits", "Cairo"), ("B-PER", "O", "B-LOC")),
        Sentence("b", ("Cairo",), ("B-LOC",)),
    ]


def test_actual_backend_shape_padding_and_loss(batch: list[Sentence]) -> None:
    tf = backend()
    tf.keras.utils.set_random_seed(42)
    vocab = Vocabulary.fit(batch)
    x, y, weights = arrays(batch, vocab)
    assert x.shape == y.shape == weights.shape == (2, 3)
    assert x[1].tolist() == [2, 0, 0]
    assert weights[1].tolist() == [1, 0, 0]
    model = build_model(vocab, 4, 3)
    output = model(x, training=False).numpy()
    assert output.shape == (2, 3, 4)
    assert np.isfinite(output).all()
    # Right padding must not change preceding real-token predictions.
    extended = model(np.pad(x, ((0, 0), (0, 2))), training=False).numpy()
    np.testing.assert_allclose(output[:, :1], extended[:, :1], atol=1e-6)
    loss = model.train_on_batch(x, y, sample_weight=weights)
    assert np.isfinite(loss)
    scores = token_scores(model, batch, vocab)
    assert scores["tokens"] == 4
    assert 0 <= scores["token_accuracy"] <= 1


def test_train_save_load_infer_synthetic_only(
    tmp_path: Path, batch: list[Sentence]
) -> None:
    output = tmp_path / "synthetic-model"
    validation = [Sentence("v" + s.key, s.words, s.tags) for s in batch]
    scores = train(
        batch, validation, output, epochs=1, batch_size=2, embedding=4, units=3
    )
    assert scores["tokens"] == 4
    result = predict(output, ("Mona", "unseen", "Cairo"))
    assert [r["word"] for r in result] == ["Mona", "unseen", "Cairo"]
    assert all(r["tag"] in {"B-PER", "B-LOC", "O"} for r in result)
    assert json.loads((output / "validation.json").read_text()) == scores
    with pytest.raises(ValueError, match="already exists"):
        train(batch, validation, output)
    with pytest.raises(ValueError):
        predict(output, ())
    # A corrupt mapping cannot silently reinterpret model output.
    (output / "vocabulary.json").write_text(
        json.dumps({"words": {"Mona": 2}, "tags": {"O": 1}})
    )
    with pytest.raises(ValueError, match="tag count"):
        predict(output, ("Mona",))


def test_invalid_ml_inputs(tmp_path: Path, batch: list[Sentence]) -> None:
    vocab = Vocabulary.fit(batch)
    with pytest.raises(ValueError, match="disjoint"):
        train(batch, batch, tmp_path / "overlap")
    with pytest.raises(ValueError):
        arrays([], vocab)
    with pytest.raises(ValueError):
        build_model(vocab, 0)
    with pytest.raises(ValueError):
        train(batch, batch, tmp_path / "new", epochs=0)
