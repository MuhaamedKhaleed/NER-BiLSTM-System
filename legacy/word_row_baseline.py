from __future__ import annotations

import argparse
import logging
from pathlib import Path
import random
from typing import Any, Optional, Union

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer

try:
    import ipywidgets as widgets
    from IPython.display import clear_output, display

    HAS_WIDGETS = True
except ImportError:
    widgets = None  # type: ignore[assignment]
    clear_output = None  # type: ignore[assignment]
    display = None  # type: ignore[assignment]
    HAS_WIDGETS = False

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# Hyperparameters and Configuration Defaults
DEFAULT_DATA_PATH = Path("ner_datasetreference.csv")
DEFAULT_EMBEDDING_DIM = 100
DEFAULT_LSTM_UNITS = 64
DEFAULT_EPOCHS = 10
DEFAULT_BATCH_SIZE = 32
DEFAULT_TEST_SIZE = 0.2
RANDOM_STATE = 42

# Global model and preprocessing state for interactive interface compatibility
tokenizer: Optional[Tokenizer] = None
max_len: int = 1
model: Optional[tf.keras.Model] = None
label_encoder: Optional[LabelEncoder] = None


def set_seeds(seed: int = RANDOM_STATE) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def resolve_default_data_path() -> Path:
    """Find the default dataset location locally or in Colab fallback."""
    local_path = Path("ner_datasetreference.csv")
    colab_path = Path("/content/ner_datasetreference.csv")
    if local_path.is_file():
        return local_path
    if colab_path.is_file():
        return colab_path
    return local_path


def load_data(
    data_path: Union[str, Path] = DEFAULT_DATA_PATH,
    encoding: str = "latin-1",
) -> pd.DataFrame:
    """Load and validate the NER dataset from a CSV file."""
    path = Path(data_path)
    if not path.is_file():
        raise FileNotFoundError(
            f"Dataset not found at '{path.resolve()}'. "
            "Please provide a valid path to ner_datasetreference.csv."
        )
    logger.info("Loading dataset from %s", path)
    df = pd.read_csv(path, encoding=encoding)
    required_cols = {"Word", "Tag"}
    if not required_cols.issubset(df.columns):
        raise ValueError(
            f"Dataset at '{path}' is missing required columns: {required_cols - set(df.columns)}"
        )
    df = df.dropna(subset=["Word", "Tag"]).copy()
    df["Word"] = df["Word"].astype(str)
    df["Tag"] = df["Tag"].astype(str)
    logger.info("Loaded %d valid rows from %s", len(df), path)
    return df


def prepare_data(
    df: pd.DataFrame,
    test_size: float = DEFAULT_TEST_SIZE,
    random_state: int = RANDOM_STATE,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    Tokenizer,
    LabelEncoder,
    int,
    int,
    int,
]:
    """Split data, tokenize words, pad sequences, and encode tag labels."""
    train_df, test_df = train_test_split(
        df, test_size=test_size, random_state=random_state
    )

    X_train_raw = train_df["Word"].values
    y_train_raw = train_df["Tag"].values
    X_test_raw = test_df["Word"].values
    y_test_raw = test_df["Tag"].values

    tok = Tokenizer()
    tok.fit_on_texts(X_train_raw)

    X_train_seq = tok.texts_to_sequences(X_train_raw)
    X_test_seq = tok.texts_to_sequences(X_test_raw)

    train_lengths = [len(seq) for seq in X_train_seq]
    seq_max_len = max(train_lengths) if train_lengths else 1
    if seq_max_len < 1:
        seq_max_len = 1

    X_train_padded = pad_sequences(X_train_seq, maxlen=seq_max_len)
    X_test_padded = pad_sequences(X_test_seq, maxlen=seq_max_len)

    vocab_size = len(tok.word_index) + 1

    lbl_encoder = LabelEncoder()
    y_train_encoded = lbl_encoder.fit_transform(y_train_raw)

    known_classes = set(lbl_encoder.classes_)
    test_mask = [tag in known_classes for tag in y_test_raw]
    if not all(test_mask):
        logger.warning(
            "Found tags in test split not present in train split; filtering out %d rows.",
            len(test_mask) - sum(test_mask),
        )
        X_test_padded = X_test_padded[test_mask]
        y_test_raw = y_test_raw[test_mask]

    y_test_encoded = lbl_encoder.transform(y_test_raw)
    num_tags = len(lbl_encoder.classes_)

    return (
        X_train_padded,
        X_test_padded,
        y_train_encoded,
        y_test_encoded,
        tok,
        lbl_encoder,
        seq_max_len,
        vocab_size,
        num_tags,
    )


def build_model(
    vocab_size: int,
    embedding_dim: int = DEFAULT_EMBEDDING_DIM,
    lstm_units: int = DEFAULT_LSTM_UNITS,
    num_tags: int = 1,
) -> tf.keras.Model:
    """Build and compile the BiLSTM entity classification model."""
    ner_model = tf.keras.Sequential([
        tf.keras.layers.Embedding(vocab_size, embedding_dim),
        tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(units=lstm_units, return_sequences=False)
        ),
        tf.keras.layers.Dense(num_tags, activation="softmax"),
    ])
    ner_model.compile(
        loss="sparse_categorical_crossentropy",
        optimizer="adam",
        metrics="accuracy",
    )
    return ner_model


def train_model(
    ner_model: tf.keras.Model,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    epochs: int = DEFAULT_EPOCHS,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> tf.keras.callbacks.History:
    """Train the model on the training set and validate on the test set."""
    logger.info("Starting training for %d epochs...", epochs)
    history = ner_model.fit(
        X_train,
        y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(X_val, y_val),
    )
    return history


def predict_words(
    words: list[str],
    ner_model: tf.keras.Model,
    tok: Tokenizer,
    lbl_encoder: LabelEncoder,
    seq_max_len: int,
) -> list[str]:
    """Predict entity tags for a list of words."""
    if not words:
        return []
    sequences = tok.texts_to_sequences(words)
    padded = pad_sequences(sequences, maxlen=seq_max_len)
    predictions = ner_model.predict(padded, verbose=0)
    predicted_indices = predictions.argmax(axis=-1)
    predicted_labels = lbl_encoder.inverse_transform(predicted_indices)
    return list(predicted_labels)


def predict_entities(b: Optional[Any] = None) -> None:
    """Widget callback to predict entity tags for the entered text."""
    if output_area is None:
        return
    with output_area:
        if clear_output is not None:
            clear_output()
        if text_input is None:
            print("Widget text input is unavailable.")
            return

        input_text = text_input.value.strip()
        if not input_text:
            print("Please enter some text.")
            return

        if model is None or tokenizer is None or label_encoder is None:
            print("Model and preprocessing artifacts are not initialized.")
            return

        words = input_text.split()
        predicted_labels = predict_words(
            words=words,
            ner_model=model,
            tok=tokenizer,
            lbl_encoder=label_encoder,
            seq_max_len=max_len,
        )

        print("Entities Identified:")
        for word, label in zip(words, predicted_labels):
            print(f"Word: {word}, Label: {label}")


def setup_interactive_widgets() -> tuple[Optional[Any], Optional[Any], Optional[Any]]:
    """Initialize ipywidgets UI components if ipywidgets is available."""
    if not HAS_WIDGETS or widgets is None:
        return None, None, None

    text_in = widgets.Textarea(
        value="",
        placeholder="Enter text here...",
        description="Text:",
        disabled=False,
    )
    btn = widgets.Button(description="Predict")
    out = widgets.Output()

    btn.on_click(predict_entities)
    return text_in, btn, out


def display_widgets() -> None:
    """Display the interactive UI widgets if supported."""
    if HAS_WIDGETS and display is not None and text_input is not None:
        display(text_input, predict_button, output_area)


# Initialize widget objects if ipywidgets is installed
text_input, predict_button, output_area = setup_interactive_widgets()


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train and evaluate BiLSTM Named Entity Recognition (NER) model."
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default=str(resolve_default_data_path()),
        help="Path to the NER CSV dataset (default: ner_datasetreference.csv)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=DEFAULT_EPOCHS,
        help=f"Number of training epochs (default: {DEFAULT_EPOCHS})",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help=f"Training batch size (default: {DEFAULT_BATCH_SIZE})",
    )
    parser.add_argument(
        "--embedding-dim",
        type=int,
        default=DEFAULT_EMBEDDING_DIM,
        help=f"Word embedding dimension (default: {DEFAULT_EMBEDDING_DIM})",
    )
    parser.add_argument(
        "--lstm-units",
        type=int,
        default=DEFAULT_LSTM_UNITS,
        help=f"Number of BiLSTM units (default: {DEFAULT_LSTM_UNITS})",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=DEFAULT_TEST_SIZE,
        help=f"Fraction of data for testing (default: {DEFAULT_TEST_SIZE})",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=RANDOM_STATE,
        help=f"Random seed for reproducibility (default: {RANDOM_STATE})",
    )
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> None:
    """Run training, evaluation, and interactive widget setup."""
    global tokenizer, max_len, model, label_encoder, text_input, predict_button, output_area

    args = parse_args(argv)
    set_seeds(args.random_state)

    data_path = Path(args.data_path)
    if not data_path.is_file():
        logger.error(
            "Dataset not found at '%s'. Please provide a valid CSV dataset path using --data-path.",
            data_path.resolve(),
        )
        return

    df = load_data(data_path)

    (
        X_train_padded,
        X_test_padded,
        y_train_encoded,
        y_test_encoded,
        tokenizer,
        label_encoder,
        max_len,
        vocab_size,
        num_tags,
    ) = prepare_data(
        df,
        test_size=args.test_size,
        random_state=args.random_state,
    )

    logger.info(
        "Vocabulary size: %d, Number of tags: %d, Max sequence length: %d",
        vocab_size,
        num_tags,
        max_len,
    )

    model = build_model(
        vocab_size=vocab_size,
        embedding_dim=args.embedding_dim,
        lstm_units=args.lstm_units,
        num_tags=num_tags,
    )

    train_model(
        ner_model=model,
        X_train=X_train_padded,
        y_train=y_train_encoded,
        X_val=X_test_padded,
        y_val=y_test_encoded,
        epochs=args.epochs,
        batch_size=args.batch_size,
    )

    loss, accuracy = model.evaluate(X_test_padded, y_test_encoded)
    print(f"Test Loss: {loss}, Test Accuracy: {accuracy}")

    # Re-link widgets to newly trained artifacts if available
    text_input, predict_button, output_area = setup_interactive_widgets()

    try:
        from IPython import get_ipython

        if get_ipython() is not None:
            display_widgets()
    except Exception:
        pass


if __name__ == "__main__":
    main()
