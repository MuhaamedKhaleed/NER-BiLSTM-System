# NER-BiLSTM-System

An end-to-end Named Entity Recognition (NER) solution leveraging Bidirectional Long Short-Term Memory (BiLSTM) networks implemented with TensorFlow and Keras. The system handles tabular token-tag datasets, performs tokenization and padding sequences, trains an entity classification model, and provides an interactive notebook widget for testing inferences.

## Table of Contents
- [Features](#features)
- [Tech Stack](#tech-stack)
- [Architecture](#architecture)
- [Project Structure](#project-structure)
- [Project Setup](#project-setup)
- [Usage](#usage)
- [Interactive Interface](#interactive-interface)
- [Limitations and Known Issues](#limitations-and-known-issues)

---

## Features
- **Data Preprocessing**: Robust loading supporting latin-1 encoding, tokenization, sequence vectorization, and dynamic padding.
- **Label Encoding**: Automated tag transformation using `scikit-learn`'s LabelEncoder.
- **Deep Learning Model**: Bidirectional LSTM network with an initial embedding layer for capturing bidirectional sequence context.
- **Evaluation**: Quantitative metrics monitoring (sparse categorical crossentropy loss and classification accuracy).
- **Interactive Inference UI**: Built-in `ipywidgets` UI for running entity tagging predictions on arbitrary user-provided text directly inside notebook environments.

---

## Tech Stack
- **Language**: Python 3.9+
- **Deep Learning**: TensorFlow / Keras
- **Machine Learning & Preprocessing**: Scikit-Learn
- **Data Handling**: Pandas, NumPy
- **Interface**: IPywidgets, IPython

---

## Architecture
```text
Raw Tokens/Tags -> Tokenizer / Sequence Padding -> Embedding Layer (100-dim) -> Bidirectional LSTM (64 units) -> Dense (Softmax) -> Predicted Tags
```

---

## Project Structure
```text
NER-BiLSTM-System/
├── main.py           # Preprocessing, BiLSTM model training, and interactive interface
├── requirements.txt  # Project dependencies
└── README.md         # Project documentation
```

---

## Project Setup

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/NER-BiLSTM-System.git
cd NER-BiLSTM-System
```

### 2. Create a Virtual Environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Dataset Configuration
Place your NER dataset CSV file in your working directory or adjust the dataset path constant in `main.py`:
```python
df = pd.read_csv('path/to/ner_datasetreference.csv', encoding='latin-1')
```

---

## Usage
Run the training and evaluation script directly:
```bash
python main.py
```

---

## Interactive Interface
When running inside a Jupyter notebook or Google Colab environment, execute the interactive widget block to instantiate the `ipywidgets` dashboard. Enter any sentence, click **Predict**, and inspect mapped entity classifications for each token in real time.

---

## Limitations and Known Issues
- The script expects a labeled dataset CSV (e.g. `ner_datasetreference.csv`) with corresponding sentence, token, and tag columns in the configured path.
- The interactive UI relies on `ipywidgets` and `IPython.display`, which require an interactive notebook frontend to render visually.
- Fixed maximum sequence lengths truncate sentences longer than the configured pad length during inference.