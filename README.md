# NER-BiLSTM-System

An end-to-end Named Entity Recognition (NER) solution leveraging Bidirectional Long Short-Term Memory (BiLSTM) networks implemented with TensorFlow and Keras. The system handles tabular token-tag datasets, performs tokenization and padding sequences, trains an entity classification model, and provides an interactive notebook widget for testing inferences.

## Table of Contents
- [Features](#features)
- [Tech Stack](#tech-stack)
- [Architecture](#architecture)
- [Project Setup](#project-setup)
- [Usage](#usage)
- [Interactive Interface](#interactive-interface)

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
text
Raw Tokens/Tags -> Tokenizer / Sequence Padding -> Embedding Layer (100-dim) -> Bidirectional LSTM (64 units) -> Dense (Softmax) -> Predicted Tags


---

## Project Setup

### 1. Clone the Repository
bash
git clone https://github.com/your-username/NER-BiLSTM-System.git
cd NER-BiLSTM-System


### 2. Create a Virtual Environment
bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate


### 3. Install Dependencies
bash
pip install -r requirements.txt


### 4. Dataset Configuration
Place your NER dataset CSV file in your working directory or adjust the dataset path in the script:
python
df = pd.read_csv('path/to/ner_datasetreference.csv', encoding='latin-1')


---

## Usage
Run the training and evaluation script directly or execute the cells within a Jupyter Notebook / Google Colab:
bash
python ner_bilstm.py


---

## Interactive Interface
When running inside a notebook, execute the interface cell to instantiate the `ipywidgets` dashboard. Enter any sentence, click **Predict**, and inspect mapped entity classifications for each token in real time.