!pip install pandas==1.5.3

import pandas as pd

# Try reading the file with 'latin-1' encoding
df = pd.read_csv('/content/ner_datasetreference.csv', encoding='latin-1')
df

#split the data
from sklearn.model_selection import train_test_split
train_df, test_df = train_test_split(df, test_size=0.2)
X_train = train_df['Word'].values
y_train = train_df['Tag'].values
X_test = test_df['Word'].values
y_test = test_df['Tag'].values

import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder

# Load your annotated dataset

# Tokenize the text
tokenizer = Tokenizer()
tokenizer.fit_on_texts(X_train)

# Convert text to sequences
X_train_seq = tokenizer.texts_to_sequences(X_train)
X_test_seq = tokenizer.texts_to_sequences(X_test)

# Pad sequences to a fixed length
max_len = max(len(seq) for seq in X_train_seq)
X_train_padded = pad_sequences(X_train_seq, maxlen=max_len)
X_test_padded = pad_sequences(X_test_seq, maxlen=max_len)

# Calculate vocab_size
vocab_size = len(tokenizer.word_index) + 1 # +1 for the padding token

# Define embedding dimension
embedding_dim = 100 # Example value, adjust as needed

# Encode labels to integers
label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train)
y_test_encoded = label_encoder.transform(y_test)

# You also need to define num_tags
num_tags = len(set(y_train)) # Assuming y_train contains your tags

# Define the model
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size, embedding_dim),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(units=64, return_sequences=False)), # Remove return_sequences=True
    tf.keras.layers.Dense(num_tags, activation='softmax') # Remove TimeDistributed layer
])

# Compile the model
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
# Use encoded labels for training
model.fit(X_train_padded, y_train_encoded, epochs=10, validation_data=(X_test_padded, y_test_encoded))

#evaluation
loss, accuracy = model.evaluate(X_test_padded, y_test_encoded)
print(f"Test Loss: {loss}, Test Accuracy: {accuracy}")

# prompt: interface

#@title NER Model Interface

import ipywidgets as widgets
from IPython.display import display, clear_output

# Create input text area
text_input = widgets.Textarea(
    value='',
    placeholder='Enter text here...',
    description='Text:',
    disabled=False
)

# Create button to trigger prediction
predict_button = widgets.Button(description='Predict')

# Create output area
output_area = widgets.Output()

def predict_entities(b):
  with output_area:
    clear_output()
    input_text = text_input.value
    if not input_text:
      print("Please enter some text.")
      return

    # Tokenize and pad the input text
    input_seq = tokenizer.texts_to_sequences([input_text])
    input_padded = pad_sequences(input_seq, maxlen=max_len)

    # Predict entities
    predictions = model.predict(input_padded)[0]
    predicted_labels = [label_encoder.inverse_transform([label_index])[0] for label_index in predictions.argmax(axis=-1)]


    # Print results
    print("Entities Identified:")
    for word, label in zip(input_text.split(), predicted_labels):
      print(f"Word: {word}, Label: {label}")


predict_button.on_click(predict_entities)

# Display the widgets
display(text_input, predict_button, output_area)


