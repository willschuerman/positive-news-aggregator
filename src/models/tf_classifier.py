# %%
import os
import pandas as pd
from ast import literal_eval
import pickle

# Deep learning libraries and APIs
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# %%
# Import data
data_dir = os.getcwd().replace('src/models', 'data/interim/')
data = pd.read_csv(data_dir + 'abcnews_labeled.csv',
                   converters={'Comments': literal_eval})

# Filtering out unlabeled data points
data = data.loc[data.label.isin([0, 1]), :]
data = data.reset_index()
# %%
# store headlines and labels in respective lists
text = list(data['text'])
labels = list(data['label'])
# sentences
training_text = text[0:15000]
testing_text = text[15000:]
# labels
training_labels = labels[0:15000]
testing_labels = labels[15000:]
# %%
# preprocess
tokenizer = Tokenizer(num_words=10000, oov_token="<OOV>")
tokenizer.fit_on_texts(training_text)
word_index = tokenizer.word_index
training_sequences = tokenizer.texts_to_sequences(training_text)
training_padded = pad_sequences(
    training_sequences, maxlen=120, padding='post', truncating='post')
testing_sequences = tokenizer.texts_to_sequences(testing_text)
testing_padded = pad_sequences(
    testing_sequences, maxlen=120, padding='post', truncating='post')
# convert lists into numpy arrays to make it work with TensorFlow
training_padded = np.array(training_padded)
training_labels = np.array(training_labels)
testing_padded = np.array(testing_padded)
testing_labels = np.array(testing_labels)
# %%
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(10000, 16, input_length=120),
    tf.keras.layers.GlobalAveragePooling1D(),
    tf.keras.layers.Dense(24, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])
# compile the model
model.compile(loss='binary_crossentropy',
              optimizer='adam', metrics=['accuracy'])
model.summary()
# %%
num_epochs = 10
history = model.fit(training_padded,
                    training_labels,
                    epochs=num_epochs,
                    validation_data=(testing_padded, testing_labels),
                    verbose=2)
# %%
new_headline = [
    "Analysis: This Democratic leader just broke the first rule of politics",
    "Kevin McCarthy previews Republicans' plans for the majority -- starting at the border",
    "Fetterman sues to have mail-in ballots counted even if not signed with valid date"
]
# prepare the sequences of the sentences in question
sequences = tokenizer.texts_to_sequences(new_headline)
padded_seqs = pad_sequences(sequences, maxlen=120,
                            padding='post', truncating='post')
print(model.predict(padded_seqs))
# %%
