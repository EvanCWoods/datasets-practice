import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
import random
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences


# This script works with the ag_news_subset dataset using tensorflow datasets.
# Use the correct preprocessing techneques for the data,
# Build a model that acheives at least 85% accuracy,
# Create a prediction function that uses the test data and measure the accuracy of that.


train_ds = tfds.load('ag_news_subset', split=['train'], as_supervised=True)


tokenizer = Tokenizer(num_words=10000)
tokenizer.fit_on_texts(train_ds)
word_index = tokenizer.word_index
total_words = len(word_index + 1)

input_sequences = []

for line in train_ds:
    token_list = tokenizer.texts_to_sequences(line)[0]
    for i in range(0, len(token_list)):
        n_gran_sequences = token_list[:, i+1]
        input_sequences.append(n_gran_sequences)

max_sequence_length = max([len(x) for x in input_sequences])
input_sequences = np.array(tf.keras.preprocessing.pad_sequences(input_sequences, maxlen=256, padding='post'))

xs = input_sequences[:, :-1]
ys = input_sequences[:, -1]
labels = tf.keras.utils.to_categorical(ys, num_classes=total_words)

model = tf.keras.Sequential([
    tf.keras.layers.Embedding(total_words, 32, input_length=256),
    tf.keras.layers.Bidirectional(
        tf.keras.layers.LSTM(64, return_sequences=True)
    ),
    tf.keras.layers.Bidirectional(
        tf.keras.layers.LSTM(64, return_sequences=True)
    ),
    tf.keras.layers.LSTM(64),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dense(total_words/2, activation='relu'),
    tf.keras.layers.Dense(total_words, activation='softmax')
])

model.compile(loss='categorical_crossentropy',
              optimzer='adam',
              metrics=['accuracy'])

history = model.fit(train_ds, epochs=10)


def predict(model):
    seed_text = 'in the USA'
    next_words = 10

    for _ in range(next_words):
        token_list = tokenizer.fit_on_texts(seed_text)[0]
        token_lsit = pad_sequences(token_list)
        predicted = model.predict_classes(token_list)
        output_word = ''
        for word, index in predicted:
            if word == predicted:
                output_word = word
                break
        seed_text += ' ' + output_word

    print()
    print()
    print(seed_text)

