# -*- coding: utf-8 -*-
"""TextClassification.ipynb

### Import libraries
"""

# from __future__ import absolute_import, division, print_function
import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

"""### Load IMDB dataset"""

imdb = keras.datasets.imdb

(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)

"""### Explore dataset"""

print(train_data[0])

print(len(train_data), len(test_data))

word_index = imdb.get_word_index()

word_index = {k:(v+3) for k,v in word_index.items()} 
word_index["<PAD>"] = 0
word_index["<START>"] = 1
word_index["<UNK>"] = 2  # unknown
word_index["<UNUSED>"] = 3

def decode_review(data):
  return ' '.join([list(word_index.keys())[list(word_index.values()).index(k)] for k in data])

print(decode_review(train_data[0]))

"""### Prepare data"""

train_data = keras.preprocessing.sequence.pad_sequences(train_data,
                                                        value=word_index["<PAD>"],
                                                        padding='post',
                                                        maxlen=256)

test_data = keras.preprocessing.sequence.pad_sequences(test_data,
                                                       value=word_index["<PAD>"],
                                                       padding='post',
                                                       maxlen=256)

print(len(train_data[0]), len(test_data[0]))

"""### Build model"""

# input shape is the vocabulary count used for the movie reviews (10,000 words)
vocab_size = 10000

model = keras.Sequential()
model.add(keras.layers.Embedding(vocab_size, 10))
model.add(keras.layers.Dropout(0.5))
model.add(keras.layers.GlobalAveragePooling1D())
model.add(keras.layers.Dense(10, activation=tf.nn.relu))
model.add(keras.layers.Dropout(0.5))
model.add(keras.layers.Dense(1, activation=tf.nn.sigmoid))

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['acc'])

"""### Split into cross validation set"""

x_val = train_data[:10000]
x_train = train_data[10000:]

y_val = train_labels[:10000]
y_train = train_labels[10000:]

"""### Train model"""

history = model.fit(x_train,y_train,epochs=40,batch_size=512,validation_data=(x_val, y_val),verbose=1)

"""### Evaluate model"""

results = model.evaluate(test_data, test_labels)

print(results)

history_dict = history.history
print(history_dict.keys())



acc = history_dict['acc']
val_acc = history_dict['val_acc']
loss = history_dict['loss']
val_loss = history_dict['val_loss']

epochs = range(1, len(acc) + 1)

# "bo" is for "blue dot"
plt.plot(epochs, loss, 'bo', label='Training loss')
# b is for "solid blue line"
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()