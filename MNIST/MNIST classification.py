# -*- coding: utf-8 -*-
"""MNIST Classification.ipynb

# MNIST dataset

### Import Libraries
"""

import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

"""### Import MNIST dataset from Keras"""

mnist = keras.datasets.mnist

"""### Load dataset"""

(train_data, train_labels),(test_data, test_labels) = mnist.load_data()

"""### Explore data"""

print(train_data.shape, test_labels.shape)

print(train_labels[0])

plt.figure()
plt.imshow(train_data[0])
plt.grid(False)

"""### Preprocess data"""

# Scaling
train_data = train_data / 255.0
test_data = test_data / 255.0

"""## Build the model

### Setup the layers
"""

model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation=tf.nn.relu),
    keras.layers.Dense(128, activation=tf.nn.relu),
    keras.layers.Dense(10, activation=tf.nn.softmax)
])

"""### Compile the model"""

model.compile(optimizer=tf.train.AdamOptimizer(), 
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

"""### Train the model"""

model.fit(train_data, train_labels, epochs=5)

"""### Evaluate accuracy on test data"""

test_loss, test_acc = model.evaluate(test_data, test_labels)

print('Test accuracy:', test_acc)

"""The accuracy on the test dataset can be a little less than the accuracy on the training dataset. This gap between training accuracy and test accuracy is an example of overfitting. Overfitting is when a machine learning model performs worse on new data than on their training data.

## Make predictions
"""

predictions = model.predict(test_data)
print(predictions[0] )    # Prints the confidence level for each class

print(np.argmax(predictions[0]))

print(test_labels[0])
