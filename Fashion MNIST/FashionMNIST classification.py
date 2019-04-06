"""# Fashion MNIST dataset

### Import Libraries
"""

import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

"""### Import Fashion MNIST dataset from Keras"""

fashion_mnist = keras.datasets.fashion_mnist

"""### Load dataset"""

(train_data, train_labels),(test_data, test_labels) = fashion_mnist.load_data()

"""### Explore data"""

print(train_data.shape, test_labels.shape)

print(train_labels[0])

plt.figure()
plt.imshow(train_data[0])
plt.grid(False)
plt.xlabel(class_names[train_labels[0]])

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

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

"""## Make predictions"""

predictions = model.predict(test_data)
print(predictions[0])     # Prints the confidence level for each class

print(class_names[np.argmax(predictions[0])])

print(class_names[test_labels[0]])