# -*- coding: utf-8 -*-
'''
https://www.tensorflow.org/guide/keras
에 있는 소스임.
'''

import tensorflow as tf
from tensorflow.keras import layers

print(tf.VERSION)
print(tf.keras.__version__)

# model = tf.keras.Sequential()
# # Adds a densely-connected layer with 64 units to the model:
# model.add(layers.Dense(64, activation='relu'))
# # Add another:
# model.add(layers.Dense(64, activation='relu'))
# # Add a softmax layer with 10 output units:
# model.add(layers.Dense(10, activation='softmax'))


model = tf.keras.Sequential([
# Adds a densely-connected layer with 64 units to the model:
layers.Dense(64, activation='relu', input_shape=(32,)),
# Add another:
layers.Dense(64, activation='relu'),
# Add a softmax layer with 10 output units:
layers.Dense(10, activation='softmax')])

model.compile(optimizer=tf.train.AdamOptimizer(0.001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])


import numpy as np

data = np.random.random((1000, 32))
labels = np.random.random((1000, 10))

model.fit(data, labels, epochs=10, batch_size=32)

