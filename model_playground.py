from tensorflow import keras
from keras import Sequential, Model
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Input
import tensorflow as tf
import numpy as np
import os

class Playground:
    def __init__(self):
        self.input_shape = (28, 28, 1)  # MNIST images are 28x28 pixels with 1 channel
        self.output_shape = 10  # 10 classes for digits 0-9
        self.model_path = 'playground_model.h5'
        self.model = self.build_model()  # Call build_model in __init__
    
    def build_model(self):
        model = Sequential([
            Input(shape=self.input_shape),
            Conv2D(8, (3, 3), activation='relu', name='conv1'),
            MaxPooling2D((2, 2), name='pool1'),
            Conv2D(16, (3, 3), activation='relu', name='conv2'),
            MaxPooling2D((2, 2), name='pool2'),
            Flatten(),
            Dense(128, activation='relu'),
            Dense(self.output_shape, activation='softmax')
        ])
        model.compile(optimizer='adam', 
                     loss='sparse_categorical_crossentropy', 
                     metrics=['accuracy'])
        return model

    def predict(self, processed_input):
        if not self.model:
            raise ValueError("Model not initialized")
        return self.model.predict(processed_input)

    def train(self, x_train, y_train, epochs=5):
        x_train = x_train.reshape(-1, 28, 28, 1) / 255.0  # Normalize and reshape
        self.model.fit(x_train, y_train, epochs=epochs)
        self.model.save(self.model_path)
