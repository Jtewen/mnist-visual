import os
import time
import tensorflow as tf
from tensorflow import keras
from keras import layers, models

import matplotlib.pyplot as plt
import numpy as np

class MNISTModel:
    def __init__(self, model_path='classifier/mnist_model.h5'):
        self.model_path = model_path
        self.model = self.build_model()  # Build the neural network architecture
        try:
            self.model.load_weights(self.model_path)  # Load pre-trained weights if available
            print("Model weights loaded.")
        except:
            print("No saved model found. Please train the model.")

    def build_model(self):
        model = models.Sequential([
            # Reshape input data to add a single channel dimension (28x28x1)
            layers.Reshape((28, 28, 1), input_shape=(28, 28)),
            # First convolutional layer: 16 filters, 3x3 kernel, ReLU activation
            layers.Conv2D(8, (3, 3), activation='relu'),
            # First max pooling layer: reduces spatial dimensions by half
            layers.MaxPooling2D((2, 2)),
            # Second convolutional layer: 32 filters, 3x3 kernel, ReLU activation
            layers.Conv2D(16, (3, 3), activation='relu'),
            # Second max pooling layer
            layers.MaxPooling2D((2, 2)),
            # Flatten the 2D feature maps to a 1D feature vector for dense layers
            layers.Flatten(),
            # Fully connected layer: 128 neurons, ReLU activation for non-linearity
            layers.Dense(128, activation='relu'),
            # Output layer: 10 neurons (one for each digit class), softmax activation for probability distribution
            layers.Dense(10, activation='softmax')
        ])
        # Compile the model with Adam optimizer and sparse categorical cross-entropy loss
        model.compile(optimizer='adam',
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])  # Track accuracy during training
        return model

    def train(self, x_train, y_train, epochs=10):
        self.model.fit(x_train, y_train, epochs=epochs)  # Fit the model to the training data
        self.model.save(self.model_path)  # Save the trained model weights
        print(f"Model weights saved to {self.model_path}.")
        while not os.path.exists(self.model_path):
            time.sleep(1)
        self.visualize_filters(x_data=x_train, layer_index=1, num_filters=6)
        self.visualize_filters(x_data=x_train, layer_index=3, num_filters=6)

    def predict(self, image):
        return self.model.predict(image)  # Returns the predicted probabilities for each class

    def load_model(self):
        try:
            self.model.load_weights(self.model_path)  # Load weights if they exist
            print("Model weights loaded successfully.")
        except:
            print("No saved model weights found. Please train the model.")

    def get_most_active_filters(self, x_data, layer_index=0, num_filters=6):
        # Create a model that outputs the activations of the specified layer
        layer_model = tf.keras.Model(inputs=self.model.input, outputs=self.model.layers[layer_index].output)
        
        # Get the activations for the input data
        activations = layer_model.predict(x_data)
        
        # Calculate the average activation for each filter
        avg_activations = np.mean(activations, axis=(0, 1, 2))  # Average over height, width, and batch size
        
        # Get the indices of the most active filters
        most_active_indices = np.argsort(avg_activations)[-num_filters:][::-1]  # Get top filters
        return most_active_indices

    def visualize_filters(self, x_data, layer_index=0, num_filters=6, save_dir='static/filters'):
        # Create the directory if it doesn't exist
        os.makedirs(save_dir, exist_ok=True)

        # Get the most active filter indices
        active_filter_indices = self.get_most_active_filters(x_data, layer_index, num_filters)
        
        # Get the weights of the specified convolutional layer
        filters, biases = self.model.layers[layer_index].get_weights()
        
        # Normalize filter values to 0-1 for better visualization
        filters = (filters - filters.min()) / (filters.max() - filters.min())
        
        for i, filter_index in enumerate(active_filter_indices):
            # Create a figure for each filter
            plt.figure(figsize=(2, 2))
            plt.imshow(filters[:, :, 0, filter_index], cmap='gray')  # Display the first channel of the filter
            plt.axis('off')
            # Save the figure
            plt.savefig(os.path.join(save_dir, f'filter_{layer_index}_{i}.png'), bbox_inches='tight', pad_inches=0)
            plt.close()  # Close the figure to free memory

    def get_feature_maps(self, image):
        """
        Get the feature maps from the convolutional layers for a given input image.
        
        Parameters:
        - image: A numpy array representing the input image (should be preprocessed).
        
        Returns:
        - A dictionary of feature maps from the convolutional layers.
        """
        # Create a model that outputs the activations of all convolutional layers
        layer_outputs = {layer.name: layer.output for layer in self.model.layers if isinstance(layer, layers.Conv2D)}
        feature_map_model = tf.keras.Model(inputs=self.model.input, outputs=list(layer_outputs.values()))
        
        # Get the feature maps for the input image
        feature_maps = feature_map_model.predict(image)
        
        # Return a dictionary mapping layer names to their feature maps
        return {name: fmap for name, fmap in zip(layer_outputs.keys(), feature_maps)}
