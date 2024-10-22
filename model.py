import os
import time
import tensorflow as tf
from tensorflow import keras
from keras import layers, models
import matplotlib.pyplot as plt
import numpy as np
import logging
from PIL import Image
import io
import base64

class MNISTModel:
    def __init__(self):
        self.model = self.build_model()
        self.model_path = 'mnist_model.h5'
        self.layer_models = {}
    
    def setup_layer_models(self):
        conv_layers = [layer for layer in self.model.layers if isinstance(layer, tf.keras.layers.Conv2D)]
        for i, layer in enumerate(conv_layers):
            self.layer_models[f'conv{i+1}'] = tf.keras.Model(inputs=self.model.input, outputs=layer.output)
    
    def build_model(self):
        model = models.Sequential([
            layers.Input(shape=(28, 28, 1)),
            layers.Conv2D(8, (3, 3), activation='relu', name='conv1'),
            layers.MaxPooling2D((2, 2), name='pool1'),
            layers.Conv2D(16, (3, 3), activation='relu', name='conv2'),
            layers.MaxPooling2D((2, 2), name='pool2'),
            layers.Flatten(),
            layers.Dense(128, activation='relu', name='fc1'),
            layers.Dense(10, activation='softmax', name='output')
        ])
        model.compile(optimizer='adam',
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])
        return model

    def train(self, x_train, y_train, epochs=5):
        self.model.fit(x_train, y_train, epochs=epochs)
        self.model.save(self.model_path)
        logging.info(f"Model weights saved to {self.model_path}.")
        
        # Setup layer models before visualizing filters
        self.setup_layer_models()
        
        # Visualize filters for both convolutional layers
        self.visualize_filters(x_train, layer_index=0)
        self.visualize_filters(x_train, layer_index=1)

    def predict(self, image):
        return self.model.predict(image)

    def load_model(self):
        self.model.load_weights(self.model_path)

    def get_feature_maps(self, x_data):
        feature_maps = {}
        
        for name, layer_model in self.layer_models.items():
            activations = layer_model.predict(x_data)
            processed_maps = []
            
            for channel in range(activations.shape[-1]):
                fmap = activations[0, :, :, channel]
                fmap = (fmap - fmap.min()) / (fmap.max() - fmap.min())
                # Invert the colors
                fmap = 1 - fmap
                fmap = (fmap * 255).astype(np.uint8)
                
                # Create a larger image by repeating pixels
                large_fmap = np.repeat(np.repeat(fmap, 20, axis=0), 20, axis=1)
                
                img = Image.fromarray(large_fmap)
                img = img.resize((60, 60), Image.Resampling.NEAREST)
                buffered = io.BytesIO()
                img.save(buffered, format="PNG")
                img_str = base64.b64encode(buffered.getvalue()).decode()
                processed_maps.append(img_str)
                
            feature_maps[name] = processed_maps
        
        return feature_maps
    
    def visualize_filters(self, x_data, layer_index=0):
        layer_name = f'conv{layer_index+1}'
        if layer_name not in self.layer_models:
            return None
        
        filters_dir = 'static/filters'
        if not os.path.exists(filters_dir):
            os.makedirs(filters_dir)
        
        layer = self.model.get_layer(layer_name)
        weights = layer.get_weights()[0]
        
        for i in range(weights.shape[-1]):
            filt = weights[:, :, 0, i]
            filt = (filt - filt.min()) / (filt.max() - filt.min())
            # Invert the colors
            filt = 1 - filt
            filt = (filt * 255).astype(np.uint8)
            
            large_filt = np.repeat(np.repeat(filt, 20, axis=0), 20, axis=1)
            
            img = Image.fromarray(large_filt)
            img = img.resize((60, 60), Image.Resampling.NEAREST)
            img.save(os.path.join(filters_dir, f'{layer_name}_filter_{i}.png'))
