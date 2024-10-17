import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras import optimizers
import numpy as np
import json
import os
from model_vae import VAE

class LatentSpaceLogger(tf.keras.callbacks.Callback):
    def __init__(self, data, labels, latent_dim=2, digits=10, save_path='digit_averages.json'):
        super(LatentSpaceLogger, self).__init__()
        self.data = data
        self.labels = labels
        self.latent_dim = latent_dim
        self.digits = digits
        self.save_path = save_path

    def on_epoch_end(self, epoch, logs=None):
        z_mean, _ = self.model.encoder.predict(self.data, batch_size=128)
        averages = {}
        for digit in range(self.digits):
            digit_z = z_mean[self.labels == digit]
            if len(digit_z) > 0:
                avg_z = np.mean(digit_z, axis=0).tolist()
                averages[str(digit)] = avg_z
        with open(self.save_path, 'w') as f:
            json.dump(averages, f, indent=4)

def main():
    # Load MNIST dataset
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.astype('float32') / 255.
    x_test = x_test.astype('float32') / 255.
    x_train = np.expand_dims(x_train, -1)
    x_test = np.expand_dims(x_test, -1)

    # Initialize VAE with beta
    vae = VAE(latent_dim=2, beta=1.0)  # You can adjust beta here

    # Compile VAE with optimizer
    optimizer = optimizers.Adam(learning_rate=1e-3)  # You can experiment with learning rates
    vae.compile(optimizer=optimizer)

    # Create directories if they don't exist
    os.makedirs('logs/vae', exist_ok=True)
    os.makedirs('vae_model', exist_ok=True)

    # Initialize callbacks
    latent_logger = LatentSpaceLogger(x_test, y_test, save_path='digit_averages.json')

    # Optionally, implement KL warm-up by creating a custom callback or modifying beta during training
    # For simplicity, we'll keep beta constant in this example

    # Train the VAE
    vae.fit(
        x_train,
        epochs=30,
        batch_size=128,
        validation_data=(x_test, None),
        callbacks=[latent_logger]
    )

    # Save the entire model in SavedModel format
    vae.save('vae_model')

if __name__ == '__main__':
    main()