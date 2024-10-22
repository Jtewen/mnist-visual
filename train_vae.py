import tensorflow as tf
from tensorflow import keras
from keras.datasets import mnist
import numpy as np
import json
from model_vae import VAE

class LatentSpaceLogger(tf.keras.callbacks.Callback):
    def __init__(self, data, labels, latent_dim=2, digits=10, save_path='digit_averages.json'):
        super().__init__()
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
            if digit_z.size:
                avg_z = np.mean(digit_z, axis=0).tolist()
                averages[str(digit)] = avg_z
        with open(self.save_path, 'w') as f:
            json.dump(averages, f, indent=4)

def main():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.astype('float32') / 255.
    x_test = x_test.astype('float32') / 255.
    x_train = np.expand_dims(x_train, -1)
    x_test = np.expand_dims(x_test, -1)

    vae = VAE(latent_dim=2, beta=1.0)
    latent_logger = LatentSpaceLogger(x_train, y_train, save_path='digit_averages.json')

    vae.fit(
        x_train,
        epochs=30,
        batch_size=128,
        validation_data=(x_test, None),
        callbacks=[latent_logger]
    )

    vae.save('vae_model')

if __name__ == '__main__':
    main()