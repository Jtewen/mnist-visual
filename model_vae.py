import tensorflow as tf
from tensorflow import keras
from keras import layers, models
import numpy as np

class VAE(tf.keras.Model):
    def __init__(self, latent_dim=2, beta=1.0, **kwargs):
        super(VAE, self).__init__(**kwargs)
        self.latent_dim = latent_dim
        self.beta = beta
        self.encoder = self.build_encoder()
        self.decoder = self.build_decoder()

    def build_encoder(self):
        encoder_inputs = layers.Input(shape=(28, 28, 1))
        x = layers.Conv2D(32, 3, activation='relu', strides=2, padding='same')(encoder_inputs)
        x = layers.Conv2D(64, 3, activation='relu', strides=2, padding='same')(x)
        x = layers.Flatten()(x)
        x = layers.Dense(16, activation='relu')(x)
        z_mean = layers.Dense(self.latent_dim, name='z_mean')(x)
        z_log_var = layers.Dense(self.latent_dim, name='z_log_var')(x)
        return models.Model(encoder_inputs, [z_mean, z_log_var], name='encoder')

    def build_decoder(self):
        latent_inputs = layers.Input(shape=(self.latent_dim,))
        x = layers.Dense(7 * 7 * 64, activation='relu')(latent_inputs)
        x = layers.Reshape((7, 7, 64))(x)
        x = layers.Conv2DTranspose(64, 3, activation='relu', strides=2, padding='same')(x)
        x = layers.Conv2DTranspose(32, 3, activation='relu', strides=2, padding='same')(x)
        decoder_outputs = layers.Conv2DTranspose(1, 3, activation='sigmoid', padding='same')(x)
        return models.Model(latent_inputs, decoder_outputs, name='decoder')

    def call(self, inputs):
        z_mean, z_log_var = self.encoder(inputs)
        z = self.reparameterize(z_mean, z_log_var)
        reconstructed = self.decoder(z)
        kl_loss = -0.5 * tf.reduce_mean(
            z_log_var - tf.square(z_mean) - tf.exp(z_log_var) + 1)
        self.add_loss(self.beta * kl_loss)
        return reconstructed

    def reparameterize(self, mean, log_var):
        epsilon = tf.random.normal(shape=tf.shape(mean))
        return mean + tf.exp(0.5 * log_var) * epsilon

    def generate_digit(self, z):
        z = np.array(z).reshape(1, self.latent_dim)
        generated = self.decoder.predict(z)
        generated = (generated * 255).astype(np.uint8).squeeze()
        return generated