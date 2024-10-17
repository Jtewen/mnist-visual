import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np

class VAE(tf.keras.Model):
    def __init__(self, latent_dim=10, beta=1.0, **kwargs):
        super(VAE, self).__init__(**kwargs)
        self.latent_dim = latent_dim
        self.beta = beta
        self.encoder = self.build_encoder()
        self.decoder = self.build_decoder()

    def build_encoder(self):
        encoder_inputs = layers.Input(shape=(28, 28, 1))
        x = layers.Conv2D(64, 3, activation='relu', strides=2, padding='same')(encoder_inputs)
        x = layers.Conv2D(128, 3, activation='relu', strides=2, padding='same')(x)
        x = layers.Flatten()(x)
        x = layers.Dense(64, activation='relu')(x)
        z_mean = layers.Dense(self.latent_dim, name='z_mean')(x)
        z_log_var = layers.Dense(self.latent_dim, name='z_log_var')(x)
        return models.Model(encoder_inputs, [z_mean, z_log_var], name='encoder')

    def build_decoder(self):
        latent_inputs = layers.Input(shape=(self.latent_dim,))
        x = layers.Dense(7 * 7 * 64, activation='relu')(latent_inputs)
        x = layers.Reshape((7, 7, 64))(x)
        x = layers.Conv2DTranspose(128, 3, strides=2, activation='relu', padding='same')(x)
        x = layers.Conv2DTranspose(64, 3, strides=2, activation='relu', padding='same')(x)
        decoder_outputs = layers.Conv2DTranspose(1, 3, activation='sigmoid', padding='same')(x)
        return models.Model(latent_inputs, decoder_outputs, name='decoder')

    def reparameterize(self, z_mean, z_log_var):
        epsilon = tf.random.normal(shape=tf.shape(z_mean))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

    def call(self, inputs, training=False):
        z_mean, z_log_var = self.encoder(inputs)
        z = self.reparameterize(z_mean, z_log_var)
        reconstructed = self.decoder(z)

        # Compute reconstruction loss
        reconstruction_loss = tf.reduce_mean(
            tf.keras.losses.binary_crossentropy(
                tf.reshape(inputs, [tf.shape(inputs)[0], -1]),
                tf.reshape(reconstructed, [tf.shape(reconstructed)[0], -1])
            )
        ) * 28 * 28

        # Compute KL divergence loss
        kl_loss = -0.5 * tf.reduce_mean(
            tf.reduce_sum(1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var), axis=1)
        )

        # Add the weighted KL divergence loss
        self.add_loss(reconstruction_loss + self.beta * kl_loss)

        return reconstructed

    def generate_digit(self, z):
        z = tf.convert_to_tensor(z, dtype=tf.float32)
        z = tf.reshape(z, (1, self.latent_dim))
        generated = self.decoder(z)
        generated = tf.squeeze(generated, axis=0).numpy() * 255

        # If there's a channel dimension, remove it
        if generated.ndim == 3 and generated.shape[-1] == 1:
            generated = np.squeeze(generated, axis=-1)
        elif generated.ndim == 3 and generated.shape[-1] == 3:
            # If the image has 3 channels, convert it to grayscale
            generated = np.dot(generated[..., :3], [0.2989, 0.5870, 0.1140])
        return generated.astype('uint8')
