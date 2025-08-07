import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

class VAE(keras.Model):
    def __init__(self, latent_dim=64, **kwargs):
        super(VAE, self).__init__(**kwargs)
        self.latent_dim = latent_dim
        self.encoder = self.build_encoder()
        self.decoder = self.build_decoder()
        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = keras.metrics.Mean(name="reconstruction_loss")
        self.kl_loss_tracker = keras.metrics.Mean(name="kl_loss")
    def build_encoder(self):
        encoder_inputs = keras.Input(shape=(64, 64, 3), name='input_image')
        x = layers.Conv2D(32, 4, strides=2, padding='same', activation='relu')(encoder_inputs)
        x = layers.Conv2D(64, 4, strides=2, padding='same', activation='relu')(x)
        x = layers.Conv2D(128, 4, strides=2, padding='same', activation='relu')(x)
        x = layers.Flatten()(x)
        x = layers.Dense(256, activation='relu')(x)
        z_mean = layers.Dense(self.latent_dim, name='z_mean')(x)
        z_log_var = layers.Dense(self.latent_dim, name='z_log_var')(x)
        return keras.Model(encoder_inputs, [z_mean, z_log_var], name='encoder')
    def build_decoder(self):
        latent_inputs = keras.Input(shape=(self.latent_dim,), name='latent_vector')
        x = layers.Dense(8 * 8 * 128, activation='relu')(latent_inputs)
        x = layers.Reshape((8, 8, 128))(x)
        x = layers.Conv2DTranspose(128, 4, strides=2, padding='same', activation='relu')(x)
        x = layers.Conv2DTranspose(64, 4, strides=2, padding='same', activation='relu')(x)
        x = layers.Conv2DTranspose(32, 4, strides=2, padding='same', activation='relu')(x)
        decoder_outputs = layers.Conv2DTranspose(3, 3, padding='same', activation='sigmoid')(x)
        return keras.Model(latent_inputs, decoder_outputs, name='decoder')
    def sampling(self, z_mean, z_log_var):
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon
    def call(self, inputs):
        z_mean, z_log_var = self.encoder(inputs)
        z = self.sampling(z_mean, z_log_var)
        reconstructed = self.decoder(z)
        return reconstructed
    def train_step(self, data):
        with tf.GradientTape() as tape:
            z_mean, z_log_var = self.encoder(data)
            z = self.sampling(z_mean, z_log_var)
            reconstruction = self.decoder(z)
            reconstruction_loss = tf.reduce_mean(
                tf.reduce_sum(
                    keras.losses.binary_crossentropy(data, reconstruction), axis=(1, 2)
                )
            )
            kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
            kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))
            total_loss = reconstruction_loss + kl_loss
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
        }
    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker,
        ]