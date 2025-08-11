import tensorflow as tf
from tensorflow.keras import layers

class AutoencoderConv(tf.keras.Model):
    def __init__(self):
        super(AutoencoderConv, self).__init__()

        # Encoder
        self.encoder = tf.keras.Sequential([
            layers.Input(shape=(28, 28, 1)),
            layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
            layers.MaxPooling2D((2, 2), padding='same'),  # 14x14x32
            layers.Conv2D(16, (3, 3), activation='relu', padding='same'),
            layers.MaxPooling2D((2, 2), padding='same'),  # 7x7x16
        ])

        # Decoder
        self.decoder = tf.keras.Sequential([
            layers.Conv2DTranspose(16, kernel_size=3, strides=2, activation='relu', padding='same'),  # 14x14x16
            layers.Conv2DTranspose(32, kernel_size=3, strides=2, activation='relu', padding='same'),  # 28x28x32
            layers.Conv2D(1, kernel_size=3, activation='sigmoid', padding='same')  # 28x28x1
        ])

    def call(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded