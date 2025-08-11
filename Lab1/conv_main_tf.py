import argparse
import random
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from tensorflow.keras.datasets import mnist
from model_tf import AutoencoderMLP4Layer, AutoencoderConv

def add_noise(image):
    new_noise = tf.random.uniform(tf.shape(image))
    noisy = new_noise + image
    noisy = tf.clip_by_value(noisy, 0, 1)
    return noisy

def visualize(index, x_train, model, model_type):
    img = x_train[index]

    if model_type == "conv":
        img_input = tf.expand_dims(img, axis=0)  # Shape: (1, 28, 28, 1)
        reconstructed_img = model(img_input, training=False)
        noisy_img = add_noise(img)

        f = plt.figure()
        f.add_subplot(1, 3, 1)
        plt.imshow(tf.squeeze(img), cmap='gray')
        f.add_subplot(1, 3, 2)
        plt.imshow(tf.squeeze(noisy_img), cmap='gray')
        f.add_subplot(1, 3, 3)
        plt.imshow(tf.squeeze(reconstructed_img), cmap='gray')
        plt.show()

    else:  # mlp
        img = tf.cast(img, tf.float32)
        img_flat = tf.reshape(img, [1, 28 * 28])
        reconstructed_img = model(img_flat, training=False)

        noisy_img = add_noise(img)
        noisy_img = tf.reshape(noisy_img, [28, 28])
        img = tf.reshape(img, [28, 28])
        reconstructed_img = tf.reshape(reconstructed_img, [28, 28])

        f = plt.figure()
        f.add_subplot(1, 3, 1)
        plt.imshow(img.numpy(), cmap='gray')
        f.add_subplot(1, 3, 2)
        plt.imshow(noisy_img.numpy(), cmap='gray')
        f.add_subplot(1, 3, 3)
        plt.imshow(reconstructed_img.numpy(), cmap='gray')
        plt.show()

def interpolate(img1, img2, n, model, model_type):
    f = plt.figure()
    f.add_subplot(1, n + 2, 1)
    plt.imshow(tf.squeeze(img1), cmap='gray')
    f.add_subplot(1, n + 2, n + 2)
    plt.imshow(tf.squeeze(img2), cmap='gray')

    if model_type == "conv":
        img1 = tf.expand_dims(img1, axis=0)
        img2 = tf.expand_dims(img2, axis=0)

        bottleneck1 = model.encoder(img1)
        bottleneck2 = model.encoder(img2)

        for i in range(1, n + 1):
            alpha = i / (n + 1)
            interpolated = (1 - alpha) * bottleneck1 + alpha * bottleneck2
            decoded = model.decoder(interpolated)
            f.add_subplot(1, n + 2, i + 1)
            plt.imshow(tf.squeeze(decoded), cmap='gray')

    else:  # mlp
        img1 = tf.cast(img1, tf.float32)
        img2 = tf.cast(img2, tf.float32)

        img1_flat = tf.reshape(img1, [1, 28 * 28])
        img2_flat = tf.reshape(img2, [1, 28 * 28])

        bottleneck1 = model.encode(img1_flat)
        bottleneck2 = model.encode(img2_flat)

        weight = 0.5
        interpolated = (1 - weight) * bottleneck1 + weight * bottleneck2
        decoded = model.decode(interpolated)
        decoded = tf.reshape(decoded, [28, 28])
        f.add_subplot(1, n + 2, 2)
        plt.imshow(decoded.numpy(), cmap='gray')

        for x in range(n - 1):
            interpolated = (1 - weight) * interpolated + weight * bottleneck2
            decoded = model.decode(interpolated)
            decoded = tf.reshape(decoded, [28, 28])
            f.add_subplot(1, n + 2, x + 3)
            plt.imshow(decoded.numpy(), cmap='gray')

    plt.show()

def return_loader_set(model_type):
    (x_train, _), _ = mnist.load_data()
    x_train = x_train.astype("float32") / 255.0

    if model_type == "conv":
        x_train = x_train[..., tf.newaxis]  # (N, 28, 28, 1)
    else:
        x_train = x_train.reshape((-1, 28 * 28))  # (N, 784)

    return x_train

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-l", "--l", help="Path to model weights (.h5 file)")
    parser.add_argument("-m", "--model", choices=["mlp", "conv"], default="mlp", help="Model type")
    parser.add_argument("-z", "--z", type=int, default=8, help="MLP bottleneck size (ignored for conv)")
    args = parser.parse_args()

    x_train = return_loader_set(args.model)

    if args.model == "conv":
        model = AutoencoderConv()
        model.build((None, 28, 28, 1))
    else:
        model = AutoencoderMLP4Layer(784, args.z, 784)
        model.build((None, 784))

    model.load_weights(args.l)

    idx = int(input("Enter an integer 0 to 59999: "))
    visualize(idx, x_train, model, args.model)

    i1 = random.randint(0, len(x_train) - 1)
    i2 = random.randint(0, len(x_train) - 1)
    interpolate(x_train[i1], x_train[i2], 8, model, args.model)
