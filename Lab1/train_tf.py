import argparse
import datetime
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from model_tf import AutoencoderMLP4Layer, AutoencoderConv

def return_loader_set(batch_size, model_type):
    (x_train, _), _ = mnist.load_data()
    x_train = x_train.astype("float32") / 255.0

    if model_type == "conv":
        x_train = x_train[..., tf.newaxis]  # Shape: (N, 28, 28, 1)
    else:
        x_train = x_train.reshape((-1, 784))  # Flattened

    dataset = tf.data.Dataset.from_tensor_slices(x_train)
    dataset = dataset.shuffle(buffer_size=1024).batch(batch_size)
    return dataset, x_train

def train(n_epochs, model, loss_fn, train_dataset, optimizer, scheduler_fn,
          plot_file=None, save_file=None):

    print("Training...")
    losses_train = []

    for epoch in range(1, n_epochs + 1):
        print(f"Epoch {epoch}")
        epoch_loss = []

        for batch in train_dataset:
            with tf.GradientTape() as tape:
                recon = model(batch, training=True)
                loss = loss_fn(batch, recon)
            grads = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))
            epoch_loss.append(loss.numpy())

        epoch_loss_avg = np.mean(epoch_loss)
        losses_train.append(epoch_loss_avg)
        print(f"{datetime.datetime.now()} Epoch {epoch}, Training loss {epoch_loss_avg}")

        # Update learning rate using scheduler function
        new_lr = scheduler_fn(epoch)
        optimizer.lr.assign(new_lr)

        # Plot loss curve
        if plot_file is not None:
            plt.figure(1, figsize=(12, 7))
            plt.clf()
            plt.plot(losses_train, label="train")
            plt.xlabel("epoch")
            plt.ylabel("loss")
            plt.legend(loc=1)
            plt.savefig(plot_file)

    plt.show()

    if save_file is not None:
        model.save_weights(save_file)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-z", "--z", type=int, help="Bottleneck size (only used for MLP)")
    parser.add_argument("-e", "--e", type=int, help="Number of epochs")
    parser.add_argument("-b", "--b", type=int, help="Batch size")
    parser.add_argument("-s", "--s", help="Save file path")
    parser.add_argument("-p", "--p", help="Plot file path")
    parser.add_argument("-m", "--model", choices=["mlp", "conv"], default="mlp",
                        help="Model type: 'mlp' or 'conv'")
    args = parser.parse_args()

    # Load dataset according to model type
    train_loader, _ = return_loader_set(args.b, args.model)

    # Initialize model
    if args.model == "conv":
        model = AutoencoderConv()
        model.build(input_shape=(None, 28, 28, 1))
    else:
        model = AutoencoderMLP4Layer(784, args.z, 784)
        model.build(input_shape=(None, 784))

    # Optimizer and loss
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    loss_fn = tf.keras.losses.MeanSquaredError()

    # Exponential LR scheduler
    def scheduler_fn(epoch):
        return 0.001 * (0.5 ** epoch)

    # Train
    train(args.e, model, loss_fn, train_loader, optimizer, scheduler_fn,
          plot_file=args.p, save_file=args.s)

    # Summary
    model.summary()
