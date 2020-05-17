import argparse
import numpy as np
from data import DataLoader
from loss import CrossEntropy, MSE, KLDivergenceStandardNormal
from model import Model
from layer import Linear, ReLU, Reparameterization, Exponential


def train_variational_autoencoder(
    learning_rate: float,
    epochs: int,
    batch_size: int,
    latent_variables: int = 10,
    print_every: int = 50,
) -> None:
    print(
        f"Training a variational autoencoder for {epochs} epochs with batch size {batch_size}"
    )
    data_loader = DataLoader(batch_size)
    image_loss = CrossEntropy()
    divergence_loss = KLDivergenceStandardNormal()
    encoder_mean = Model([Linear(784, 50), ReLU(), Linear(50, latent_variables)])
    encoder_variance = Model(
        [Linear(784, 50), ReLU(), Linear(50, latent_variables), Exponential()]
    )
    reparameterization = Reparameterization()
    decoder = Model([Linear(latent_variables, 50), ReLU(), Linear(50, 784)])

    for i in range(epochs):
        # One training loop
        training_data = data_loader.get_training_data()

        for j, batch in enumerate(training_data):
            input, target = batch
            # Forward pass
            mean = encoder_mean(input)
            variance = encoder_variance(input)
            z = reparameterization(mean=mean, variance=variance)
            generated_samples = decoder(z)
            # Loss calculation
            divergence_loss_value = divergence_loss(mean, variance)
            generation_loss = image_loss(generated_samples, input)
            if j % print_every == 0:
                print(
                    f"Epoch {i+1}/{epochs}, "
                    f"training iteration {j+1}/{len(training_data)}"
                )
                print(
                    f"KL loss {np.round(divergence_loss_value, 2)}\t"
                    f"Generation loss {np.round(generation_loss, 2)}"
                )

            # Backward pass
            decoder_gradient = image_loss.gradient()
            decoder_gradient = decoder.backward(decoder_gradient)
            decoder_mean_gradient, decoder_variance_gradient = reparameterization.backward(
                decoder_gradient
            )
            encoder_mean_gradient, encoder_variance_gradient = (
                divergence_loss.gradient()
            )
            encoder_mean.backward(decoder_mean_gradient + encoder_mean_gradient)
            encoder_variance.backward(
                decoder_variance_gradient + encoder_variance_gradient
            )

            # Update model weights
            # encoder_mean.update(learning_rate)
            # encoder_variance.update(learning_rate)
            # decoder.update(learning_rate)


def _validate_args(args: argparse.Namespace) -> None:
    assert 0 < args.learning_rate, args.learning_rate
    assert 0 < args.number_of_epochs, args.number_of_epochs
    assert 0 < args.batch_size, args.batch_size
    assert 10000 % args.batch_size == 0, 10000 % args.batch_size == 0


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-lr",
        "--learning_rate",
        default=0.1,
        type=float,
        help="How fast the model will be updated",
    )
    parser.add_argument(
        "-e",
        "--number_of_epochs",
        default=10,
        type=int,
        help="How many times shall will the network see the training data",
    )
    parser.add_argument(
        "-bs",
        "--batch_size",
        default=5,
        type=int,
        help="On how many examples should be trained on per pass",
    )
    return parser.parse_args()


if __name__ == "__main__":
    """ Run the neural network implemented in numpy on MNIST """
    options = _parse_args()
    _validate_args(options)

    train_variational_autoencoder(
        learning_rate=options.learning_rate,
        epochs=options.number_of_epochs,
        batch_size=options.batch_size,
    )
