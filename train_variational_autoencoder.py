import argparse


def train_variational_autoencoder(
    learning_rate: float, epochs: int, batch_size: int, hidden_units: int = 100
) -> None:
    print("no failure")


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
        default=100,
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
