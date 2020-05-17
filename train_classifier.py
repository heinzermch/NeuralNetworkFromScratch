import numpy as np
import argparse
from data import DataLoader
from loss import CrossEntropy, MSE, MSES
from model import Model
from layer import Linear, ReLU
from tool import calculate_accuracy


def train_classifier(
    learning_rate: float, epochs: int, batch_size: int, print_every: int = 50
) -> None:
    data_loader = DataLoader(batch_size)

    loss = CrossEntropy()
    model = Model([Linear(784, 50), ReLU(), Linear(50, 10)])

    for i in range(epochs):
        # One training loop
        training_data = data_loader.get_training_data()
        validation_data = data_loader.get_validation_data()
        for j, batch in enumerate(training_data):
            input, target = batch
            y = model(input)
            loss(y, target)
            gradient = loss.gradient()
            model.backward(gradient)
            model.update(learning_rate)
            if j % print_every == 0:
                print(
                    f"Epoch {i+1}/{epochs}, training iteration {j+1}/{len(training_data)}"
                )

        accuracy_values = []
        loss_values = []
        # One validation loop
        for j, batch in enumerate(validation_data):
            input, target = batch
            y = model(input)
            loss_value = loss(y, target)
            accuracy = calculate_accuracy(y, target)
            accuracy_values.append(accuracy)
            loss_values.append(loss_value)

        print(
            f"Epoch {i+1}: loss {np.round(np.average(loss_values), 2)}, accuracy {np.round(np.average(accuracy_values), 2)}"
        )


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

    train_classifier(
        learning_rate=options.learning_rate,
        epochs=options.number_of_epochs,
        batch_size=options.batch_size,
    )
