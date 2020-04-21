import numpy as np
import argparse
from data import load_data
from loss import CrossEntropy, MSE, MSES
from model import Model
from layer import Linear, ReLU


def calculate_accuracy(output: np.ndarray, target: np.ndarray) -> np.ndarray:
    return np.mean(np.argmax(output, axis=1) == np.argmax(target, axis=1))


def train_network(learning_rate: float, epochs: int, batch_size: int):
    x_train, y_train, x_valid, y_valid = load_data()

    input, target = x_train[:100], y_train[:100]

    loss = CrossEntropy()
    model = Model([Linear(784, 50), ReLU(), Linear(50, 10)])

    for i in range(epochs):
        y = model(input)
        loss_value = loss(y, target)
        accuracy = calculate_accuracy(y, target)
        gradient = loss.gradient()
        model.backward(gradient)
        model.update(learning_rate)

        print(f"Epoch {i+1}: loss {np.round(loss_value, 2)}, accuracy {np.round(accuracy, 2)}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-lr",
        "--learning_rate",
        default=0.5,
        type=float,
        help="How fast the model will be updated",
    )
    parser.add_argument(
        "-e",
        "--number_of_epochs",
        default=30,
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

    args = parser.parse_args()

    assert 0 < args.learning_rate, args.learning_rate
    assert 0 < args.number_of_epochs, args.number_of_epochs
    assert 0 < args.batch_size, args.batch_size
    assert 10000 % args.batch_size == 0, 10000 % args.batch_size == 0

    return parser.parse_args()


if __name__ == "__main__":
    """ Run the neural network implemented in numpy on MNIST """
    options = parse_args()

    train_network(
        learning_rate=options.learning_rate,
        epochs=options.number_of_epochs,
        batch_size=options.batch_size
    )
