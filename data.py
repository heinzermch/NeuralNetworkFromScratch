from fastai import datasets
import gzip
import pickle
import numpy as np
import typing
from tool import index_to_one_hot, normalize_row_wise

MNIST_URL = "http://deeplearning.net/data/mnist/mnist.pkl"


def load_data() -> typing.Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    path = datasets.download_data(MNIST_URL, ext=".gz")
    with gzip.open(path, "rb") as f:
        ((x_train, y_train), (x_valid, y_valid), _) = pickle.load(f, encoding="latin-1")
    y_train, y_valid = index_to_one_hot(y_train), index_to_one_hot(y_valid)
    x_train, x_valid = normalize_row_wise(x_train), normalize_row_wise(x_valid)
    return x_train, y_train, x_valid, y_valid


class DataSet:
    def __init__(self, x: np.ndarray, y: np.ndarray, batch_size: int):
        self.x, self.y = x, y
        self.batch_size = batch_size

    def __iter__(self) -> typing.Tuple[np.ndarray, np.ndarray]:
        for i in range(0, len(self.x), self.batch_size):
            yield self.x[i : i + self.batch_size], self.y[i : i + self.batch_size]

    def __len__(self) -> int:
        return len(self.x) // self.batch_size


class DataLoader:
    def __init__(self, batch_size: int):
        x_train, y_train, x_valid, y_valid = load_data()
        self.x_train, self.y_train = x_train, y_train
        self.x_valid, self.y_valid = x_valid, y_valid
        self.batch_size = batch_size

    def get_training_data(self) -> DataSet:
        return DataSet(x=self.x_train, y=self.y_train, batch_size=self.batch_size)

    def get_validation_data(self) -> DataSet:
        return DataSet(x=self.x_valid, y=self.y_valid, batch_size=self.batch_size)
