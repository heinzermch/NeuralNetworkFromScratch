from fastai import datasets
import gzip
import pickle
import numpy as np
import typing
from tool import index_to_one_hot

MNIST_URL = "http://deeplearning.net/data/mnist/mnist.pkl"


def load_data() -> typing.Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    path = datasets.download_data(MNIST_URL, ext=".gz")
    with gzip.open(path, "rb") as f:
        ((x_train, y_train), (x_valid, y_valid), _) = pickle.load(f, encoding="latin-1")
    y_train, y_valid = index_to_one_hot(y_train), index_to_one_hot(y_valid)
    return x_train, y_train, x_valid, y_valid
