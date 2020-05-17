import numpy as np


class Model:
    def __init__(self, layers: list):
        self.layers = layers

    def __call__(self, x: np.ndarray) -> np.ndarray:
        for layer in self.layers:
            x = layer(x)
        return x

    def backward(self, gradient: np.ndarray) -> np.ndarray:
        for layer in reversed(self.layers):
            gradient = layer.backward(gradient)
        return gradient

    def update(self, learning_rate: float):
        for layer in self.layers:
            layer.update(learning_rate)
