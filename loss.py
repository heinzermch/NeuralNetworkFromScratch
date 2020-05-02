import numpy as np
from layer import Softmax


class Loss:
    def gradient(self) -> np.ndarray:
        pass

    def __call__(self, input: np.ndarray, target: np.ndarray) -> np.ndarray:
        pass


class MSE(Loss):
    def __call__(self, input: np.ndarray, target: np.ndarray) -> np.ndarray:
        self.input, self.target = input, target
        return np.mean(np.power(input - target, 2))

    def gradient(self) -> np.ndarray:
        return 2.0 * (self.input - self.target) / np.prod(self.input.shape)


class MSES(Loss):
    def __init__(self):
        self.softmax = Softmax()

    def __call__(self, input: np.ndarray, target: np.ndarray) -> np.ndarray:
        self.input, self.target = input, target
        return np.mean(np.power(self.softmax(input) - target, 2))

    def gradient(self) -> np.ndarray:
        batch_size, classes = self.input.shape
        softmax = self.softmax(self.input)
        left_terms = softmax[:, None, :] - self.target[:, None, :]
        middle_terms = softmax[:, None, :]
        right_terms = np.diag([1] * classes)[None, :, :] - softmax[:, :, None]
        gradients = (
            2.0 / (batch_size * classes) * left_terms * middle_terms * right_terms
        )
        return np.sum(gradients, axis=2)


class CrossEntropy(Loss):
    def __init__(self):
        self.softmax = Softmax()

    def __call__(self, input: np.ndarray, target: np.ndarray) -> np.ndarray:
        self.input, self.target = input, target
        input_probabilities = self.softmax(input)
        return np.mean(-1 * np.log(input_probabilities)[target > 0])

    def gradient(self) -> np.ndarray:
        gradient = self.softmax(self.input)
        gradient[self.target.astype(np.bool)] -= 1
        return gradient / gradient.shape[0]
