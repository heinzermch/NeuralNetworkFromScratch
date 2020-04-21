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
        return np.mean(np.power(input-target, 2))

    def gradient(self) -> np.ndarray:
        return 2.0 * (self.input - self.target) / np.multiply(*self.input.shape)


class MSES(Loss):
    def __init__(self):
        self.softmax = Softmax()

    def __call__(self, input: np.ndarray, target: np.ndarray) -> np.ndarray:
        self.input, self.target = input, target
        return np.mean(np.power(self.softmax(input)-target, 2))

    def gradient(self) -> np.ndarray:
        softmax = self.softmax(self.input)
        mse_gradient = 2.0 * (softmax - self.target) / np.multiply(*self.input.shape)
        softmax_horizontal = softmax[:, :, None]
        softmax_vertical = softmax[:, None, :]
        indicator = np.diag([1] * softmax.shape[1])[None, :, :]
        gradient = mse_gradient[:, :, None] * softmax_vertical * (indicator - softmax_horizontal)
        return np.sum(gradient, axis=1)


class CrossEntropy(Loss):
    def __init__(self):
        self.softmax = Softmax()

    def __call__(self, input: np.ndarray, target: np.ndarray) -> np.ndarray:
        self.input, self.target = input, target
        input_probabilities = self.softmax(input)
        return np.mean(-1*np.log(input_probabilities)[target > 0])

    def gradient(self) -> np.ndarray:
        gradient = self.softmax(self.input)
        gradient[self.target.astype(np.bool)] -= 1
        return gradient / gradient.shape[0]


