import numpy as np
from layer import Softmax
import typing


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


class KLDivergenceStandardNormal(Loss):
    def __call__(self, means: np.ndarray, variances: np.ndarray) -> np.ndarray:
        n, k = means.shape
        self.means, self.variances = means, variances
        trace = np.sum(variances, axis=1)
        prod = np.sum(means ** 2, axis=1)
        det = np.log(np.sum(variances, axis=1))
        log_det = np.log(det)
        loss_per_sample = 1 / 2 * (trace + prod - k - log_det)
        return np.mean(loss_per_sample)

    def gradient(self) -> typing.Tuple[np.ndarray, np.ndarray]:
        return self.means, self.variances - 1
