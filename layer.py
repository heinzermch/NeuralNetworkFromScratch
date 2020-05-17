import numpy as np
import typing


class Layer:
    def __init__(self, **kwargs):
        self.input, self.output = None, None

    def __call__(self, input: np.ndarray) -> np.ndarray:
        raise NotImplementedError()

    def backward(self, gradient: np.ndarray) -> np.ndarray:
        raise NotImplementedError()

    def update(self, learning_rate: float) -> None:
        pass


class Linear(Layer):
    def __init__(self, input_units: int, output_units: int):
        super().__init__()
        self.weight = np.random.randn(input_units, output_units) * np.sqrt(2 / 512)
        self.bias = np.random.randn(output_units)
        self.weight_gradient = np.zeros((input_units, output_units))
        self.bias_gradient = np.zeros(output_units)

    def __call__(self, input: np.ndarray) -> np.ndarray:
        self.input, self.output = input, input.dot(self.weight) + self.bias
        return self.output

    def backward(self, gradient: np.ndarray) -> np.ndarray:
        self.weight_gradient = np.sum(
            self.input[:, :, None] * gradient[:, None, :], axis=0
        )
        self.bias_gradient = np.sum(gradient, axis=0)
        return gradient.dot(self.weight.transpose())

    def update(self, learning_rate: float) -> None:
        self.weight -= learning_rate * self.weight_gradient
        self.bias -= learning_rate * self.bias_gradient


class ReLU(Layer):
    def __call__(self, input: np.ndarray):
        self.output = np.maximum(input, 0)
        return self.output

    def backward(self, gradient: np.ndarray):
        gradient[self.output == 0] = 0
        return gradient


class Exponential(Layer):
    def __call__(self, input: np.ndarray):
        self.output = np.exp(input)
        return self.output

    def backward(self, gradient: np.ndarray):
        return np.exp(gradient)


class Softmax(Layer):
    def __call__(self, input: np.ndarray) -> np.ndarray:
        input_max = np.max(input, axis=1)[:, None]
        input_exp = np.exp(input - input_max)
        self.output = input_exp / np.sum(input_exp, axis=1)[:, None]
        return self.output

    def backward(self, gradient: np.ndarray):
        softmax = self.output
        softmax_horizontal = softmax[:, :, None]
        softmax_vertical = softmax[:, None, :]
        indicator = np.diag([1] * softmax.shape[1])[None, :, :]
        gradient = (
            gradient[:, :, None] * softmax_vertical * (indicator - softmax_horizontal)
        )
        return np.sum(gradient, axis=1)


class Reparameterization:
    def __call__(self, mean: np.ndarray, variance: np.ndarray):
        standard_normal_variables = np.random.standard_normal(
            np.prod(mean.shape)
        ).reshape(mean.shape)
        self.mean, self.variance = mean, variance
        self.output = mean + variance * standard_normal_variables
        return self.output

    def backward(self, gradient: np.ndarray) -> typing.Tuple[np.ndarray, np.ndarray]:
        return gradient, gradient
