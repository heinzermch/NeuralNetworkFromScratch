import numpy as np
import torch
from loss import Loss
from layer import Layer


def assert_tensor_near_zero(tensor: np.ndarray, tol=1e-3) -> None:
    assert np.all(np.abs(tensor) < tol), f"Not zero: {tensor}"


def to_numpy(tensor: torch.Tensor) -> np.ndarray:
    return tensor.detach().numpy()


def index_to_one_hot(index: np.ndarray, classes: int = 0) -> np.ndarray:
    if not classes:
        max_elem, min_elem = np.max(index), np.min(index)
        classes = max_elem + 1 - min_elem
    n = index.size
    one_hot = np.zeros((n, classes))
    one_hot[np.arange(n), index] = 1
    return one_hot


class LossWrapper(Loss):
    def __init__(self, loss, **kwargs):
        self.loss = loss(**kwargs)

    def gradient(self) -> np.ndarray:
        return self.loss.gradient()

    def __call__(self, input: torch.Tensor, target: torch.Tensor, classes: int = 0) -> np.ndarray:
        pass


class RegressionLossWrapper(LossWrapper):
    def __call__(self, input: torch.Tensor, target: torch.Tensor, classes: int = 0) -> np.ndarray:
        return self.loss(to_numpy(input), to_numpy(target))


class ClassificationLossWrapper(LossWrapper):
    def __call__(self, input: torch.Tensor, target: torch.Tensor, classes: int = 0) -> np.ndarray:
        return self.loss(to_numpy(input), index_to_one_hot(to_numpy(target), classes))


class LayerWrapper(Layer):
    def __init__(self, layer, **kwargs):
        super().__init__()
        self.layer = layer(**kwargs)

    def __call__(self, input: torch.Tensor) -> np.ndarray:
        return self.layer(to_numpy(input))

    def backward(self, gradient: torch.Tensor) -> np.ndarray:
        return self.layer.backward(to_numpy(gradient))

    def update(self, learning_rate: float) -> None:
        self.layer.update(learning_rate)

    def update_attribute(self, name: str, value: np.ndarray) -> None:
        setattr(self.layer, name, value)

    def get_attribute(self, name: str) -> np.ndarray:
        return getattr(self.layer, name)
