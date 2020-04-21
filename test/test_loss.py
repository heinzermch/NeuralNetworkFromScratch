from loss import MSE, CrossEntropy, MSES
import torch
from torch import nn
from tool import assert_tensor_near_zero, ClassificationLossWrapper, RegressionLossWrapper, to_numpy


class TorchMSES:
    def __init__(self):
        self.softmax = nn.Softmax(dim=1)
        self.mse = nn.MSELoss()

    def __call__(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.mse(self.softmax(input), target)


def test_mses() -> None:
    batch_size, features = 3, 5
    torch_mse = TorchMSES()
    mses = RegressionLossWrapper(MSES)
    torch.manual_seed(0)
    input = torch.randn(batch_size, features, requires_grad=True)
    target = torch.ones(batch_size, features)

    torch_loss = torch_mse(input, target)
    loss = mses(input, target)
    # Check forward loss
    assert_tensor_near_zero(to_numpy(torch_loss) - loss)
    # Check gradient
    torch_loss.backward()
    torch_gradient = input.grad
    gradient = mses.gradient()
    assert_tensor_near_zero(to_numpy(torch_gradient) - gradient)


def test_mse() -> None:
    batch_size, features = 3, 5
    torch_mse = nn.MSELoss()
    mse = RegressionLossWrapper(MSE)

    input = torch.randn(batch_size, features, requires_grad=True)
    target = torch.randn(batch_size, features)

    torch_loss = torch_mse(input, target)
    loss = mse(input, target)
    # Check forward loss
    assert_tensor_near_zero(to_numpy(torch_loss) - loss)
    # Check gradient
    torch_loss.backward()
    torch_gradient = input.grad
    gradient = mse.gradient()
    assert_tensor_near_zero(to_numpy(torch_gradient) - gradient)


def test_cross_entropy() -> None:
    batch_size, features = 3, 5
    torch_cross_entropy = nn.CrossEntropyLoss()
    cross_entropy = ClassificationLossWrapper(CrossEntropy)

    input = torch.randn(batch_size, features, requires_grad=True)
    target = torch.empty(batch_size, dtype=torch.long).random_(features)
    # Forward loss
    loss = cross_entropy(input, target, classes=features)
    torch_loss = torch_cross_entropy(input, target)
    assert_tensor_near_zero(to_numpy(torch_loss) - loss)
    # Check gradient
    torch_loss.backward()
    torch_gradient = input.grad
    gradient = cross_entropy.gradient()
    assert_tensor_near_zero(to_numpy(torch_gradient) - gradient)
