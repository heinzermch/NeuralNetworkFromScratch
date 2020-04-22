import torch
from torch import nn
from layer import ReLU, Linear, Softmax
from loss import MSE
from tool import assert_tensor_near_zero, to_numpy, LayerWrapper, RegressionLossWrapper


def test_relu():
    batch_size = 3
    features = 5
    # Forward pass, reference layer and layer
    torch_relu = nn.ReLU()
    relu = LayerWrapper(ReLU)
    # Forward pass
    input = torch.randn(batch_size, features, requires_grad=True)
    assert_tensor_near_zero(to_numpy(torch_relu(input)) - relu(input))

    # Backward pass, losses
    torch_loss = nn.MSELoss()
    loss = RegressionLossWrapper(MSE)
    # Backward pass, loss calculation and check
    target = torch.randn(batch_size, features)
    torch_loss_output = torch_loss(torch_relu(input), target)
    torch_loss_output.backward()
    loss_output = loss(torch.from_numpy(relu(input)), target)
    assert_tensor_near_zero(loss_output - to_numpy(torch_loss_output))
    # Backward pass, gradient calculation
    gradient = relu.backward(torch.from_numpy(loss.gradient()))
    torch_gradient = input.grad
    assert_tensor_near_zero(to_numpy(torch_gradient) - gradient)


def test_linear():
    batch_size = 3
    in_features = 768
    out_features = 50
    torch_linear = nn.Linear(in_features=in_features, out_features=out_features)
    linear = LayerWrapper(Linear, input_units=in_features, output_units=out_features)
    input = torch.randn(batch_size, in_features, requires_grad=True)
    # Forward pass, to ensure same operation we copy weight and bias
    linear.update_attribute("weight", to_numpy(torch_linear.weight.T))
    linear.update_attribute("bias", to_numpy(torch_linear.bias))
    assert_tensor_near_zero(to_numpy(torch_linear(input)) - linear(input))

    # Backward pass, losses
    torch_loss = nn.MSELoss()
    loss = RegressionLossWrapper(MSE)
    # Backward pass, loss calculation and check
    target = torch.randn(batch_size, out_features)
    torch_loss_output = torch_loss(torch_linear(input), target)
    torch_loss_output.backward()
    loss_output = loss(torch.from_numpy(linear(input)), target)
    assert_tensor_near_zero(loss_output - to_numpy(torch_loss_output))
    # Backward pass, gradient calculation
    gradient = linear.backward(torch.from_numpy(loss.gradient()))
    torch_gradient = input.grad
    assert_tensor_near_zero(to_numpy(torch_gradient) - gradient)
    # Backward pass, weight gradient
    torch_weight_gradient = torch_linear.weight.grad.T
    weight_gradient = linear.get_attribute("weight_gradient")
    assert_tensor_near_zero(to_numpy(torch_weight_gradient) - weight_gradient)
    # Backward pass, bias gradient
    torch_bias_gradient = torch_linear.bias.grad
    bias_gradient = linear.get_attribute("bias_gradient")
    assert_tensor_near_zero(to_numpy(torch_bias_gradient) - bias_gradient)
    # Optimization, gradient update


def test_softmax():
    batch_size = 3
    features = 5
    torch_softmax = nn.Softmax(dim=1)
    softmax = LayerWrapper(Softmax)
    # Forward pass
    input = torch.randn(batch_size, features, requires_grad=True)
    assert_tensor_near_zero(to_numpy(torch_softmax(input)) - softmax(input))

    # Backward pass, losses
    torch_loss = nn.MSELoss()
    loss = RegressionLossWrapper(MSE)
    # Backward pass, loss calculation and check
    target = torch.randn(batch_size, features)
    torch_loss_output = torch_loss(torch_softmax(input), target)
    torch_loss_output.backward()
    loss_output = loss(torch.from_numpy(softmax(input)), target)
    assert_tensor_near_zero(loss_output - to_numpy(torch_loss_output))
    # Backward pass, gradient calculation
    gradient = softmax.backward(torch.from_numpy(loss.gradient()))
    torch_gradient = input.grad
    assert_tensor_near_zero(to_numpy(torch_gradient) - gradient)
