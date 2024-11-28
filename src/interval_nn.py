# interval_nn.py

import torch.nn as nn
import torch.nn.quantized.dynamic as qnn
import torch

class Interval:
    """
    Represents an interval [lower, upper].
    """
    def __init__(self, lower, upper):
        self.lower = lower
        self.upper = upper

    def __add__(self, other):
        """
        Add two intervals element-wise.
        """
        return Interval(self.lower + other.lower, self.upper + other.upper)

    def __mul__(self, other):
        """
        Multiply two intervals element-wise.
        """
        return Interval(self.lower * other.lower, self.upper * other.upper)


def interval_linear(interval_input, weight_interval, bias_interval):
    """
    Propagate intervals through a linear layer.
    """
    lower = interval_input.lower @ weight_interval.lower.T + bias_interval.lower
    upper = interval_input.upper @ weight_interval.upper.T + bias_interval.upper
    return Interval(lower, upper)


class IntervalNeuralNetwork:
    """
    Interval neural network for verification, considering quantization errors.
    """
    def __init__(self, model, quant_error):
        self.model = model
        self.quant_error = quant_error

    def to_interval(self, weights, biases):
        """
        Convert weights and biases to intervals by considering quantization error.
        """
        # Ensure weights are tensors
        if callable(weights):
            weights = weights().dequantize()  # Dequantize weights if callable (quantized layers)

        weight_interval = Interval(weights - self.quant_error, weights + self.quant_error)

        if biases is not None:
            # Ensure biases are tensors
            if callable(biases):
                biases = biases()  # Call biases if it's a callable method
            bias_interval = Interval(biases - self.quant_error, biases + self.quant_error)
        else:
            # If bias is None, set bias interval to zero
            bias_interval = Interval(
                torch.zeros(weights.size(0), device=weights.device),
                torch.zeros(weights.size(0), device=weights.device)
            )
        return weight_interval, bias_interval

    def propagate(self, input_interval):
        """
        Propagate intervals through the network layer by layer.
        """
        current_interval = input_interval

        for layer in self.model.children():
            if isinstance(layer, nn.Linear):
                # Handle standard linear layers
                weight_interval, bias_interval = self.to_interval(layer.weight, layer.bias)
                current_interval = interval_linear(current_interval, weight_interval, bias_interval)

            elif isinstance(layer, qnn.Linear):
                # Handle quantized dynamic linear layers
                weight = layer.weight  # Access weights directly
                bias = layer.bias  # Bias might be None
                weight_interval, bias_interval = self.to_interval(weight, bias)
                current_interval = interval_linear(current_interval, weight_interval, bias_interval)

            elif isinstance(layer, torch.ao.nn.quantized.modules.linear.Linear):
                # Handle statically quantized linear layers (QAT converted)
                weight = layer.weight().dequantize()  # Dequantize weights to floating-point
                bias = layer.bias  # Bias might be None
                weight_interval, bias_interval = self.to_interval(weight, bias)
                current_interval = interval_linear(current_interval, weight_interval, bias_interval)


            elif isinstance(layer, nn.ReLU):
                # Handle ReLU activation
                current_interval = Interval(
                    torch.relu(current_interval.lower),
                    torch.relu(current_interval.upper)
                )

            else:
                raise ValueError(f"Unsupported layer type: {type(layer)}")

        return current_interval
