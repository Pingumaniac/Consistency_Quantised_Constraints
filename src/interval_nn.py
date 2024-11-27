import torch.nn as nn
import torch.nn.quantized.dynamic as qnn
import torch
class Interval:
    def __init__(self, lower, upper):
        self.lower = lower
        self.upper = upper

    def __add__(self, other):
        return Interval(self.lower + other.lower, self.upper + other.upper)

    def __mul__(self, other):
        return Interval(self.lower * other.lower, self.upper * other.upper)

def interval_linear(interval_input, weight_interval, bias_interval):
    lower = interval_input.lower @ weight_interval.lower.T + bias_interval.lower # should use X * transpose(X)
    upper = interval_input.upper @ weight_interval.upper.T + bias_interval.upper
    return Interval(lower, upper)
#
# class IntervalNeuralNetwork:
#     def __init__(self, model, quant_error):
#         self.model = model
#         self.quant_error = quant_error
#
#     def to_interval(self, weights, biases):
#         weight_interval = Interval(weights - self.quant_error, weights + self.quant_error)
#         bias_interval = Interval(biases - self.quant_error, biases + self.quant_error)
#         return weight_interval, bias_interval

    # def propagate(self, input_interval):
    #     current_interval = input_interval
    #     for layer in self.model.children():
    #         if isinstance(layer, nn.Linear):
    #             weight_interval, bias_interval = self.to_interval(layer.weight, layer.bias)
    #             current_interval = interval_linear(current_interval, weight_interval, bias_interval)
    #         elif isinstance(layer, nn.ReLU):
    #             current_interval = Interval(
    #                 torch.relu(current_interval.lower),
    #                 torch.relu(current_interval.upper)
    #             )
    #     return current_interval

import torch.nn.quantized.dynamic as qnn

import torch.nn.quantized.dynamic as qnn

import torch.nn.quantized.dynamic as qnn


class IntervalNeuralNetwork:
    def __init__(self, model, quant_error):
        self.model = model
        self.quant_error = quant_error

    def to_interval(self, weights, biases):
        # Create an interval for weights and biases by considering quantization error
        weight_interval = Interval(weights - self.quant_error, weights + self.quant_error)
        if biases is not None:
            bias_interval = Interval(biases - self.quant_error, biases + self.quant_error)
        else:
            # If bias is None, set bias interval to zero
            bias_interval = Interval(torch.zeros(weights.size(0)), torch.zeros(weights.size(0)))
        return weight_interval, bias_interval

    def propagate(self, input_interval):
        current_interval = input_interval

        for layer in self.model.children():
            if isinstance(layer, nn.Linear):
                # Convert the layer's weights and biases to intervals considering quantization error
                weight_interval, bias_interval = self.to_interval(layer.weight, layer.bias)
                # Propagate the interval through the linear layer
                lower_output = current_interval.lower @ weight_interval.lower.T + bias_interval.lower
                upper_output = current_interval.upper @ weight_interval.upper.T + bias_interval.upper

            elif isinstance(layer, qnn.Linear):
                # For quantized dynamic linear layers, access the weights using `.weight()`
                # and dequantize them for arithmetic operations
                weight = layer.weight().dequantize()  # Dequantize the weight to floating-point
                bias = layer.bias  # Bias might be None

                if isinstance(bias, torch.Tensor):
                    # Create intervals for weights and biases
                    weight_interval = Interval(weight - self.quant_error, weight + self.quant_error)
                    bias_interval = Interval(bias - self.quant_error, bias + self.quant_error)
                else:
                    # If bias is None, create zero biases for interval propagation
                    weight_interval = Interval(weight - self.quant_error, weight + self.quant_error)
                    bias_interval = Interval(torch.zeros(weight.size(0)), torch.zeros(weight.size(0)))

                # Propagate the interval through the quantized layer
                lower_output = current_interval.lower @ weight_interval.lower.T + bias_interval.lower
                upper_output = current_interval.upper @ weight_interval.upper.T + bias_interval.upper

            elif isinstance(layer, nn.ReLU):
                # Apply ReLU to both lower and upper bounds
                lower_output = torch.relu(current_interval.lower)
                upper_output = torch.relu(current_interval.upper)

            # Update the current interval
            current_interval = Interval(lower_output, upper_output)

        return current_interval



