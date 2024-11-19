class Interval:
    def __init__(self, lower, upper):
        self.lower = lower
        self.upper = upper

    def __add__(self, other):
        return Interval(self.lower + other.lower, self.upper + other.upper)

    def __mul__(self, other):
        return Interval(self.lower * other.lower, self.upper * other.upper)

def interval_linear(interval_input, weight_interval, bias_interval):
    lower = interval_input.lower @ weight_interval.lower + bias_interval.lower
    upper = interval_input.upper @ weight_interval.upper + bias_interval.upper
    return Interval(lower, upper)

class IntervalNeuralNetwork:
    def __init__(self, model, quant_error):
        self.model = model
        self.quant_error = quant_error

    def to_interval(self, weights, biases):
        weight_interval = Interval(weights - self.quant_error, weights + self.quant_error)
        bias_interval = Interval(biases - self.quant_error, biases + self.quant_error)
        return weight_interval, bias_interval

    def propagate(self, input_interval):
        current_interval = input_interval
        for layer in self.model.children():
            if isinstance(layer, nn.Linear):
                weight_interval, bias_interval = self.to_interval(layer.weight, layer.bias)
                current_interval = interval_linear(current_interval, weight_interval, bias_interval)
            elif isinstance(layer, nn.ReLU):
                current_interval = Interval(
                    torch.relu(current_interval.lower),
                    torch.relu(current_interval.upper)
                )
        return current_interval
