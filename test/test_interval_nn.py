import unittest
import torch
from models import PolicyNetwork
from src.interval_nn import Interval, IntervalNeuralNetwork

class TestIntervalNN(unittest.TestCase):
    def setUp(self):
        self.input_dim = 4
        self.output_dim = 2
        self.policy = PolicyNetwork(self.input_dim, self.output_dim)
        self.policy.load_state_dict(torch.load("models/policy.pth"))  # Load trained baseline model
        self.quant_error = 0.01
        self.interval_nn = IntervalNeuralNetwork(self.policy, self.quant_error)

    def test_interval_creation(self):
        lower = torch.tensor([1.0, 2.0])
        upper = torch.tensor([1.5, 2.5])
        interval = Interval(lower, upper)
        self.assertTrue(torch.equal(interval.lower, lower), "Interval lower bound mismatch")
        self.assertTrue(torch.equal(interval.upper, upper), "Interval upper bound mismatch")
        self.assertTrue(torch.all(interval.upper >= interval.lower), "Upper bounds must be >= lower bounds")

    def test_interval_linear_layer(self):
        input_interval = Interval(
            torch.tensor([1.0, 2.0, 3.0, 4.0]) - self.quant_error,
            torch.tensor([1.0, 2.0, 3.0, 4.0]) + self.quant_error
        )
        weight = torch.rand((self.output_dim, self.input_dim))
        bias = torch.rand(self.output_dim)

        weight_interval = Interval(weight - self.quant_error, weight + self.quant_error)
        bias_interval = Interval(bias - self.quant_error, bias + self.quant_error)

        output_interval = self.interval_nn.propagate_linear(input_interval, weight_interval, bias_interval)

        self.assertEqual(output_interval.lower.shape, (self.output_dim,))
        self.assertEqual(output_interval.upper.shape, (self.output_dim,))
        self.assertTrue(torch.all(output_interval.upper >= output_interval.lower), "Invalid interval bounds")

    def test_interval_network_propagation(self):
        input_tensor = torch.tensor([1.0, 2.0, 3.0, 4.0])
        input_interval = Interval(input_tensor - self.quant_error, input_tensor + self.quant_error)

        output_interval = self.interval_nn.propagate(input_interval)

        self.assertEqual(output_interval.lower.shape, (self.output_dim,))
        self.assertEqual(output_interval.upper.shape, (self.output_dim,))
        self.assertTrue(torch.all(output_interval.upper >= output_interval.lower), "Invalid output interval bounds")

    def test_consistency_with_baseline(self):
        input_tensor = torch.tensor([1.0, 2.0, 3.0, 4.0])
        unquantised_output = self.policy(input_tensor)
        input_interval = Interval(input_tensor - self.quant_error, input_tensor + self.quant_error)

        output_interval = self.interval_nn.propagate(input_interval)

        self.assertTrue(torch.all(unquantised_output >= output_interval.lower), "Unquantised output out of interval bounds")
        self.assertTrue(torch.all(unquantised_output <= output_interval.upper), "Unquantised output out of interval bounds")

if __name__ == "__main__":
    unittest.main()
