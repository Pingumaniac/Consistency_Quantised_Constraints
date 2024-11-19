import unittest
import torch
from src.models import PolicyNetwork
from src.verify_consistency import verify_decision_consistency
from src.interval_nn import IntervalNeuralNetwork

class TestVerifyConsistency(unittest.TestCase):
    def setUp(self):
        self.input_dim = 4
        self.output_dim = 2
        self.policy = PolicyNetwork(self.input_dim, self.output_dim)
        self.policy.load_state_dict(torch.load("models/policy.pth"))

        self.ptq_policy = PolicyNetwork(self.input_dim, self.output_dim)
        self.ptq_policy.load_state_dict(torch.load("models/ptq_policy.pth"))

    def test_consistency_within_bounds(self):
        test_inputs = [torch.rand(self.input_dim) for _ in range(10)]
        quant_error = 0.01
        consistency = verify_decision_consistency(self.policy, self.ptq_policy, test_inputs, quant_error)
        self.assertTrue(consistency, "Quantised model should remain consistent with the baseline")

if __name__ == '__main__':
    unittest.main()
