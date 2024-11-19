import unittest
import torch
from src.models import PolicyNetwork
from apply_quantisation import apply_ptq, apply_qat

class TestApplyQuantisation(unittest.TestCase):
    def setUp(self):
        self.input_dim = 4
        self.output_dim = 2
        self.policy = PolicyNetwork(self.input_dim, self.output_dim)
        self.policy.load_state_dict(torch.load("models/policy.pth"))

    def test_ptq_reduces_size(self):
        ptq_policy = apply_ptq(self.policy)
        unquantized_size = sum(p.numel() for p in self.policy.parameters())
        quantized_size = sum(p.numel() for p in ptq_policy.parameters())
        self.assertLess(quantized_size, unquantized_size, "PTQ should reduce model size")

    def test_qat_fine_tuning_preserves_accuracy(self):
        qat_policy = apply_qat(self.policy)
        input_sample = torch.rand(self.input_dim)
        original_output = self.policy(input_sample)
        quantized_output = qat_policy(input_sample)
        self.assertTrue(torch.allclose(original_output, quantized_output, atol=0.1), "QAT should preserve accuracy")

if __name__ == '__main__':
    unittest.main()
