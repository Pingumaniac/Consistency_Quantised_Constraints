import unittest
import gymnasium as gym
import torch
from src.models import PolicyNetwork
from src.evaluate_models import evaluate_policy

class TestEvaluateModels(unittest.TestCase):
    def setUp(self):
        self.env = gym.make('CartPole-v1')
        self.input_dim = self.env.observation_space.shape[0]
        self.output_dim = self.env.action_space.n

        self.policy = PolicyNetwork(self.input_dim, self.output_dim)
        self.policy.load_state_dict(torch.load("models/policy.pth"))

        self.ptq_policy = PolicyNetwork(self.input_dim, self.output_dim)
        self.ptq_policy.load_state_dict(torch.load("models/ptq_policy.pth"))

    def test_evaluation_produces_rewards(self):
        baseline_reward = evaluate_policy(self.env, self.policy)
        ptq_reward = evaluate_policy(self.env, self.ptq_policy)
        self.assertGreater(baseline_reward, 0, "Baseline model should produce rewards")
        self.assertGreater(ptq_reward, 0, "Quantized model should produce rewards")

if __name__ == '__main__':
    unittest.main()
