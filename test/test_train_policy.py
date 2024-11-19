import unittest
import gymnasium as gym
import torch
from src.models import PolicyNetwork
from src.train_policy import train_policy

class TestTrainPolicy(unittest.TestCase):
    def setUp(self):
        self.env = gym.make('CartPole-v1')
        self.input_dim = self.env.observation_space.shape[0]
        self.output_dim = self.env.action_space.n
        self.policy = PolicyNetwork(self.input_dim, self.output_dim)
        self.optimszer = torch.optim.Adam(self.policy.parameters(), lr=1e-3)

    def test_training_improves_reward(self):
        # Measure initial reward
        state, _ = self.env.reset()
        initial_reward = 0
        for _ in range(10):  # Simulate 10 steps
            action = torch.argmax(self.policy(torch.tensor(state, dtype=torch.float32))).item()
            state, reward, done, _, _ = self.env.step(action)
            initial_reward += reward
            if done:
                break

        # Train policy
        train_policy(self.env, self.policy, self.optimiser, num_episodes=10)

        # Measure reward after training
        state, _ = self.env.reset()
        trained_reward = 0
        for _ in range(10):  # Simulate 10 steps
            action = torch.argmax(self.policy(torch.tensor(state, dtype=torch.float32))).item()
            state, reward, done, _, _ = self.env.step(action)
            trained_reward += reward
            if done:
                break

        self.assertGreater(trained_reward, initial_reward, "Training should improve reward")

if __name__ == '__main__':
    unittest.main()
