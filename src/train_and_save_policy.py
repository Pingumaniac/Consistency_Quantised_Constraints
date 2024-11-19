import torch
from policy_model import PolicyNetwork, save_policy_model
import gymnasium as gym
from train_policy import train_policy

env = gym.make('CartPole-v1')
input_dim = env.observation_space.shape[0]
output_dim = env.action_space.n
policy = PolicyNetwork(input_dim, output_dim)

optimiser = torch.optim.Adam(policy.parameters(), lr=1e-3)
train_policy(env, policy, optimiser, num_episodes=500)

save_policy_model(policy, path="models/policy.pth")
print("Baseline policy model saved as models/policy.pth")
