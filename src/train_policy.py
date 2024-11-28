import os
import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
from models import PolicyNetwork

def train_policy(env, policy, optimiser, num_episodes):
    for episode in range(num_episodes):
        state, _ = env.reset()
        log_probs = []
        rewards = []
        done = False

        while not done:
            state_tensor = torch.tensor(state, dtype=torch.float32)
            action_probs = policy(state_tensor)
            action = torch.multinomial(action_probs, num_samples=1).item()
            log_probs.append(torch.log(action_probs[action]))

            state, reward, done, _, _ = env.step(action)
            rewards.append(reward)

        discounted_rewards = []
        total_reward = 0
        for r in reversed(rewards):
            total_reward = r + 0.99 * total_reward
            discounted_rewards.insert(0, total_reward)

        # Normalize rewards
        discounted_rewards = torch.tensor(discounted_rewards)
        if discounted_rewards.std() > 0:
            discounted_rewards = (discounted_rewards - discounted_rewards.mean()) / (discounted_rewards.std() + 1e-9)

        # Compute policy loss
        policy_loss = []
        for log_prob, reward in zip(log_probs, discounted_rewards):
            policy_loss.append(-log_prob * reward)
        optimiser.zero_grad()
        policy_loss = torch.stack(policy_loss).sum()
        policy_loss.backward()
        optimiser.step()

        print(f"Episode {episode + 1}/{num_episodes}, Total Reward: {sum(rewards)}")


if __name__ == "__main__":
    env = gym.make('CartPole-v1')
    input_dim = env.observation_space.shape[0]
    output_dim = env.action_space.n

    policy = PolicyNetwork(input_dim, output_dim)
    optimiser = optim.Adam(policy.parameters(), lr=1e-3)

    train_policy(env, policy, optimiser, num_episodes=500)

    os.makedirs("./models", exist_ok=True)
    torch.save(policy.state_dict(), "./models/policy.pth")
    print("Model saved to ./models/policy.pth")
