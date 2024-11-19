import gymnasium as gym
import torch
from models import PolicyNetwork

def evaluate_policy(env, policy, num_episodes=100):
    total_rewards = []
    for _ in range(num_episodes):
        state, _ = env.reset()
        done = False
        total_reward = 0
        while not done:
            state_tensor = torch.tensor(state, dtype=torch.float32)
            action_prob = policy(state_tensor)
            action = torch.argmax(action_prob).item()
            state, reward, done, _, _ = env.step(action)
            total_reward += reward
        total_rewards.append(total_reward)
    return sum(total_rewards) / num_episodes

if __name__ == "__main__":
    env = gym.make('CartPole-v1')
    input_dim = env.observation_space.shape[0]
    output_dim = env.action_space.n

    # Load models
    policy = PolicyNetwork(input_dim, output_dim)
    policy.load_state_dict(torch.load("models/policy.pth"))

    ptq_policy = PolicyNetwork(input_dim, output_dim)
    ptq_policy.load_state_dict(torch.load("models/ptq_policy.pth"))

    # Evaluate models
    baseline_reward = evaluate_policy(env, policy)
    ptq_reward = evaluate_policy(env, ptq_policy)

    print(f"Baseline Reward: {baseline_reward}")
    print(f"PTQ Reward: {ptq_reward}")
