import gymnasium as gym
import torch
import torch.nn as nn
from models import PolicyNetwork
import warnings

# Suppress TypedStorage deprecation warning
warnings.filterwarnings("ignore", message="TypedStorage is deprecated")

def prepare_quantized_model(model):
    """
    Prepare the model for dynamic quantization.
    """
    model.fc1 = nn.quantized.dynamic.Linear(model.fc1.in_features, model.fc1.out_features)
    model.fc2 = nn.quantized.dynamic.Linear(model.fc2.in_features, model.fc2.out_features)
    return model

def prepare_qat_model(model):
    """
    Prepare the model for Quantization-Aware Training (QAT).
    """
    model.qconfig = torch.quantization.get_default_qconfig('qnnpack')
    torch.quantization.prepare_qat(model, inplace=True)
    torch.quantization.convert(model, inplace=True)
    return model

def evaluate_policy(env, policy, num_episodes=10, max_steps=100, device="cpu"):
    """
    Evaluate the policy on the given environment with a maximum number of steps per episode.
    """
    total_rewards = []
    for episode in range(num_episodes):
        state, _ = env.reset()
        total_reward = 0
        for step in range(max_steps):
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
            action_prob = policy(state_tensor)
            action = torch.argmax(action_prob).item()
            state, reward, done, _, _ = env.step(action)
            total_reward += reward
            if done:
                break
        total_rewards.append(total_reward)
        print(f"Episode {episode + 1} completed with total reward: {total_reward}")
    return sum(total_rewards) / num_episodes

if __name__ == "__main__":
    # Wrap environment with a time limit
    env = gym.make('CartPole-v1', max_episode_steps=200)
    input_dim = env.observation_space.shape[0]
    output_dim = env.action_space.n

    # Set the quantized backend
    torch.backends.quantized.engine = 'qnnpack'

    # Device setup (use GPU if available)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load models
    print("Loading baseline model...")
    policy = PolicyNetwork(input_dim, output_dim)
    policy.load_state_dict(torch.load("./models/policy.pth"))
    policy.to(device)

    print("Loading PTQ model...")
    ptq_policy = PolicyNetwork(input_dim, output_dim)
    ptq_policy = prepare_quantized_model(ptq_policy)
    ptq_policy.load_state_dict(torch.load("./models/ptq_policy.pth"))
    ptq_policy.to(device)

    print("Loading QAT model...")
    qat_policy = PolicyNetwork(input_dim, output_dim)
    qat_policy = prepare_qat_model(qat_policy)
    qat_policy.load_state_dict(torch.load("./models/qat_policy.pth"))
    qat_policy.to(device)

    # Evaluate models
    print("Evaluating baseline model...")
    baseline_reward = evaluate_policy(env, policy, num_episodes=3, max_steps=200, device=device)
    print(f"Baseline Reward: {baseline_reward}")

    print("Evaluating PTQ model...")
    ptq_reward = evaluate_policy(env, ptq_policy, num_episodes=3, max_steps=200, device=device)
    print(f"PTQ Reward: {ptq_reward}")

    print("Evaluating QAT model...")
    qat_reward = evaluate_policy(env, qat_policy, num_episodes=3, max_steps=200, device=device)
    print(f"QAT Reward: {qat_reward}")
