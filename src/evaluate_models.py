# evaluate_models.py

import gymnasium as gym
import torch
from models import PolicyNetwork
import warnings

# Suppress TypedStorage deprecation warning
warnings.filterwarnings("ignore", message="TypedStorage is deprecated")

def load_quantized_model(model_path, input_dim, output_dim):
    """
    Load a quantized model with dynamic quantization applied.
    """
    # Instantiate the model and dynamically quantize it
    model = PolicyNetwork(input_dim, output_dim)
    model = torch.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)
    # Load the state dictionary
    model.load_state_dict(torch.load(model_path))
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
    torch.backends.quantized.engine = 'fbgemm'

    # Device setup (use CPU for quantized models)
    device = torch.device("cpu")

    # Load models
    print("Loading baseline model...")
    policy = PolicyNetwork(input_dim, output_dim)
    policy.load_state_dict(torch.load("./models/policy.pth"))
    policy.to(device)

    print("Loading PTQ model...")
    ptq_policy = load_quantized_model("./models/ptq_policy.pth", input_dim, output_dim)
    ptq_policy.to(device)

    print("Loading QAT model...")
    qat_policy = load_quantized_model("./models/qat_policy.pth", input_dim, output_dim)
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
