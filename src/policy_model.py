import torch.nn as nn
import torch

class PolicyNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return torch.softmax(self.fc2(x), dim=-1)

def save_policy_model(model, path="models/policy.pth"):
    torch.save(model.state_dict(), path)

def load_policy_model(path="models/policy.pth", input_dim=4, output_dim=2):
    model = PolicyNetwork(input_dim, output_dim)
    model.load_state_dict(torch.load(path))
    return model
