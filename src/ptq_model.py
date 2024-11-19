import torch.nn as nn
import torch
from torch.quantization import quantize_dynamic

class PTQPolicyNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(PTQPolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return torch.softmax(self.fc2(x), dim=-1)

def save_ptq_model(model, path="models/ptq_policy.pth"):
    torch.save(model.state_dict(), path)

def load_ptq_model(path="models/ptq_policy.pth", input_dim=4, output_dim=2):
    model = PTQPolicyNetwork(input_dim, output_dim)
    model.load_state_dict(torch.load(path))
    return model

def apply_ptq(policy_model):
    return quantize_dynamic(policy_model, {nn.Linear}, dtype=torch.qint8)
