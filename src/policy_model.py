import torch.nn as nn
import torch

class PolicyNetwork(nn.Module):
    def __init__(self, input_dim, output_dim, quantized=False):
        """
        Policy Network with optional support for quantized layers.
        Args:
            input_dim (int): Number of input features.
            output_dim (int): Number of output actions.
            quantized (bool): Whether to use quantized layers.
        """
        super(PolicyNetwork, self).__init__()
        if quantized:
            # Use quantized layers for a quantized model
            self.fc1 = nn.quantized.Linear(input_dim, 128)
            self.fc2 = nn.quantized.Linear(128, output_dim)
        else:
            # Use standard layers for an unquantized model
            self.fc1 = nn.Linear(input_dim, 128)
            self.fc2 = nn.Linear(128, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return torch.softmax(self.fc2(x), dim=-1)

def save_policy_model(model, path="models/policy.pth"):
    torch.save(model.state_dict(), path)

def load_policy_model(path="models/policy.pth", input_dim=4, output_dim=2, quantized=False):
    """
    Load a Policy Network from the given path.
    Args:
        path (str): Path to the saved model.
        input_dim (int): Number of input features.
        output_dim (int): Number of output actions.
        quantized (bool): Whether the model is quantized.
    Returns:
        PolicyNetwork: The loaded model.
    """
    model = PolicyNetwork(input_dim, output_dim, quantized=quantized)
    model.load_state_dict(torch.load(path))
    return model
