import torch
from torch.quantization import quantize_dynamic, QuantWrapper, prepare_qat, convert
from models import PolicyNetwork

# Apply Post-Training Quantisation (PTQ)
def apply_ptq(policy):
    quantized_policy = quantize_dynamic(policy, {torch.nn.Linear}, dtype=torch.qint8)
    return quantized_policy

# Apply Quantisation-Aware Training (QAT)
def apply_qat(policy):
    quantized_policy = QuantWrapper(policy)
    quantized_policy.qconfig = torch.quantization.get_default_qat_qconfig('fbgemm')
    prepare_qat(quantized_policy, inplace=True)
    # Simulated fine-tuning
    optimizer = torch.optim.Adam(quantized_policy.parameters(), lr=1e-4)
    for epoch in range(10):  # Fine-tuning loop
        optimizer.zero_grad()
        # Simulated loss computation
        loss = torch.tensor(0.1, requires_grad=True)  # Dummy loss
        loss.backward()
        optimizer.step()
    convert(quantized_policy, inplace=True)
    return quantized_policy

if __name__ == "__main__":
    input_dim = 4  # CartPole observation space
    output_dim = 2  # CartPole action space

    policy = PolicyNetwork(input_dim, output_dim)
    policy.load_state_dict(torch.load("models/policy.pth"))

    # Apply PTQ
    ptq_policy = apply_ptq(policy)
    torch.save(ptq_policy.state_dict(), "models/ptq_policy.pth")

    # Apply QAT
    qat_policy = apply_qat(policy)
    torch.save(qat_policy.state_dict(), "models/qat_policy.pth")
