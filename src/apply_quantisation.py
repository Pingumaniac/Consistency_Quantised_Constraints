import torch
from torch.quantization import quantize_dynamic, QuantWrapper, prepare_qat, convert
from models import PolicyNetwork

# Apply Post-Training Quantization (PTQ)
def apply_ptq(policy):
    torch.backends.quantized.engine = 'qnnpack'
    quantised_policy = quantize_dynamic(policy, {torch.nn.Linear}, dtype=torch.qint8)
    return quantised_policy

# Apply Quantization-Aware Training (QAT)
def apply_qat(policy, calibration_data):
    quantised_policy = QuantWrapper(policy)
    quantised_policy.qconfig = torch.quantization.get_default_qat_qconfig('qnnpack')
    prepare_qat(quantised_policy, inplace=True)

    # Calibrate the model with real input data
    for data in calibration_data:
        with torch.no_grad():
            quantised_policy(data)

    # Simulated fine-tuning
    optimiser = torch.optim.Adam(quantised_policy.parameters(), lr=1e-4)
    for epoch in range(10):  # Fine-tuning loop
        optimiser.zero_grad()
        # Simulated loss computation
        loss = torch.tensor(0.1, requires_grad=True)  # Dummy loss
        loss.backward()
        optimiser.step()

    convert(quantised_policy, inplace=True)
    return quantised_policy


if __name__ == "__main__":
    input_dim = 4  # CartPole observation space
    output_dim = 2  # CartPole action space

    # Load the trained policy
    policy = PolicyNetwork(input_dim, output_dim)
    policy.load_state_dict(torch.load("./models/policy.pth"))

    # Generate dummy calibration data for demonstration
    calibration_data = [torch.randn(1, input_dim) for _ in range(100)]

    # Apply PTQ
    ptq_policy = apply_ptq(policy)
    torch.save(ptq_policy.state_dict(), "./models/ptq_policy.pth")
    print("PTQ model saved to './models/ptq_policy.pth'.")

    # Apply QAT
    qat_policy = apply_qat(policy, calibration_data)
    torch.save(qat_policy.state_dict(), "./models/qat_policy.pth")
    print("QAT model saved to './models/qat_policy.pth'.")
