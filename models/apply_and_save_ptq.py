import torch
from models.policy_model import load_policy_model
from models.ptq_model import apply_ptq, save_ptq_model

policy = load_policy_model(path="models/policy.pth", input_dim=4, output_dim=2)

ptq_policy = apply_ptq(policy)

save_ptq_model(ptq_policy, path="models/ptq_policy.pth")
print("PTQ policy model saved as models/ptq_policy.pth")
