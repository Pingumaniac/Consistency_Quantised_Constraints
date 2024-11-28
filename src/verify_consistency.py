# verify_consistency.py

import torch
from interval_nn import Interval, IntervalNeuralNetwork
from models import PolicyNetwork
import torch.nn as nn


def prepare_quantized_model(model, is_dynamic=True):
    """
    Prepare the model for quantization.
    Args:
        model: The baseline model.
        is_dynamic: If True, prepares the model for dynamic quantization (PTQ).
                    If False, assumes the model is already quantized (QAT).
    """
    if is_dynamic:
        model.fc1 = nn.quantized.dynamic.Linear(model.fc1.in_features, model.fc1.out_features)
        model.fc2 = nn.quantized.dynamic.Linear(model.fc2.in_features, model.fc2.out_features)
    return model


def verify_decision_consistency(unquantised_model, quantised_model, test_inputs, quant_error):
    """
    Verify consistency between unquantized and quantized models.
    """
    interval_model = IntervalNeuralNetwork(quantised_model, quant_error)
    consistent = True
    for i, test_input in enumerate(test_inputs):
        input_interval = Interval(test_input, test_input)
        unquantised_output = unquantised_model(test_input)
        interval_output = interval_model.propagate(input_interval)

        print(f"Test Input {i}:")
        print(f"Baseline Output: {unquantised_output}")
        print(f"Quantized Output Interval: {interval_output.lower}, {interval_output.upper}")

        if not ((interval_output.lower <= unquantised_output).all() and (unquantised_output <= interval_output.upper).all()):
            consistent = False
            break
    return consistent


if __name__ == "__main__":
    input_dim = 4  # CartPole observation space
    output_dim = 2  # CartPole action space
    quant_error = 0.05  # Adjusted quantization error

    # Set the quantized backend
    torch.backends.quantized.engine = 'qnnpack'

    # Load the baseline model
    policy = PolicyNetwork(input_dim, output_dim)
    policy.load_state_dict(torch.load("./models/policy.pth"))

    # Load the PTQ model
    ptq_policy = PolicyNetwork(input_dim, output_dim)
    ptq_policy = prepare_quantized_model(ptq_policy, is_dynamic=True)
    ptq_policy.load_state_dict(torch.load("./models/ptq_policy.pth"))

    # Generate test inputs
    test_inputs = [torch.rand(input_dim) for _ in range(10)]

    # Verify decision consistency for PTQ
    consistency_ptq = verify_decision_consistency(policy, ptq_policy, test_inputs, quant_error)
    print(f"Decision Consistency (Baseline vs PTQ): {consistency_ptq}")
