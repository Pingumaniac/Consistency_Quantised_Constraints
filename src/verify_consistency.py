import torch
from interval_nn import Interval, IntervalNeuralNetwork
from models import PolicyNetwork

def verify_decision_consistency(unquantised_model, quantised_model, test_inputs, quant_error):
    interval_model = IntervalNeuralNetwork(quantised_model, quant_error)
    consistent = True
    for test_input in test_inputs:
        input_interval = Interval(test_input, test_input)
        unquantised_output = unquantised_model(test_input)
        interval_output = interval_model.propagate(input_interval)

        if not (interval_output.lower <= unquantised_output <= interval_output.upper).all():
            consistent = False
            break
    return consistent

if __name__ == "__main__":
    input_dim = 4  # CartPole observation space
    output_dim = 2  # CartPole action space
    quant_error = 0.01  # Quantisation error can be adjusted

    policy = PolicyNetwork(input_dim, output_dim)
    policy.load_state_dict(torch.load("models/policy.pth"))

    ptq_policy = PolicyNetwork(input_dim, output_dim)
    ptq_policy.load_state_dict(torch.load("models/ptq_policy.pth"))

    test_inputs = [torch.rand(input_dim) for _ in range(10)]

    consistency = verify_decision_consistency(policy, ptq_policy, test_inputs, quant_error)
    print(f"Decision Consistency (Baseline vs PTQ): {consistency}")
