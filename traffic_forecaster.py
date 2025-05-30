# This is the traffic_forecaster.py file.
# It will contain classes and functions related to traffic forecasting.

import torch
import torch.nn as nn
from kan_layer import KANLayer # Added import

class TrafficForecasterNet(nn.Module):
    def __init__(self, 
                 input_sequence_length, 
                 num_features_per_step, 
                 output_sequence_length, 
                 hidden_dim_kan1=64, 
                 grid_size_kan=5, 
                 spline_order_kan=3, 
                 grid_range_kan=(-1.0, 1.0)): # Default grid_range
        super(TrafficForecasterNet, self).__init__()

        self.input_sequence_length = input_sequence_length
        self.num_features_per_step = num_features_per_step
        self.output_sequence_length = output_sequence_length
        
        self.flatten = nn.Flatten()
        
        flattened_input_dim = input_sequence_length * num_features_per_step
        
        # Instantiate the first KAN layer
        self.kan1 = KANLayer(
            input_dim=flattened_input_dim, 
            output_dim=hidden_dim_kan1, 
            grid_size=grid_size_kan, 
            spline_order=spline_order_kan, 
            grid_range=grid_range_kan
        )
        
        # Instantiate the output KAN layer
        # The output of kan1 is the input to kan_out
        self.kan_out = KANLayer(
            input_dim=hidden_dim_kan1, 
            output_dim=output_sequence_length, 
            grid_size=grid_size_kan, 
            spline_order=spline_order_kan, 
            grid_range=grid_range_kan # KAN paper suggests grid_range is often [-1,1] for hidden, but could be data-dependent for output
                                      # For simplicity, using the same grid_range. Output KAN might need different grid if output is not normalized.
        )

    def forward(self, x):
        """
        Defines the forward pass of the KAN-based TrafficForecasterNet.
        Args:
            x: Input tensor of shape (batch_size, input_sequence_length, num_features_per_step)
               IMPORTANT: KANLayers expect inputs to be normalized to their grid_range.
                          This model assumes input 'x' has features that are already (or will be prior to calling)
                          normalized approximately to the 'grid_range_kan' for optimal performance.
        Returns:
            Output tensor of shape (batch_size, output_sequence_length)
        """
        x = self.flatten(x) # Shape: (batch_size, input_sequence_length * num_features_per_step)
        
        # Note: KANLayers (as implemented) expect their direct input features to be within `grid_range`.
        # The output of self.kan1 might not be guaranteed to be within `grid_range_kan` for self.kan_out.
        # This is a common consideration in stacking KAN layers. Options:
        # 1. Normalize output of kan1 before passing to kan_out (e.g., with LayerNorm, or by knowing activation bounds).
        # 2. Assume spline activations in kan_out can handle varied input distribution from kan1.
        # 3. Use different grid_ranges for different KAN layers.
        # For this implementation, we'll directly pass it, assuming the KANs can adapt or that inputs to KANs are generally well-behaved.
        
        x = self.kan1(x)    # Shape: (batch_size, hidden_dim_kan1)
        x = self.kan_out(x) # Shape: (batch_size, output_sequence_length)
        
        return x

if __name__ == "__main__":
    # Define example parameters for the KAN-based network and input
    input_seq_len = 10
    features_per_step = 1
    output_seq_len = 1 
    batch_size = 4 # Increased batch size for better testing diversity

    # KAN specific parameters
    hidden_dim_kan1_test = 32 # Smaller hidden dim for quicker test
    grid_size_test = 5
    spline_order_test = 3
    # KANs typically expect input normalized to grid_range.
    # For testing, let's use [0,1] and generate dummy data in [0,1].
    grid_range_test = [0.0, 1.0] 

    print("--- KAN-based TrafficForecasterNet Test Block ---")
    print(f"Input sequence length: {input_seq_len}, Features per step: {features_per_step}")
    print(f"Output sequence length: {output_seq_len}, Batch size: {batch_size}")
    print(f"KAN params: hidden_dim={hidden_dim_kan1_test}, grid_size={grid_size_test}, spline_order={spline_order_test}, grid_range={grid_range_test}")

    # Instantiate the KAN-based network
    kan_forecaster_net = TrafficForecasterNet(
        input_sequence_length=input_seq_len,
        num_features_per_step=features_per_step,
        output_sequence_length=output_seq_len,
        hidden_dim_kan1=hidden_dim_kan1_test,
        grid_size_kan=grid_size_test,
        spline_order_kan=spline_order_test,
        grid_range_kan=grid_range_test
    )
    print("\nKAN-based TrafficForecasterNet instantiated.")
    # print(kan_forecaster_net) # Can be verbose due to KAN parameters

    # Create a dummy input tensor.
    # Data should be normalized to be within `grid_range_test` for KAN layers.
    # Here, [0,1] due to grid_range_test = [0.0, 1.0]
    # Input shape for network: (batch_size, input_seq_len, features_per_step)
    dummy_input_sequence = torch.rand(batch_size, input_seq_len, features_per_step, dtype=torch.float32)
    # If grid_range was [-1,1], dummy_input = 2 * torch.rand(...) - 1
    
    print(f"\nDummy Input Shape: {dummy_input_sequence.shape}")
    # print(f"Dummy Input Sample (first feature of first batch item): {dummy_input_sequence[0, :, 0]}")


    # Pass the dummy input through the network
    kan_forecaster_net.eval() 
    with torch.no_grad():
        prediction = kan_forecaster_net(dummy_input_sequence)

    print(f"Predicted Output Shape: {prediction.shape}") # Expected: (batch_size, output_seq_len)
    print(f"Predicted Output (first sample): {prediction[0]}")

    # Test backward pass
    # Re-enable gradients for parameters and create input that requires grad
    kan_forecaster_net.train() # Set back to train mode to ensure params are updated if model was eval
    dummy_input_grad_test = torch.rand(batch_size, input_seq_len, features_per_step, dtype=torch.float32, requires_grad=True)
    
    try:
        prediction_for_grad = kan_forecaster_net(dummy_input_grad_test)
        dummy_target = torch.randn_like(prediction_for_grad)
        loss = nn.MSELoss()(prediction_for_grad, dummy_target)
        loss.backward()
        print("\nBackward pass successful for KAN-based TrafficForecasterNet.")
        
        # Check a gradient for a KANLayer parameter
        if hasattr(kan_forecaster_net, 'kan1') and kan_forecaster_net.kan1.spline_coeffs.grad is not None:
            print("Gradient sum for kan1.spline_coeffs:", torch.sum(kan_forecaster_net.kan1.spline_coeffs.grad).item())
        else:
            print("No gradient for kan1.spline_coeffs or kan1 does not exist.")
            
    except Exception as e:
        print(f"\nError during backward pass test for KAN-based net: {e}")


    # (Optional) Check KANLayer import again, though it's implicitly tested by using TrafficForecasterNet
    # try:
    #     kan_test_layer = KANLayer(input_dim=5, output_dim=2) # Default grid_range [-1,1]
    #     print("\nSuccessfully re-instantiated a KANLayer (from traffic_forecaster.py test block).")
    # except Exception as e:
    #     print(f"\nError during KANLayer basic re-check in traffic_forecaster.py: {e}")

    print("\n--- Test Block Finished ---")
