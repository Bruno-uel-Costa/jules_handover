# This is the traffic_forecaster.py file.
# It will contain classes and functions related to traffic forecasting.

import torch
import torch.nn as nn

class TrafficForecasterNet(nn.Module):
    def __init__(self, input_sequence_length, num_features_per_step, output_sequence_length, hidden_dim1=128, hidden_dim2=64):
        super(TrafficForecasterNet, self).__init__()

        # Store dimensions if needed for other methods or debugging, otherwise, they are used directly here.
        self.input_sequence_length = input_sequence_length
        self.num_features_per_step = num_features_per_step
        self.output_sequence_length = output_sequence_length

        # Define the layers
        self.flatten = nn.Flatten()
        
        # Calculate the input size for the first fully connected layer
        flattened_input_size = input_sequence_length * num_features_per_step
        
        self.fc1 = nn.Linear(flattened_input_size, hidden_dim1)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim1, hidden_dim2)
        self.relu2 = nn.ReLU()
        
        # The output layer predicts a sequence of length `output_sequence_length`.
        # If each step in the output sequence is a single value, this is correct.
        # If each step in the output sequence has multiple features, this would need adjustment.
        # Based on typical forecasting of a single variable (e.g., traffic volume), this is a common setup.
        self.fc_out = nn.Linear(hidden_dim2, output_sequence_length)

    def forward(self, x):
        """
        Defines the forward pass of the TrafficForecasterNet.
        Args:
            x: Input tensor of shape (batch_size, input_sequence_length, num_features_per_step)
        Returns:
            Output tensor of shape (batch_size, output_sequence_length)
        """
        # Flatten the input sequence
        # Input x shape: (batch_size, seq_len, features_per_step)
        # After flatten: (batch_size, seq_len * features_per_step)
        x = self.flatten(x)
        
        # Pass through the first fully connected layer and ReLU activation
        x = self.relu1(self.fc1(x))
        
        # Pass through the second fully connected layer and ReLU activation
        x = self.relu2(self.fc2(x))
        
        # Pass through the output layer
        x = self.fc_out(x)
        
        return x

if __name__ == "__main__":
    # Define example parameters for the network and input
    input_seq_len = 10       # Length of the input sequence (e.g., 10 time steps)
    features_per_step = 1    # Number of features at each time step (e.g., just traffic volume)
    output_seq_len = 1       # Length of the output sequence (e.g., predict 1 time step ahead)
    batch_size = 1           # Number of samples in a batch

    print("--- TrafficForecasterNet Test Block ---")
    print(f"Input sequence length: {input_seq_len}")
    print(f"Features per step: {features_per_step}")
    print(f"Output sequence length: {output_seq_len}")
    print(f"Batch size: {batch_size}")

    # Instantiate the network
    forecaster_net = TrafficForecasterNet(
        input_sequence_length=input_seq_len,
        num_features_per_step=features_per_step,
        output_sequence_length=output_seq_len
    )
    print("\nTrafficForecasterNet instantiated.")
    print(forecaster_net)

    # Create a dummy input tensor
    # Shape: (batch_size, input_sequence_length, num_features_per_step)
    dummy_input_sequence = torch.randn(batch_size, input_seq_len, features_per_step)
    
    # Pass the dummy input through the network
    # Set the network to evaluation mode (important if dropout/batchnorm were used)
    forecaster_net.eval() 
    with torch.no_grad(): # No need to track gradients for this test pass
        prediction = forecaster_net(dummy_input_sequence)

    # Print the shapes and values
    print(f"\nDummy Input Shape: {dummy_input_sequence.shape}")
    print(f"Predicted Output Shape: {prediction.shape}") # Expected: (batch_size, output_seq_len)
    print(f"Predicted Output: {prediction}")
    
    print("\n--- Test Block Finished ---")
