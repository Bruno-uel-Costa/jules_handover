# This is the traffic_forecaster.py file.
# It will contain classes and functions related to traffic forecasting.

import torch
import torch.nn as nn
import numpy as np # Added numpy import
from kan_layer import KANLayer 

def generate_sine_wave_data(num_points=500, sequence_length=10, predict_steps=1, noise_level=0.05, normalization_range=(0.0, 1.0)):
    # num_points: total de pontos na série temporal
    # sequence_length: o L da nossa entrada (X_t, X_t-1, ..., X_t-L+1)
    # predict_steps: How many steps ahead to predict (for this task, it's 1, so Y is the next point)
    # noise_level: magnitude do ruído
    # normalization_range: tuple (min_val, max_val) for data normalization

    time_steps = np.linspace(0, 100, num_points) 
    data = np.sin(time_steps * 0.5) + np.sin(time_steps * 0.1) 
    data += np.random.normal(0, noise_level, num_points)

    # Normalize data to the specified range
    data_min_orig = np.min(data)
    data_max_orig = np.max(data)
    
    norm_min, norm_max = normalization_range
    
    if data_max_orig - data_min_orig != 0:
         data_normalized = norm_min + (data - data_min_orig) * (norm_max - norm_min) / (data_max_orig - data_min_orig)
    else: 
         # Handle cases where data is constant, to avoid division by zero
         data_normalized = np.full_like(data, norm_min) # All values will be norm_min if data is constant

    X, Y = [], []
    for i in range(len(data_normalized) - sequence_length - predict_steps + 1):
        X.append(data_normalized[i:(i + sequence_length)])
        Y.append(data_normalized[i + sequence_length : i + sequence_length + predict_steps])

    X_tensor = torch.tensor(np.array(X), dtype=torch.float32).unsqueeze(-1) 
    Y_tensor = torch.tensor(np.array(Y), dtype=torch.float32)
    
    if predict_steps == 1 and len(Y_tensor.shape) == 1: 
        Y_tensor = Y_tensor.unsqueeze(-1) 
    elif predict_steps > 1 and len(Y_tensor.shape) == 1: 
        Y_tensor = Y_tensor.reshape(-1, predict_steps)
    elif len(Y_tensor.shape) == 2 and Y_tensor.shape[1] != predict_steps: 
        raise ValueError(f"Y_tensor shape {Y_tensor.shape} not consistent with predict_steps {predict_steps}")

    return X_tensor, Y_tensor, data_normalized, time_steps


class TrafficForecasterNet(nn.Module):
    def __init__(self, 
                 input_sequence_length, 
                 num_features_per_step, 
                 output_sequence_length, 
                 hidden_dim_kan1=64, 
                 grid_size_kan=5, 
                 spline_order_kan=3, 
                 grid_range_kan=(-1.0, 1.0)): 
        super(TrafficForecasterNet, self).__init__()

        self.input_sequence_length = input_sequence_length
        self.num_features_per_step = num_features_per_step
        self.output_sequence_length = output_sequence_length
        
        self.flatten = nn.Flatten()
        
        flattened_input_dim = input_sequence_length * num_features_per_step
        
        self.kan1 = KANLayer(
            input_dim=flattened_input_dim, 
            output_dim=hidden_dim_kan1, 
            grid_size=grid_size_kan, 
            spline_order=spline_order_kan, 
            grid_range=grid_range_kan
        )
        
        self.kan_out = KANLayer(
            input_dim=hidden_dim_kan1, 
            output_dim=output_sequence_length, 
            grid_size=grid_size_kan, 
            spline_order=spline_order_kan, 
            grid_range=grid_range_kan 
        )

    def forward(self, x):
        x = self.flatten(x) 
        x = self.kan1(x)    
        x = self.kan_out(x) 
        return x

if __name__ == "__main__":
    print("--- TrafficForecasterNet Training Setup ---")

    # 1. Define Training Hyperparameters
    learning_rate = 0.001
    num_epochs = 200  # Increased for a potential trend
    # Batch size will be implicitly defined by using the full X_train, Y_train for now (full-batch training)

    # 2. Define Network and KAN Parameters
    input_seq_len = 20       # Input sequence length for data generation and network
    features_per_step = 1    # Number of features at each time step (sine wave is univariate)
    output_seq_len = 1       # Predicting one step ahead
    
    hidden_dim_kan1 = 32     # KAN hidden dimension
    grid_size_kan = 5        # KAN grid size
    spline_order_kan = 3     # KAN spline order (cubic)
    
    # Normalization range for data generation and KAN grid_range
    # KANs expect input features to be within their grid_range.
    normalization_target_range = (0.0, 1.0) 

    print(f"Training Params: LR={learning_rate}, Epochs={num_epochs}")
    print(f"Network Params: InputSeqLen={input_seq_len}, OutputSeqLen={output_seq_len}, HiddenKAN={hidden_dim_kan1}")
    print(f"KAN Config: GridSize={grid_size_kan}, SplineOrder={spline_order_kan}, GridRange={normalization_target_range}")

    # 3. Generate Synthetic Data
    # Data will be normalized to normalization_target_range, suitable for KAN input.
    # We are not using data_normalized_plot or time_steps_plot in this version of the plotting code.
    X_train, Y_train, _, _ = generate_sine_wave_data(
        num_points=500, 
        sequence_length=input_seq_len, 
        predict_steps=output_seq_len, 
        noise_level=0.05, 
        normalization_range=normalization_target_range
    )
    print(f"Generated training data: X_train shape {X_train.shape}, Y_train shape {Y_train.shape}")

    # 4. Instantiate the Network & Device Handling
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    kan_forecaster_net = TrafficForecasterNet(
        input_sequence_length=input_seq_len,
        num_features_per_step=features_per_step,
        output_sequence_length=output_seq_len,
        hidden_dim_kan1=hidden_dim_kan1,
        grid_size_kan=grid_size_kan,
        spline_order_kan=spline_order_kan,
        grid_range_kan=normalization_target_range # Critical: KAN grid matches data normalization
    ).to(device)
    
    X_train = X_train.to(device)
    Y_train = Y_train.to(device)
    
    print("KAN-based TrafficForecasterNet instantiated and moved to device.")

    # 5. Define Loss Function and Optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(kan_forecaster_net.parameters(), lr=learning_rate)
    
    print("Loss function (MSELoss) and Optimizer (Adam) defined.")
    print("\n--- Training Setup Complete. Starting Training Loop ---")

    # Training loop
    loss = None # Initialize loss to ensure it's available if num_epochs is 0
    for epoch in range(num_epochs):
        kan_forecaster_net.train()  # Set model to training mode

        # Forward pass
        predictions = kan_forecaster_net(X_train) # X_train is already on the correct device
        
        # Calculate loss
        loss = criterion(predictions, Y_train) # Y_train is already on the correct device
        
        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.6f}")

    print("--- Training complete. ---")
    if loss is not None: # Check if training loop ran at least once
        print(f"Final Loss after {num_epochs} epochs: {loss.item():.6f}")
    else:
        print("Training did not run (num_epochs might be 0).")


    # Visualization (Optional, requires matplotlib)
    print("\n--- Visualization Section ---")
    try:
        import matplotlib.pyplot as plt
        
        print("Attempting to visualize predictions...")
        kan_forecaster_net.eval() # Set model to evaluation mode
        with torch.no_grad():
            # X_train is already on the correct device
            trained_predictions = kan_forecaster_net(X_train)

        # Move data to CPU and convert to NumPy for plotting
        original_data_plot = Y_train.cpu().numpy().squeeze() 
        predictions_plot = trained_predictions.cpu().numpy().squeeze()
        
        # Plotting against sample index
        plt.figure(figsize=(12, 6))
        
        # Plot a segment for clarity if data is large
        plot_segment_length = 200
        original_segment = original_data_plot[:plot_segment_length]
        predicted_segment = predictions_plot[:plot_segment_length]
        indices = np.arange(len(original_segment))

        plt.plot(indices, original_segment, label=f'Original Y_train (Normalized Segment - First {plot_segment_length} points)', alpha=0.7, marker='.')
        plt.plot(indices, predicted_segment, label=f'Trained Predictions (Normalized Segment - First {plot_segment_length} points)', alpha=0.7, marker='x')

        plt.title('TrafficForecasterNet (KAN) - Training Sanity Check')
        plt.xlabel('Sample Index (Segment)')
        plt.ylabel('Normalized Value')
        plt.legend()
        plt.grid(True)
        # In a headless environment, plt.show() might not work or might block.
        # Saving the plot is often more reliable.
        plot_filename = 'training_sanity_check.png'
        plt.savefig(plot_filename)
        print(f"Visualization complete. Plot saved to {plot_filename}")
        # plt.show() # This might be problematic in some environments; replaced by savefig.

    except ImportError:
        print("Matplotlib not found. Skipping visualization.")
    except Exception as e:
        print(f"An error occurred during visualization: {e}")

    print("\n--- Test Block Finished ---")
