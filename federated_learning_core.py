import torch
import torch.nn as nn
import torch.optim as optim
import copy

# Assuming traffic_forecaster.py is in the same directory or accessible in PYTHONPATH
try:
    from traffic_forecaster import TrafficForecasterNet, generate_sine_wave_data
except ImportError as e: # Catch the specific error for better debugging
    print(f"ERROR: Could not import from traffic_forecaster: {e}")
    print("Ensure traffic_forecaster.py is in the same directory or accessible via PYTHONPATH.")
    # Define placeholder classes/functions if the import fails
    # This allows the rest of the FL structure to be initially laid out.
    class TrafficForecasterNetPlaceholder(nn.Module):
        def __init__(self, *args, **kwargs): # Accept generic args for robustness
            super().__init__()
            print("Warning: Using TrafficForecasterNetPlaceholder due to import error.")
            # Determine input_sequence_length and num_features_per_step from args or kwargs
            input_sequence_length = kwargs.get('input_sequence_length', 10)
            num_features_per_step = kwargs.get('num_features_per_step', 1)
            output_sequence_length = kwargs.get('output_sequence_length', 1)

            # Simplified structure
            self.flatten = nn.Flatten()
            flattened_input_dim = input_sequence_length * num_features_per_step
            self.dummy_layer = nn.Linear(flattened_input_dim, output_sequence_length)

        def forward(self, x):
            # Check if input x is 3D (batch, seq_len, features) as TrafficForecasterNet might expect
            if len(x.shape) == 3 and hasattr(self, 'flatten'):
                 x = self.flatten(x)
            return self.dummy_layer(x)


    def generate_sine_wave_data_placeholder(*args, **kwargs):
        print("Warning: Using generate_sine_wave_data_placeholder due to import error.")
        sequence_length = kwargs.get('sequence_length', 10)
        predict_steps = kwargs.get('predict_steps', 1)
        # Return dummy tensors that mimic the expected output structure
        # X: (num_samples, sequence_length, num_features=1)
        # Y: (num_samples, output_sequence_length)
        # Make num_samples slightly more realistic, e.g., 50
        num_samples = 50
        return torch.randn(num_samples, sequence_length, 1), torch.randn(num_samples, predict_steps), None, None

    TrafficForecasterNet = TrafficForecasterNetPlaceholder
    generate_sine_wave_data = generate_sine_wave_data_placeholder

# FLClient Class Definition
class FLClient:
    def __init__(self, client_id, model_template, learning_rate, criterion_class, device):
        self.client_id = client_id
        self.device = device

        if model_template is not None and isinstance(model_template, nn.Module):
            self.local_model = copy.deepcopy(model_template).to(self.device)
        else:
            print(f"ERROR: Client {self.client_id}: model_template is not a valid nn.Module or is None. Cannot initialize local_model.")
            self.local_model = None

        if self.local_model:
            self.optimizer = optim.Adam(self.local_model.parameters(), lr=learning_rate)
        else:
            self.optimizer = None

        self.criterion = criterion_class() # Instantiated criterion
        self.local_data_X = None
        self.local_data_Y = None
        # Reduced verbosity at client init, more info when data/weights are set.
        # print(f"Client {self.client_id}: Initialized on device {self.device}.")


    def set_local_data(self, data_x, data_y):
        if data_x is not None and data_y is not None:
            self.local_data_X = data_x.to(self.device)
            self.local_data_Y = data_y.to(self.device)
            # print(f"Client {self.client_id}: Local data set. X shape: {self.local_data_X.shape}, Y shape: {self.local_data_Y.shape}")
        else:
            print(f"Client {self.client_id}: Warning - received None for local data.")
            self.local_data_X = None
            self.local_data_Y = None


    def set_global_model_weights(self, global_weights):
        if self.local_model and global_weights:
            try:
                self.local_model.load_state_dict(global_weights)
                # print(f"Client {self.client_id}: Loaded global model weights successfully.")
            except RuntimeError as e:
                print(f"Client {self.client_id}: Error loading global model weights. Ensure model architectures match. Details: {e}")
            except Exception as e:
                print(f"Client {self.client_id}: An unexpected error occurred while loading global model weights: {e}")
        else:
            if not self.local_model:
                 print(f"Client {self.client_id}: Cannot set global model weights because local_model is not initialized.")
            if not global_weights:
                 print(f"Client {self.client_id}: Received empty or None global_weights.")


    def train_local_epoch(self, num_epochs_local):
        if self.local_model is None:
            print(f"Client {self.client_id}: Cannot train, local_model not initialized.")
            return None
        if self.local_data_X is None or self.local_data_Y is None:
            print(f"Client {self.client_id}: Local data not available. Skipping local training.")
            return None

        # print(f"Client {self.client_id}: Starting local training for {num_epochs_local} epochs.")
        self.local_model.train()

        final_loss = None
        for epoch in range(num_epochs_local):
            self.optimizer.zero_grad()
            predictions = self.local_model(self.local_data_X)
            loss = self.criterion(predictions, self.local_data_Y)
            loss.backward()
            self.optimizer.step()
            # Reduced verbosity for local epochs
            # if (epoch + 1) % 5 == 0:
            #     print(f"Client {self.client_id}: Local Epoch [{epoch+1}/{num_epochs_local}], Loss: {loss.item():.6f}")
            final_loss = loss.item()

        # if final_loss is not None:
        #     print(f"Client {self.client_id}: Local training finished. Final local loss: {final_loss:.6f}")
        # else:
        #     print(f"Client {self.client_id}: Local training did not run for any epochs (num_epochs_local might be 0).")
        return final_loss


    def get_local_model_updates(self):
        if self.local_model is None:
            print(f"Client {self.client_id}: Cannot get updates, local_model not initialized.")
            return None
        # print(f"Client {self.client_id}: Providing local model updates (state_dict).")
        return copy.deepcopy(self.local_model.state_dict())

# ---- CONCEPTUAL INTEGRATION OF ACTOR-CRITIC AGENT TRAINING IN FLCLIENT ----
#
# The following outlines how the `ActorCriticAgent`'s training mechanism, specifically
# its `learn()` method and trajectory collection, would integrate into a
# Federated Learning (FL) client's local training epoch (e.g., `FLClient.train_local_epoch()`).
#
# 1. Model Synchronization:
#    - The `FLClient` receives the latest global model weights (for both Actor and Critic networks)
#      from the FL server.
#    - The client updates its local `ActorCriticAgent` instance with these weights
#      (e.g., using `agent.actor.load_state_dict()` and `agent.critic.load_state_dict()`).
#
# 2. Local Training Loop (`FLClient.train_local_epoch()`):
#    - This method would typically run for a defined number of local episodes or simulation steps.
#    - Inside the loop:
#      a. Episode/Interaction Phase:
#         - Reset the simulation environment (`simulation_core.py` or its interface) to get an initial `state`.
#         - For each step in an episode (or for a fixed number of steps, `N_A2C_steps`):
#           i.   Action Selection: The local `ActorCriticAgent` selects an `action` and `log_prob`
#                based on the current `state` using `agent.select_action(state)`. The agent should
#                be in `eval()` mode during interaction.
#           ii.  Environment Step: The chosen `action` is applied to the `simulation_core.py`,
#                which returns the `next_state`, `reward`, and `done` status.
#           iii. Store Experience: The collected tuple `(state, action, reward, next_state, done, log_prob)`
#                is stored in the agent's internal on-policy trajectory buffer via
#                `agent.store_experience(...)`.
#           iv.  State Update: `state = next_state`.
#           v.   If `done` or if the number of collected experiences reaches a threshold suitable for
#                an A2C update (e.g., `N_A2C_steps`), proceed to the learning phase.
#
#      b. Learning Phase:
#         - If sufficient experiences have been collected in `agent.trajectory_states`:
#           i.  Call `actor_loss, critic_loss = agent.learn()`. This method handles:
#               - Switching the agent to `train()` mode.
#               - Calculating advantages (GAE) and returns.
#               - Computing actor and critic losses.
#               - Performing gradient updates on the local actor and critic networks.
#               - Clearing the agent's trajectory buffer.
#               - Switching the agent back to `eval()` mode.
#         - This learning phase can occur at the end of each episode or after a fixed
#           number of interaction steps (common in A2C).
#
# 3. Result Aggregation:
#    - After the `train_local_epoch()` has completed (e.g., after a certain number of
#      local training iterations/episodes or a compute budget is exhausted):
#    - The updated weights from the local `agent.actor.state_dict()` and
#      `agent.critic.state_dict()` are extracted.
#    - These weights (or weight differences/gradients, depending on the FL strategy) are
#      then sent back to the FL server for aggregation.
#
# Key Considerations for FL Integration:
#    - On-Policy Nature of A2C: Since A2C is on-policy, the experiences collected by a client
#      are specific to its current local policy. This aligns well with FL where clients
#      train on their local data (experiences).
#    - Data Heterogeneity: Different clients might experience different state distributions
#      from their `simulation_core.py` interactions, leading to heterogeneous experience data.
#      This is a standard challenge in FL.
#    - Communication Costs: Sending full model weights can be costly. Strategies like
#      sending weight deltas or using more advanced aggregation methods might be considered.
#
# ---- END OF CONCEPTUAL INTEGRATION NOTES ----

# FLServer Class Definition
class FLServer:
    def __init__(self, global_model_instance, device):
        self.device = device
        if global_model_instance is not None and isinstance(global_model_instance, nn.Module):
            self.global_model = global_model_instance.to(self.device)
            # print(f"FL Server: Initialized with global model on device '{self.device}'.")
        else:
            print("ERROR: FL Server: global_model_instance is not a valid nn.Module or is None. Cannot initialize global_model.")
            self.global_model = None

    def get_global_model_weights(self):
        if self.global_model is None:
            print("FL Server: Cannot get weights, global_model not initialized.")
            return None
        # print("FL Server: Providing global model weights.")
        return copy.deepcopy(self.global_model.state_dict())

    def aggregate_model_updates(self, client_updates_list):
        if self.global_model is None:
            print("FL Server: Cannot aggregate updates, global_model not initialized.")
            return

        if not client_updates_list:
            print("FL Server: Received no client updates to aggregate. Skipping aggregation.")
            return

        # print(f"FL Server: Received updates from {len(client_updates_list)} clients. Starting aggregation (FedAvg).")

        aggregated_weights = copy.deepcopy(self.global_model.state_dict())

        for key in aggregated_weights:
            # Ensure aggregated_weights are initialized with the correct dtype and device from the global model
            aggregated_weights[key] = torch.zeros_like(self.global_model.state_dict()[key], dtype=self.global_model.state_dict()[key].dtype, device=self.device)

        num_clients_with_valid_updates = 0
        for client_state_dict in client_updates_list:
            if not client_state_dict:
                print("FL Server: Warning - received an empty update from one client. Skipping this update.")
                continue

            num_clients_with_valid_updates +=1
            for key in aggregated_weights:
                if key in client_state_dict:
                    aggregated_weights[key] += client_state_dict[key].to(self.device, dtype=aggregated_weights[key].dtype)
                else:
                    print(f"FL Server: Warning - key '{key}' not found in one of the client updates. Skipping for this client on this key.")

        if num_clients_with_valid_updates > 0:
            for key in aggregated_weights:
                aggregated_weights[key] /= num_clients_with_valid_updates
        else:
            print("FL Server: No valid client updates found to average. Global model remains unchanged.")
            return

        try:
            self.global_model.load_state_dict(aggregated_weights)
            # print("FL Server: Aggregated client updates using FedAvg. Global model updated.")
        except RuntimeError as e:
            print(f"FL Server: Error loading aggregated weights into global model. Details: {e}")
        except Exception as e:
            print(f"FL Server: An unexpected error occurred while loading aggregated weights: {e}")


if __name__ == '__main__':
    # --- Configuration ---
    # KAN Parameters for TrafficForecasterNet
    input_seq_len = 20
    features_per_step = 1
    output_seq_len = 1 # Predicting 1 step ahead
    hidden_dim_kan1 = 32
    grid_size_kan = 5
    spline_order_kan = 3
    normalization_target_range = (0.0, 1.0) # Consistent with KAN's expected input range

    # Federated Learning Parameters
    num_clients = 3
    num_rounds = 5 # Number of federated training rounds
    local_epochs_per_round = 10 # Number of local training epochs on each client per round
    learning_rate_client = 0.001

    # Device Configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--- Running on device: {device} ---")

    # --- Data Generation and Distribution ---
    print("--- Generating and distributing synthetic data ---")
    # Generate data once
    # Using slightly more data points to ensure enough for multiple clients
    # Adjusted total_num_points logic slightly for clarity if generate_sine_wave_data is robust to small num_points
    min_points_per_client = 50 # Ensure each client has at least this many after sequence creation
    required_data_length_per_client = input_seq_len + output_seq_len + min_points_per_client
    total_num_points = required_data_length_per_client * num_clients

    X_global, Y_global, _, _ = generate_sine_wave_data(
        num_points=total_num_points,
        sequence_length=input_seq_len,
        predict_steps=output_seq_len,
        noise_level=0.05,
        normalization_range=normalization_target_range
    )

    # Simple data distribution: split the data among clients
    # This is a basic split; real FL would have naturally partitioned data.
    if X_global.shape[0] < num_clients : # Check if enough samples were generated for chunking
        print(f"Error: Not enough data samples ({X_global.shape[0]}) generated to distribute among {num_clients} clients.")
        print("Consider increasing num_points in generate_sine_wave_data or reducing num_clients.")
        exit()

    client_data_X = torch.chunk(X_global, num_clients, dim=0)
    client_data_Y = torch.chunk(Y_global, num_clients, dim=0)

    print(f"Total X shape: {X_global.shape}, Y shape: {Y_global.shape}")
    for i in range(num_clients):
        print(f"Client {i} data: X shape {client_data_X[i].shape}, Y shape {client_data_Y[i].shape}")

    # --- Model Template ---
    # Create a template instance of the KAN-based TrafficForecasterNet
    # Ensure TrafficForecasterNet is imported and working
    if TrafficForecasterNet is None or TrafficForecasterNet.__name__ == 'TrafficForecasterNetPlaceholder' or \
       generate_sine_wave_data is None or generate_sine_wave_data.__name__ == 'generate_sine_wave_data_placeholder':
        print("ERROR: Actual TrafficForecasterNet or generate_sine_wave_data not imported correctly (using placeholders). Exiting.")
        exit()

    model_template = TrafficForecasterNet(
        input_sequence_length=input_seq_len,
        num_features_per_step=features_per_step,
        output_sequence_length=output_seq_len,
        hidden_dim_kan1=hidden_dim_kan1,
        grid_size_kan=grid_size_kan,
        spline_order_kan=spline_order_kan,
        grid_range_kan=normalization_target_range
    )
    criterion_class = nn.MSELoss # Class, not instance, to be instantiated by clients

    # --- FL System Initialization ---
    print("--- Initializing FL Server and Clients ---")
    fl_server = FLServer(global_model_instance=copy.deepcopy(model_template), device=device)

    clients = []
    for i in range(num_clients):
        client = FLClient(
            client_id=f"Client-{i}", # String client ID
            model_template=model_template, # Each client gets a fresh copy from this template
            learning_rate=learning_rate_client,
            criterion_class=criterion_class,
            device=device
        )
        client.set_local_data(client_data_X[i], client_data_Y[i])
        clients.append(client)
    print(f"Initialized {len(clients)} clients and 1 FL Server.")

    # --- Federated Training Rounds ---
    print(f"\n--- Starting Federated Training for {num_rounds} rounds ---")
    for round_num in range(num_rounds):
        print(f"\n--- Round {round_num + 1}/{num_rounds} ---")

        current_global_weights = fl_server.get_global_model_weights()
        if current_global_weights is None:
            print("Halting FL: Server could not provide global weights.")
            break

        client_model_updates_for_aggregation = []
        round_client_losses = []

        for client in clients:
            # print(f"-- Processing Client {client.client_id} --")
            client.set_global_model_weights(current_global_weights)

            local_loss = client.train_local_epoch(num_epochs_local=local_epochs_per_round)
            if local_loss is not None:
                round_client_losses.append(local_loss)

            local_updates = client.get_local_model_updates()
            if local_updates:
                client_model_updates_for_aggregation.append(local_updates)
            else:
                print(f"Client {client.client_id} did not provide updates.")

        avg_client_loss = sum(round_client_losses) / len(round_client_losses) if round_client_losses else float('nan')
        print(f"Round {round_num + 1}: Average client local training loss: {avg_client_loss:.6f}")

        if client_model_updates_for_aggregation:
            fl_server.aggregate_model_updates(client_model_updates_for_aggregation)
            print(f"Round {round_num + 1}: Server aggregated updates from {len(client_model_updates_for_aggregation)} clients.")
        else:
            print(f"Round {round_num + 1}: FL Server: No updates received from clients in this round. Global model not changed.")

        # Optional: Evaluate global model performance on a test set here
        print(f"--- Round {round_num + 1} completed. ---")

    print("\n--- Federated Learning Simulation Finished ---")

    # Optional: Save the global model
    final_global_weights = fl_server.get_global_model_weights()
    if final_global_weights:
       model_save_path = "final_global_model_fedkan.pth"
       torch.save(final_global_weights, model_save_path)
       print(f"Saved final global model weights to {model_save_path}")
```
