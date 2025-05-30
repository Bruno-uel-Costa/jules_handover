# Weights for reward calculation
W_B = 10.0  # Weight for Probability of Blocking New Calls
W_F = 20.0  # Weight for Probability of Handover Failure
W_U_NON_GC = 0.5 # Weight for Utilization of non-GC channels
W_N_GC = 0.1  # Weight for the number of Guard Channels

import torch
import torch.nn as nn
import torch.nn.functional as F

class ActorCriticAgent(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(ActorCriticAgent, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim

        # Actor Network (Policy Network)
        self.actor = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim),
            nn.Softmax(dim=-1)  # Softmax activation for action probabilities
        )

        # Critic Network (Value Network)
        self.critic = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)  # Output is a single value, no activation
        )

    def forward(self, state):
        # This method is not strictly required by the problem description for the agent structure itself,
        # but it's good practice for nn.Module.
        # It typically defines how to get actions or values from the networks.
        # For now, let's make it return action probabilities from the actor and value from the critic.
        action_probs = self.actor(state)
        value = self.critic(state)
        return action_probs, value

    def select_action(self, state, device='cpu'):
        """
        Selects an action based on the current state using the actor network.
        Args:
            state: The current state of the environment (e.g., list or numpy array).
            device: The device to perform calculations on ('cpu' or 'cuda').
        Returns:
            A tuple containing the selected action (int) and its log probability (torch.Tensor).
        """
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
        self.actor.eval() # Set actor to evaluation mode
        with torch.no_grad(): # No gradient calculation needed for action selection
            action_probs = self.actor(state_tensor)
        dist = torch.distributions.Categorical(probs=action_probs)
        action = dist.sample()
        return action.item(), dist.log_prob(action)

    def evaluate_state(self, state, device='cpu'):
        """
        Evaluates the given state using the critic network.
        Args:
            state: The current state of the environment (e.g., list or numpy array).
            device: The device to perform calculations on ('cpu' or 'cuda').
        Returns:
            The estimated value of the state (torch.Tensor).
        """
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
        self.critic.eval() # Set critic to evaluation mode
        with torch.no_grad(): # No gradient calculation needed
            state_value = self.critic(state_tensor)
        return state_value

def calculate_reward(P_B: float, P_F: float, U_nonGC: float, N_GC: int) -> float:
    """
    Calculates the reward based on system parameters and predefined weights.
    """
    # P_B: Probability of Blocking New Calls
    # P_F: Probability of Handover Failure
    # U_nonGC: Utilization of non-GC channels
    # N_GC: Number of Guard Channels
    reward = -(W_B * P_B) - (W_F * P_F) + (W_U_NON_GC * U_nonGC) - (W_N_GC * N_GC)
    return reward

if __name__ == "__main__":
    # Define state and action dimensions
    state_dim = 5
    action_dim = 3

    # Instantiate the agent
    agent = ActorCriticAgent(state_dim, action_dim)
    print("ActorCriticAgent instantiated.")

    # Define a dummy state as a Python list
    dummy_state_list = [0.1, 0.2, 0.3, 0.4, 0.5]
    print(f"\nDummy state (list): {dummy_state_list}")

    # Test select_action method
    print("\nTesting select_action...")
    action, log_prob = agent.select_action(dummy_state_list)
    print(f"Selected Action: {action}")
    print(f"Log Probability of Action: {log_prob}")

    # Test evaluate_state method
    print("\nTesting evaluate_state...")
    state_value = agent.evaluate_state(dummy_state_list)
    print(f"Estimated State Value: {state_value}")
    # The output is a tensor like tensor([[0.0512]]), to get the float: state_value.item() or state_value[0,0].item()
    print(f"Estimated State Value (float): {state_value.item()}") 


    # Test calculate_reward function
    print("\nTesting calculate_reward...")
    P_B_sample = 0.05  # Example: 5% blocking probability
    P_F_sample = 0.01  # Example: 1% handover failure probability
    U_nonGC_sample = 0.7 # Example: 70% utilization of non-GC channels
    N_GC_sample = 2      # Example: 2 Guard Channels
    
    reward = calculate_reward(P_B_sample, P_F_sample, U_nonGC_sample, N_GC_sample)
    print(f"Calculated Reward with P_B={P_B_sample}, P_F={P_F_sample}, U_nonGC={U_nonGC_sample}, N_GC={N_GC_sample}: {reward}")
    # Expected: -(10*0.05) - (20*0.01) + (0.5*0.7) - (0.1*2)
    #         = -0.5      - 0.2       + 0.35      - 0.2
    #         = -0.7 + 0.35 - 0.2
    #         = -0.35 - 0.2
    #         = -0.55
    expected_reward = -(W_B * P_B_sample) - (W_F * P_F_sample) + (W_U_NON_GC * U_nonGC_sample) - (W_N_GC * N_GC_sample)
    print(f"Expected Reward based on constants: {expected_reward}")


    # Old direct calls (now commented out as requested, select_action and evaluate_state cover these)
    # dummy_state_tensor = torch.randn(state_dim)
    # print(f"\nDummy state tensor (shape: {dummy_state_tensor.shape}):\n{dummy_state_tensor}")
    # action_probabilities = agent.actor(dummy_state_tensor)
    # print(f"\nOutput from Actor network (action probabilities) (shape: {action_probabilities.shape}):\n{action_probabilities}")
    # state_value_direct = agent.critic(dummy_state_tensor)
    # print(f"\nOutput from Critic network (state value) (shape: {state_value_direct.shape}):\n{state_value_direct}")

    print("\nTest block finished.")
