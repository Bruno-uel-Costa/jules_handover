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

    def select_action(self, state):
        # Helper method to select an action based on the current state
        # This typically involves passing state through the actor network
        # and then sampling from the resulting distribution.
        # This will be implemented later if needed by other subtasks.
        pass

    def evaluate_state(self, state):
        # Helper method to get the value of a state from the critic network.
        # This will be implemented later if needed by other subtasks.
        pass

def calculate_reward(P_B, P_F, U_nonGC, N_GC) -> float:
    """
    Calculates the reward based on the given parameters.
    For now, this function returns a placeholder value.
    """
    # P_B: Power of B_used
    # P_F: Power of F_used
    # U_nonGC: Utility of non-GC user
    # N_GC: Number of GC users
    return 0.0

if __name__ == "__main__":
    # Define state and action dimensions
    state_dim = 5
    action_dim = 3

    # Instantiate the agent
    agent = ActorCriticAgent(state_dim, action_dim)
    print("ActorCriticAgent instantiated.")

    # Create a dummy state tensor
    # The dummy state should be a batch of 1, with state_dim features.
    # So, its shape should be (1, state_dim) or just (state_dim) if the network can handle it.
    # Given nn.Linear expects [batch_size, num_features], let's use (state_dim) and it will be treated as a single batch.
    dummy_state = torch.randn(state_dim)
    print(f"\nDummy state (shape: {dummy_state.shape}):\n{dummy_state}")

    # Pass the dummy state through the Actor network
    # Ensure the network is in evaluation mode if dropout or batchnorm were present (not the case here)
    # agent.actor.eval() 
    action_probabilities = agent.actor(dummy_state)
    print(f"\nOutput from Actor network (action probabilities) (shape: {action_probabilities.shape}):\n{action_probabilities}")

    # Pass the dummy state through the Critic network
    # agent.critic.eval()
    state_value = agent.critic(dummy_state)
    print(f"\nOutput from Critic network (state value) (shape: {state_value.shape}):\n{state_value}")

    # Test calculate_reward function
    reward = calculate_reward(P_B=10.0, P_F=5.0, U_nonGC=100.0, N_GC=5)
    print(f"\nOutput from calculate_reward function: {reward}")

    print("\nTest block finished.")
