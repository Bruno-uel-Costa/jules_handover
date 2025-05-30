# ---- STATE (S) and ACTION (A) SPACE DEFINITIONS ----
#
# === State (S) ===
# The state vector input to the ActorCriticAgent is composed of the following features,
# which should be NORMALIZED (e.g., to [0, 1] or standardized) before being fed to the agent.
#
# 1. N_GC_atual (float): Current number of guard channels in use.
#    - Normalization: E.g., N_GC_atual / N_GC_max.
# 2. ocupacao_media_non_gc (float): Average utilization of non-Guard Channels.
#    - Range: [0.0, 1.0]. Already normalized or needs normalization based on observed max.
# 3. ocupacao_media_gc (float): Average utilization of Guard Channels.
#    - Range: [0.0, 1.0]. Already normalized or needs normalization.
#    - If N_GC_atual is 0, this could be 0 or a special value.
# 4. P_B_recente (float): Recent probability of blocking new calls.
#    - Range: [0.0, 1.0].
# 5. P_F_recente (float): Recent probability of handover failure.
#    - Range: [0.0, 1.0].
# 6. previsao_trafego_futuro (float or list/array of floats):
#    - Traffic prediction(s) from traffic_forecaster.py.
#    - If a sequence, its length must be fixed.
#    - Normalization: Depends on the forecaster's output range. Should be scaled
#      similarly to other state features.
#
# The exact `state_dim` of the ActorCriticAgent will be the sum of the dimensions
# of these features (e.g., if previsao_trafego_futuro is a single value, state_dim = 6).
#
# === Action (A) ===
# The Actor network outputs a discrete action index. This index is mapped to a
# change in the number of guard channels (Delta_NGC).
#
# Let `action_dim` be the number of possible actions.
# Example: If `action_dim = 3`, actions could map to:
#   - Action 0: Delta_NGC = -1 (decrease N_GC by 1)
#   - Action 1: Delta_NGC =  0 (keep N_GC unchanged)
#   - Action 2: Delta_NGC = +1 (increase N_GC by 1)
#
# The actual number of guard channels `N_GC` will then be:
#   N_GC_novo = N_GC_atual + Delta_NGC
#
# Constraints:
#   - N_GC_novo must be kept within predefined limits [N_GC_min, N_GC_max].
#     The environment/simulation_core is responsible for clipping N_GC_novo to these limits.
#   - The choice of `action_dim` and the range of `Delta_NGC` (e.g., just -1,0,1 or larger steps)
#     are hyperparameters of the agent and system design.
#
# ---- END OF S/A DEFINITIONS ----

# Weights for reward calculation
W_B = 10.0  # Weight for Probability of Blocking New Calls
W_F = 20.0  # Weight for Probability of Handover Failure
W_U_NON_GC = 0.5 # Weight for Utilization of non-GC channels
W_N_GC = 0.1  # Weight for the number of Guard Channels

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

class ActorCriticAgent(nn.Module):
    def __init__(self, state_dim, action_dim,
                 gamma=0.99, gae_lambda=0.95, entropy_coefficient=0.01,
                 actor_lr=0.0003, critic_lr=0.001):
        super(ActorCriticAgent, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.entropy_coefficient = entropy_coefficient

        # Actor Network (Policy Network)
        self.actor = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim),
            nn.Softmax(dim=-1)
        )

        # Critic Network (Value Network)
        self.critic = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_lr)

        self.trajectory_states = []
        self.trajectory_actions = []
        self.trajectory_rewards = []
        self.trajectory_next_states = []
        self.trajectory_dones = []
        self.trajectory_log_probs = []

    def store_experience(self, state, action, reward, next_state, done, log_prob):
        self.trajectory_states.append(state)
        self.trajectory_actions.append(action)
        self.trajectory_rewards.append(reward)
        self.trajectory_next_states.append(next_state)
        self.trajectory_dones.append(done)
        self.trajectory_log_probs.append(log_prob) # log_prob is expected to be a tensor

    def clear_trajectory(self):
        self.trajectory_states.clear()
        self.trajectory_actions.clear()
        self.trajectory_rewards.clear()
        self.trajectory_next_states.clear()
        self.trajectory_dones.clear()
        self.trajectory_log_probs.clear()

    def forward(self, state):
        action_probs = self.actor(state)
        value = self.critic(state)
        return action_probs, value

    def select_action(self, state, device='cpu'):
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
        with torch.no_grad(): # Always no_grad for action selection based on current policy
            action_probs = self.actor(state_tensor)
        dist = torch.distributions.Categorical(probs=action_probs)
        action = dist.sample()
        return action.item(), dist.log_prob(action) # Return action_item and log_prob tensor

    def evaluate_state(self, state, device='cpu'):
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
        with torch.no_grad(): # Always no_grad for state evaluation
            state_value = self.critic(state_tensor)
        return state_value

    def learn(self, device='cpu'):
        if not self.trajectory_states:
            return None, None

        self.actor.train()
        self.critic.train()

        # Convert trajectory data to tensors
        # Ensure states are correctly batched (list of lists/arrays -> tensor)
        states_np = np.array(self.trajectory_states, dtype=np.float32)
        next_states_np = np.array(self.trajectory_next_states, dtype=np.float32)

        states_tensor = torch.tensor(states_np).to(device)
        actions_tensor = torch.tensor(self.trajectory_actions, dtype=torch.int64).to(device)
        rewards_tensor = torch.tensor(self.trajectory_rewards, dtype=torch.float32).to(device)
        next_states_tensor = torch.tensor(next_states_np).to(device)
        dones_tensor = torch.tensor(self.trajectory_dones, dtype=torch.float32).to(device)

        # Calculate Critic Values for GAE
        # V_s needs gradients for critic loss later, but not for delta calculation if delta is detached.
        # V_s_prime is a target, so always detach.
        V_s_eval = self.critic(states_tensor).squeeze()
        V_s_prime_eval = self.critic(next_states_tensor).squeeze().detach()

        # Calculate Advantages and Returns (for Critic target) using GAE
        num_steps = len(rewards_tensor)
        advantages = torch.zeros_like(rewards_tensor).to(device)
        last_gae_lam = 0

        for t in reversed(range(num_steps)):
            # V_s_eval[t] is used here. If V_s_eval has .grad_fn, then delta will too.
            delta = rewards_tensor[t] + self.gamma * V_s_prime_eval[t] * (1.0 - dones_tensor[t]) - V_s_eval[t]
            # Detach delta before using in recurrence for GAE to prevent complex gradient paths through time.
            advantages[t] = last_gae_lam = delta.detach() + self.gamma * self.gae_lambda * (1.0 - dones_tensor[t]) * last_gae_lam

        # This is the target for the critic's value function V(s_t)
        returns_for_critic = advantages + V_s_eval.detach() # Detach V_s_eval as it's part of target

        # Actor Loss
        # Re-evaluate actions under current policy to get log_probs and entropy
        action_probs_new = self.actor(states_tensor)
        dist_new = torch.distributions.Categorical(probs=action_probs_new)
        current_log_probs = dist_new.log_prob(actions_tensor) # actions_tensor are actions taken by old policy
        entropy = dist_new.entropy().mean()

        # Advantages are detached: they are treated as empirical estimates of advantage
        actor_loss = - (advantages.detach() * current_log_probs).mean() - self.entropy_coefficient * entropy

        # Critic Loss
        # Values are re-evaluated from states_tensor to ensure gradients flow for the critic network
        # This V_s is different from V_s_eval if critic has been updated (not in A2C single pass)
        # but for a single pass, it's the same calculation, now with intent to get gradients.
        current_state_values_for_loss = self.critic(states_tensor).squeeze()
        critic_loss = F.mse_loss(current_state_values_for_loss, returns_for_critic.detach()) # Target must be detached

        # Optimization
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        # torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_grad_norm) # Optional
        self.actor_optimizer.step()

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        # torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_grad_norm) # Optional
        self.critic_optimizer.step()

        self.clear_trajectory()

        # Set back to eval mode after training step
        self.actor.eval()
        self.critic.eval()

        return actor_loss.item(), critic_loss.item()


def calculate_reward(P_B: float, P_F: float, U_nonGC: float, N_GC: int) -> float:
    reward = -(W_B * P_B) - (W_F * P_F) + (W_U_NON_GC * U_nonGC) - (W_N_GC * N_GC)
    return reward

if __name__ == "__main__":
    state_dim_g = 5
    action_dim_g = 3
    device_g = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device_g}")

    agent_g = ActorCriticAgent(state_dim_g, action_dim_g, actor_lr=1e-4, critic_lr=3e-4, entropy_coefficient=0.005)
    agent_g.to(device_g)
    print("ActorCriticAgent instantiated.")

    # Initial mode for select_action/evaluate_state tests
    agent_g.actor.eval()
    agent_g.critic.eval()

    dummy_state_list_g = np.random.rand(state_dim_g).tolist()
    print(f"
Dummy state (list): {dummy_state_list_g}")

    # select_action returns action_item (int) and log_prob (tensor)
    action_g, log_prob_tensor_g = agent_g.select_action(dummy_state_list_g, device=device_g)
    print(f"Selected Action: {action_g}, Log Probability: {log_prob_tensor_g.item()}")

    state_value_tensor_g = agent_g.evaluate_state(dummy_state_list_g, device=device_g)
    print(f"Estimated State Value: {state_value_tensor_g.item()}")

    print("
--- Testing Learning Logic ---")
    num_dummy_steps_g = 10 # Collect a small trajectory

    for i in range(num_dummy_steps_g):
        current_state_g = np.random.rand(state_dim_g).tolist()
        # select_action needs eval mode if it's not managed inside.
        # agent_g.actor.eval() # already in eval from above
        act_g, lp_tensor_g = agent_g.select_action(current_state_g, device=device_g)

        next_st_g = np.random.rand(state_dim_g).tolist()
        rew_g = np.random.uniform(-1, 1) # More realistic reward range
        # Make 'done' true for the last step to test terminal condition handling in GAE
        dn_g = True if i == num_dummy_steps_g - 1 else False

        agent_g.store_experience(current_state_g, act_g, rew_g, next_st_g, dn_g, lp_tensor_g)

    if agent_g.trajectory_states:
        print(f"
Collected {len(agent_g.trajectory_states)} experiences. Attempting to learn...")
        # learn method handles train/eval mode switching internally
        actor_loss_g, critic_loss_g = agent_g.learn(device=device_g)
        if actor_loss_g is not None and critic_loss_g is not None:
            print(f"Learning completed. Actor Loss: {actor_loss_g:.4f}, Critic Loss: {critic_loss_g:.4f}")
        else:
            print("Learning did not run (e.g. no experiences).")
    else:
        print("No experiences collected, skipping learn test.")

    # agent.learn() sets mode to eval at the end.

    print("
Testing calculate_reward...")
    P_B_sample_g, P_F_sample_g, U_nonGC_sample_g, N_GC_sample_g = 0.05, 0.01, 0.7, 2
    reward_val_g = calculate_reward(P_B_sample_g, P_F_sample_g, U_nonGC_sample_g, N_GC_sample_g)
    expected_reward_g = -(W_B * P_B_sample_g) - (W_F * P_F_sample_g) + (W_U_NON_GC * U_nonGC_sample_g) - (W_N_GC * N_GC_sample_g)
    print(f"Calculated Reward: {reward_val_g}, Expected: {expected_reward_g}")

    print("
Test block finished.")
