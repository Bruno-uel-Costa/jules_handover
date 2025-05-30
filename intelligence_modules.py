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

# +-------------------------------------------------------------------------------------+
# | SimpleGuardChannelEnv for testing RL agent learning                               |
# +-------------------------------------------------------------------------------------+
class SimpleGuardChannelEnv:
    def __init__(self, max_gc=5, max_traffic=10.0, max_steps_per_episode=50):
        self.N_GC_min = 0
        self.N_GC_max = int(max_gc)
        self.max_traffic = float(max_traffic)

        # State: [current_N_GC_normalized, current_traffic_normalized]
        self.state_dim = 2
        # Action: Delta_NGC -> maps to -1, 0, +1
        self.action_dim = 3

        self.current_N_GC = 0
        self.current_traffic_level = 0.0 # Actual traffic level

        self.current_step = 0
        self.max_steps_per_episode = int(max_steps_per_episode)

        # Action mapping: 0 -> -1, 1 -> 0, 2 -> +1 GC change
        self.action_to_delta_gc = {0: -1, 1: 0, 2: 1}

    def _normalize_state(self):
        # Normalize N_GC to [0, 1]
        norm_gc = self.current_N_GC / self.N_GC_max if self.N_GC_max > 0 else 0.0
        # Normalize traffic to [0, 1]
        norm_traffic = self.current_traffic_level / self.max_traffic if self.max_traffic > 0 else 0.0
        return np.array([norm_gc, norm_traffic], dtype=np.float32)

    def reset(self):
        # Initial N_GC can be halfway or random within a range
        self.current_N_GC = self.N_GC_max // 2
        # Initial traffic level can be random
        self.current_traffic_level = np.random.uniform(0, self.max_traffic * 0.75) # Start with moderate traffic
        self.current_step = 0
        return self._normalize_state()

    def step(self, action_idx):
        if not isinstance(action_idx, int):
            # If action_idx is a tensor or numpy int, convert to Python int
            action_idx = int(action_idx)

        if action_idx not in self.action_to_delta_gc:
            raise ValueError(f"Invalid action_idx: {action_idx}. Must be in {list(self.action_to_delta_gc.keys())}")

        delta_N_GC = self.action_to_delta_gc[action_idx]
        self.current_N_GC = np.clip(self.current_N_GC + delta_N_GC, self.N_GC_min, self.N_GC_max)

        # Reward logic:
        # Ideal N_GC is proportional to traffic.
        if self.max_traffic > 0 and self.N_GC_max > 0:
            # Target N_GC aims to be a fraction of N_GC_max, scaled by traffic intensity
            target_N_GC = round((self.current_traffic_level / self.max_traffic) * self.N_GC_max)
        else:
            target_N_GC = 0

        reward = -abs(self.current_N_GC - target_N_GC) # Primary reward: closeness to target
        reward -= 0.05 * (self.current_N_GC / self.N_GC_max if self.N_GC_max > 0 else 0) # Small penalty for N_GC usage

        if self.current_N_GC == target_N_GC:
            reward += 0.5 # Bonus for being at target

        # Simulate traffic change for next state
        # Traffic can fluctuate, e.g., small random walk
        traffic_change_factor = np.random.uniform(-0.15, 0.15) # Traffic changes by up to 15% of max_traffic
        self.current_traffic_level = np.clip(self.current_traffic_level + (traffic_change_factor * self.max_traffic),
                                             0, self.max_traffic)

        self.current_step += 1
        done = self.current_step >= self.max_steps_per_episode

        next_state_normalized = self._normalize_state()
        info = {} # Placeholder for additional info

        return next_state_normalized, float(reward), bool(done), info

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
    # --- 1. Configurações ---
    run_on_gpu_if_available = True
    num_episodes_train = 300  # Number of episodes for the learning test (was 500, 300 for faster test)
    max_steps_per_episode_env = 50
    log_interval_train = 50

    # Agent Hyperparameters (example values, can be tuned)
    config_gamma = 0.98
    config_gae_lambda = 0.90
    config_entropy_coefficient = 0.01
    config_actor_lr = 5e-4
    config_critic_lr = 1e-3

    if run_on_gpu_if_available and torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    print(f"--- Teste de Aprendizado do Agente RL (Execução Direta) ---")
    print(f"Usando dispositivo: {device}")

    # --- 2. Inicializar Ambiente e Agente ---
    env = SimpleGuardChannelEnv(max_gc=5, max_traffic=10, max_steps_per_episode=max_steps_per_episode_env)
    state_dim = env.state_dim
    action_dim = env.action_dim

    agent = ActorCriticAgent(
        state_dim,
        action_dim,
        gamma=config_gamma,
        gae_lambda=config_gae_lambda,
        entropy_coefficient=config_entropy_coefficient,
        actor_lr=config_actor_lr,
        critic_lr=config_critic_lr
    )
    agent.to(device) # Move agent's networks to the selected device

    print(f"Ambiente Simples: state_dim={state_dim}, action_dim={action_dim}")
    print(f"Agente ActorCritic instanciado com: gamma={config_gamma}, actor_lr={config_actor_lr}, critic_lr={config_critic_lr}")

    # --- 3. Loop de Treinamento e Avaliação ---
    all_episode_rewards = []
    avg_rewards_history = [] # Stores average rewards for period defined by log_interval_train

    print(f"
Iniciando loop de treinamento para {num_episodes_train} episódios...")
    for episode in range(num_episodes_train):
        state_np = env.reset() # Returns a NumPy array
        current_episode_reward = 0.0

        # Set agent to evaluation mode for action selection during experience collection
        agent.actor.eval()
        agent.critic.eval()

        for step_num in range(env.max_steps_per_episode):
            # state_np is a NumPy array, select_action can handle it
            action, log_prob_tensor = agent.select_action(state_np, device=device)

            next_state_np, reward, done, _ = env.step(action) # env.step also returns NumPy array for state

            # store_experience expects state and next_state as NumPy arrays or Python lists.
            # log_prob_tensor is already a tensor.
            agent.store_experience(state_np, action, reward, next_state_np, done, log_prob_tensor)

            state_np = next_state_np
            current_episode_reward += reward

            if done:
                break

        all_episode_rewards.append(current_episode_reward)

        # Call learn() after each episode if experiences were collected
        # The learn() method handles its own train/eval mode switching internally.
        if agent.trajectory_states: # Check if there's anything to learn from
            actor_loss_val, critic_loss_val = agent.learn(device=device)
        else:
            actor_loss_val, critic_loss_val = None, None # No learning if no trajectory

        if (episode + 1) % log_interval_train == 0:
            # Calculate average reward for the last 'log_interval_train' episodes
            avg_r = np.mean(all_episode_rewards[-log_interval_train:])
            avg_rewards_history.append(avg_r)
            print(f"Episódio {episode+1}/{num_episodes_train}, Recompensa Média (últ. {log_interval_train}): {avg_r:.2f}", end="")
            if actor_loss_val is not None:
                print(f", Perda Ator: {actor_loss_val:.4f}, Perda Crítico: {critic_loss_val:.4f}")
            else:
                print(" (Nenhuma aprendizagem nesta etapa pois não havia trajetória)")

    print("
--- Treinamento/Teste de Aprendizado Concluído ---")

    # --- 4. Verificação de Aprendizado ---
    if len(avg_rewards_history) >= 2: # Need at least two periods to compare
        initial_avg_reward = avg_rewards_history[0]
        final_avg_reward = avg_rewards_history[-1]
        print(f"Recompensa média do primeiro período ({log_interval_train} eps): {initial_avg_reward:.2f}")
        print(f"Recompensa média do último período ({log_interval_train} eps): {final_avg_reward:.2f}")

        # A simple check for improvement
        if final_avg_reward > initial_avg_reward + 0.1: # Expect at least a small improvement
            print("VERIFICAÇÃO: Agente demonstrou melhora na recompensa média.")
        else:
            print("AVISO: Agente NÃO demonstrou melhora significativa na recompensa média durante este teste.")
    elif num_episodes_train > 0 and len(all_episode_rewards) > 0:
         print(f"Treinamento com número de episódios ({num_episodes_train}) ou log_interval ({log_interval_train}) "
               f"insuficiente para comparar múltiplos períodos de recompensa média robustamente.")
         print(f"Recompensa média geral em todos os episódios: {np.mean(all_episode_rewards):.2f}")
    else:
        print("Nenhum dado de recompensa foi coletado durante o treinamento.")

    # --- 5. Teste da Função de Recompensa (mantido do original) ---
    print("
--- Testando calculate_reward ---")
    # W_B, W_F, W_U_NON_GC, W_N_GC are assumed to be global constants defined at the top of the file.

    P_B_sample = 0.05
    P_F_sample = 0.01
    U_nonGC_sample = 0.7
    N_GC_sample = 2

    reward_val = calculate_reward(P_B_sample, P_F_sample, U_nonGC_sample, N_GC_sample)

    expected_reward = -(W_B * P_B_sample) - (W_F * P_F_sample) +                       (W_U_NON_GC * U_nonGC_sample) - (W_N_GC * N_GC_sample)

    print(f"P_B={P_B_sample}, P_F={P_F_sample}, U_nonGC={U_nonGC_sample}, N_GC={N_GC_sample}")
    print(f"Recompensa Calculada: {reward_val:.4f}")
    print(f"Recompensa Esperada: {expected_reward:.4f}")
    if abs(reward_val - expected_reward) < 1e-5:
        print("Teste calculate_reward: SUCESSO")
    else:
        print("Teste calculate_reward: FALHOU")

    print("
--- Bloco de Teste Principal Concluído ---")
