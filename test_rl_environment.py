import numpy as np

class SimpleGuardChannelEnv:
    def __init__(self, max_gc=5, max_traffic=10, max_steps_per_episode=50):
        self.N_GC_min = 0
        self.N_GC_max = int(max_gc)
        self.max_traffic = float(max_traffic)
        self.state_dim = 2  # [current_N_GC_normalized, current_traffic_normalized]
        self.action_dim = 3 # Delta_NGC: -1, 0, +1

        self.current_N_GC = 0
        self.current_traffic_level = 0.0

        self.current_step = 0
        self.max_steps_per_episode = int(max_steps_per_episode)

        # Action mapping: 0 -> -1, 1 -> 0, 2 -> +1
        self.action_to_delta_gc = {0: -1, 1: 0, 2: 1}

    def _normalize_state(self):
        # Handles division by zero if max_traffic or N_GC_max is zero, though they shouldn't be.
        norm_gc = self.current_N_GC / self.N_GC_max if self.N_GC_max > 0 else 0.0
        norm_traffic = self.current_traffic_level / self.max_traffic if self.max_traffic > 0 else 0.0
        return np.array([norm_gc, norm_traffic], dtype=np.float32)

    def reset(self):
        self.current_N_GC = self.N_GC_max // 2
        self.current_traffic_level = np.random.uniform(0, self.max_traffic)
        self.current_step = 0
        return self._normalize_state()

    def step(self, action_idx):
        if action_idx not in self.action_to_delta_gc:
            raise ValueError(f"Invalid action_idx: {action_idx}. Must be in {list(self.action_to_delta_gc.keys())}")

        delta_N_GC = self.action_to_delta_gc[action_idx]
        self.current_N_GC = np.clip(self.current_N_GC + delta_N_GC, self.N_GC_min, self.N_GC_max)

        # Reward logic:
        # Ideal N_GC is proportional to traffic.
        # Avoid division by zero if max_traffic or N_GC_max is zero.
        if self.max_traffic > 0 and self.N_GC_max > 0:
            target_N_GC = round((self.current_traffic_level / self.max_traffic) * self.N_GC_max)
        else:
            target_N_GC = 0

        # Primary reward: negative absolute difference from target N_GC
        reward = -abs(self.current_N_GC - target_N_GC)
        # Small penalty for using GCs to encourage efficiency if multiple N_GC lead to similar primary reward
        reward -= 0.05 * self.current_N_GC
        # Bonus for being exactly at target
        if self.current_N_GC == target_N_GC:
            reward += 0.5

        # Simulate traffic change for next state
        # Traffic drifts randomly but tends to stay within bounds
        traffic_change = np.random.uniform(-self.max_traffic * 0.1, self.max_traffic * 0.1) # Change up to 10% of max_traffic
        self.current_traffic_level = np.clip(self.current_traffic_level + traffic_change, 0, self.max_traffic)

        self.current_step += 1
        done = self.current_step >= self.max_steps_per_episode

        next_state_normalized = self._normalize_state()

        # info dict - can be empty for this simple env
        info = {}

        return next_state_normalized, float(reward), bool(done), info

if __name__ == '__main__':
    # Example usage
    env = SimpleGuardChannelEnv(max_gc=5, max_traffic=10, max_steps_per_episode=20)

    print(f"State dimension: {env.state_dim}")
    print(f"Action dimension: {env.action_dim}")

    state = env.reset()
    print(f"Initial state: {state}")
    total_reward = 0

    for step in range(env.max_steps_per_episode + 5): # Run a bit beyond max_steps to see done=True
        action = np.random.randint(0, env.action_dim) # Random action
        next_state, reward, done, _ = env.step(action)
        total_reward += reward
        print(f"Step: {env.current_step}, Action: {action} (maps to {env.action_to_delta_gc[action]} GC change), "
              f"N_GC: {env.current_N_GC}, Traffic: {env.current_traffic_level:.2f}, "
              f"Next State: {next_state}, Reward: {reward:.2f}, Done: {done}")
        state = next_state
        if done:
            print(f"Episode finished after {env.current_step} steps. Total reward: {total_reward:.2f}")
            state = env.reset() # Reset for a new conceptual episode
            total_reward = 0
            print(f"New initial state after reset: {state}")

    print("Example usage finished.")
