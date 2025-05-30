import unittest
import torch
import numpy as np
import os

# If intelligence_modules and test_rl_environment are in the same directory as this test file,
# direct imports should work when running tests (e.g., via 'python -m unittest discover' from root,
# or 'python test_intelligence_modules.py' from root).
from intelligence_modules import ActorCriticAgent
from test_rl_environment import SimpleGuardChannelEnv

class TestActorCriticLearning(unittest.TestCase):

    def test_agent_learns_in_mock_env(self):
        # Configuration
        run_on_gpu_if_available = True
        num_episodes_test = 300
        log_interval = 50
        reward_improvement_threshold = 0.5

        if run_on_gpu_if_available and torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")

        print(f"Using device: {device} for test_agent_learns_in_mock_env")

        env = SimpleGuardChannelEnv(max_gc=5, max_traffic=10, max_steps_per_episode=50)

        state_dim = env.state_dim
        action_dim = env.action_dim

        agent = ActorCriticAgent(
            state_dim,
            action_dim,
            gamma=0.98,
            gae_lambda=0.90,
            entropy_coefficient=0.01,
            actor_lr=5e-4,
            critic_lr=1e-3
        )
        agent.to(device)

        episode_rewards = []
        avg_rewards_history_for_assertion = []

        print(f"Starting training test for {num_episodes_test} episodes...")

        for episode in range(num_episodes_test):
            state_np = env.reset()
            current_episode_reward = 0

            agent.actor.eval()
            agent.critic.eval()

            for step_num in range(env.max_steps_per_episode):
                action, log_prob_tensor = agent.select_action(state_np, device=device)
                next_state_np, reward, done, _ = env.step(action)

                # Store states as lists if np arrays cause issues with tensor conversion later, though np arrays should be fine.
                # The ActorCriticAgent.learn method converts lists of states (which can be np arrays or lists) to np.array then to tensor.
                agent.store_experience(state_np, action, reward, next_state_np, done, log_prob_tensor)

                state_np = next_state_np
                current_episode_reward += reward

                if done:
                    break

            episode_rewards.append(current_episode_reward)

            if agent.trajectory_states:
                actor_loss, critic_loss = agent.learn(device=device)

                if (episode + 1) % log_interval == 0 and actor_loss is not None:
                    avg_r = np.mean(episode_rewards[-log_interval:])
                    avg_rewards_history_for_assertion.append(avg_r)
                    print(f"Episode {episode+1}/{num_episodes_test}, Avg Reward (last {log_interval}): {avg_r:.2f}, "
                          f"Actor Loss: {actor_loss:.4f}, Critic Loss: {critic_loss:.4f}")

            agent.actor.eval()
            agent.critic.eval()

        print("Training test finished.")
        self.assertIsNotNone(agent, "Agent should be instantiated.")

        if len(avg_rewards_history_for_assertion) >= 2:
            initial_avg_reward_period = avg_rewards_history_for_assertion[0]
            final_avg_reward_period = avg_rewards_history_for_assertion[-1]
            print(f"Initial avg reward period (first {log_interval} eps batch): {initial_avg_reward_period:.2f}")
            print(f"Final avg reward period (last {log_interval} eps batch): {final_avg_reward_period:.2f}")

            self.assertTrue(final_avg_reward_period > initial_avg_reward_period + reward_improvement_threshold,
                            f"Agent did not show significant learning. "
                            f"Initial avg reward period: {initial_avg_reward_period:.2f}, "
                            f"Final avg reward period: {final_avg_reward_period:.2f}. "
                            f"Expected improvement: > {reward_improvement_threshold}")
        elif num_episodes_test > 0 and len(episode_rewards) > 0:
            msg_part1 = f"Not enough distinct '{log_interval}-episode average reward' periods to compare robustly "
            msg_part2 = f"(need at least 2, got {len(avg_rewards_history_for_assertion)}). "
            msg_part3 = "Consider increasing num_episodes_test or decreasing log_interval if this test fails. "
            msg_part4 = f"Overall average reward: {np.mean(episode_rewards):.2f}"
            print(msg_part1 + msg_part2 + msg_part3 + msg_part4)
             self.assertTrue(len(episode_rewards) == num_episodes_test, "Training did not complete all episodes.")
        else:
            self.fail("Training loop did not produce enough data or did not run.")

if __name__ == '__main__':
    unittest.main()
