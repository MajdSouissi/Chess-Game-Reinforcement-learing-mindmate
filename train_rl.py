import gymnasium as gym
from pettingzoo.classic import chess_v6
from stable_baselines3 import PPO
from stable_baselines3.ppo import MlpPolicy
import numpy as np
import os


# Create a custom environment wrapper
class ChessEnv(gym.Env):
    def __init__(self):
        super(ChessEnv, self).__init__()

        # Initialize the PettingZoo chess environment
        self.env = chess_v6.env()
        self.env.reset()

        # Get agent IDs
        self.agents = self.env.possible_agents
        self.current_agent_idx = 0

        # Define action and observation spaces
        # Chess has around 4672 possible moves, but let's simplify for now
        self.action_space = gym.spaces.Discrete(4672)

        # Define a simplified observation space
        # 8x8 board, 12 piece types + empty = 13 possible states per square
        self.observation_space = gym.spaces.Box(
            low=-1,
            high=1,
            shape=(8 * 8 * 13,),  # Flattened board representation
            dtype=np.float32
        )

    def reset(self, **kwargs):
        self.env.reset()
        self.agents = self.env.possible_agents  # Update agent list
        self.current_agent_idx = 0

        if len(self.agents) > 0:
            obs = self._get_observation()
            return obs, {}
        else:
            # Default observation if no agents
            return np.zeros(self.observation_space.shape, dtype=np.float32), {}

    def step(self, action):
        # Safety check for agent index
        if self.current_agent_idx >= len(self.agents):
            self.current_agent_idx = 0

        agent = self.agents[self.current_agent_idx]

        # Execute the action
        try:
            self.env.step(action)
        except Exception as e:
            print(f"Error executing action: {e}")
            # Return a default observation and a negative reward
            return (
                np.zeros(self.observation_space.shape, dtype=np.float32),
                -1.0,
                True,  # done
                True,  # truncated
                {}  # info
            )

        # Check if the game is done before updating agent index
        if len(self.env.agents) == 0 or all(self.env.terminations.values()) or all(self.env.truncations.values()):
            # Game is over
            obs = self._get_observation()
            reward = self.env.rewards.get(agent, 0)
            done = True
            return obs, reward, done, done, {}

        # Update agent index
        self.current_agent_idx = (self.current_agent_idx + 1) % len(self.agents)

        # Get next agent
        agent = self.agents[self.current_agent_idx]

        # Get observation and reward
        obs = self._get_observation()
        reward = self.env.rewards.get(agent, 0)

        # Check if the game is done
        done = self.env.terminations.get(agent, False) or self.env.truncations.get(agent, False)

        return obs, reward, done, done, {}

    def _get_observation(self):
        """Get a simplified observation that's compatible with our observation space"""
        # For simplicity, return a zero array of the appropriate shape
        # In a real implementation, you'd extract the actual board state
        return np.zeros(self.observation_space.shape, dtype=np.float32)


# Create and train function
def train_model(timesteps=10000):
    # Create the custom environment
    env = ChessEnv()

    print("Observation space:", env.observation_space)
    print("Action space:", env.action_space)

    # Initialize the model
    model = PPO(MlpPolicy, env, verbose=1)

    try:
        # Train the model
        model.learn(total_timesteps=timesteps)
        print("Training completed successfully!")
        return model
    except Exception as e:
        print(f"Error during training: {e}")
        return None


# Main function
def main():
    # Train the model
    model = train_model(timesteps=10000)

    if model is not None:
        try:
            # Try to save the model
            model_path = "chess_rl_model"
            # Save model parameters without the environment
            model.policy.save(f"{model_path}_policy")
            print(f"Model policy saved successfully to {model_path}_policy")
        except Exception as e:
            print(f"Error saving model: {e}")

            # Alternative saving approach
            try:
                # Save the model parameters as numpy arrays
                import numpy as np
                params = {}
                for name, param in model.policy.state_dict().items():
                    params[name] = param.detach().cpu().numpy()
                np.savez(f"{model_path}_params.npz", **params)
                print(f"Model parameters saved as numpy arrays to {model_path}_params.npz")
            except Exception as nested_e:
                print(f"Error saving parameters as numpy arrays: {nested_e}")


if __name__ == "__main__":
    main()