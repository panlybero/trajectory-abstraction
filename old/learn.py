import gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
from env import GridWorldEnv
import torch


if __name__=='__main__':
    # Set the device to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Number of parallel environments
    num_envs =12

    # Create the GridWorldEnv environment function
    def make_env():
        return GridWorldEnv(5)

    # Create the vectorized environment
    env = SubprocVecEnv([make_env] * num_envs)

    # Define and initialize the PPO agent
    model = PPO("MultiInputPolicy", env, device=device, verbose=1)
    model.load("gridworld_policy")
    model.learn(total_timesteps=100000)

    # Save the trained policy
    model.save("gridworld_policy")

    # Evaluate the trained agent
    mean_reward, _ = evaluate_policy(model, env, n_eval_episodes=10,deterministic=False)
    print(f"Mean reward: {mean_reward}")
