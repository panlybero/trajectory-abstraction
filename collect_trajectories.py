import gym
from env import GridWorldEnv
import numpy as np
from hardcoded_policy import HardcodedPolicy

import pickle as pkl


if __name__ == '__main__':

    trajectories = []
    # Reset the environment
    crafting_goals = ['decoration', 'stick']
    actions = ['up', 'down', 'left', 'right', 'craft_planks',
               'craft_chair_parts', 'craft_chair', 'craft_decoration', 'craft_stick']
    for ep in range(100):
        # Create the GridWorld environment
        crafting_goal = crafting_goals[0] if ep < 50 else crafting_goals[1]
        env = GridWorldEnv(n=10, crafting_goal=crafting_goal)
        # Create an instance of the HardcodedPolicy
        policy = HardcodedPolicy(n=10, crafting_goal=crafting_goal)
        obs = env.reset()
        policy.reset()
        done = False
        total_reward = 0
        trajectory = []
        prev_step = (None, None, None)
        while not done:

            # env.render_ascii()
            # print(f"Inventory: {obs['inventory']}")
            # print(f"Distance to wood: {obs['distance_to_wood']}")

            # Get the action from the HardcodedPolicy
            action = policy.act(obs['distance_to_wood'], obs['inventory'])
            # print(actions[action])
            # Take a step in the environment

            new_obs, reward, done, _ = env.step(action)

            total_reward += reward
            trajectory.append(
                (prev_step[0], prev_step[1], prev_step[2], obs, action))
            prev_step = (obs, action, reward)
            obs = new_obs

        trajectory.append(
            (prev_step[0], prev_step[1], prev_step[2], obs, None))

        print(f"Total reward: {total_reward}")
        trajectories.append(trajectory[1:])

    print(trajectories[0][-1])
    pkl.dump(trajectories, open(
        f'results/trajectories_{"_or_".join(crafting_goals)}.pkl', 'wb'))
