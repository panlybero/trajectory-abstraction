import gym
from env import GridWorldEnv
import numpy as np
from hardcoded_policy import HardcodedPolicy
from PolicyFromAgentModel import PolicyFromAgentModel, PlanningPolicyFromAgentModel
from AgentModel import AgentModel

import pickle as pkl    


if __name__=='__main__':
# Create the GridWorld environment
    crafting_goal = 'decoration'
    env = GridWorldEnv(n=10, crafting_goal=crafting_goal)
    
    agent_model = pkl.load(open('agent_model_decoration.pkl', 'rb'))

    # Create an instance of the HardcodedPolicy
    policy = PolicyFromAgentModel(n=10,agent_model=agent_model)
    
    trajectories = []
    # Reset the environment
    
    actions = ['up', 'down', 'left', 'right', 'craft_planks', 'craft_chair_parts', 'craft_chair', 'craft_decoration']
    for _ in range(100):
        obs = env.reset()
        policy.reset()
        done = False
        total_reward = 0
        trajectory = []
        while not done:
            
            action = policy.act(obs['distance_to_wood'], obs['inventory'])
      
            new_obs, reward, done, _ = env.step(action)
            total_reward += reward
            trajectory.append((obs, action, reward, new_obs))
            obs = new_obs
            
    
            
        print(f"Total reward: {total_reward}")
        trajectories.append(trajectory)

    #pkl.dump(trajectories, open(f'trajectories_{crafting_goal}.pkl', 'wb'))


    

