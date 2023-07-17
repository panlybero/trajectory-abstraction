import gym
from env import GridWorldEnv
import numpy as np
from hardcoded_policy import HardcodedPolicy
from PolicyFromAgentModel import PolicyFromAgentModel
from AgentModel import AgentModel, prepare_step
from StateCluster import CategoricalStateCluster, StateDependentStateCluster

import pickle as pkl    




def run_ep(env, agent_model, expert_agent, trajectories):
    obs = env.reset()
    expert_agent.reset()
    done = False
    expert_reward = 0
    agent_reward = 0
    trajectory = []
    policy = PolicyFromAgentModel(n=10,agent_model=agent_model)
    done = False

    while not done:
        action = policy.act(obs['distance_to_wood'], obs['inventory'])
        new_obs, reward, done, _ = env.step(action)
        agent_reward += reward
        obs = new_obs

    print(f"Total Agent reward: {agent_reward}")
    done = False
    
        # if agent_model.n_clusters>6:
        #     agent_model.merge_threshold = 0.9*agent_model.merge_threshold
        #     print(f"Lowering Merge threshold to: {agent_model.merge_threshold}, n_clusters: {agent_model.n_clusters}")
        # else:
        #     agent_model.merge_threshold = 1.1*agent_model.merge_threshold
        #     print(f"Raising Merge threshold to: {agent_model.merge_threshold}, n_clusters: {agent_model.n_clusters}")
        #agent_model.plot_transition_graph(actions,f'agent_model_plots/file{ep}.dot')
    
    obs = env.reset()
    while not done:
        action = expert_agent.act(obs['distance_to_wood'], obs['inventory'])
      
        new_obs, reward, done, _ = env.step(action)
        expert_reward += reward
        trajectory.append((obs, action, reward, new_obs))
        
        step = prepare_step(agent_model,(obs, action, reward, new_obs))
        agent_model.process_step(step)
        obs = new_obs

    print(f"Total Expert reward: {expert_reward}")
        
    trajectories.append(trajectory)
    return expert_reward, agent_reward





def agent_model_experiment():
    crafting_goal = 'chair'
    env = GridWorldEnv(n=10, crafting_goal=crafting_goal, max_timesteps=20)
    
     
    actions = ['up', 'down', 'left', 'right', 'craft_planks', 'craft_chair_parts', 'craft_chair', 'craft_decoration']
    possible_predicates = ['(next_to wood)', '(has wood)', '(has planks)', '(has chair_parts)', '(has chair)', '(has decoration)']
    
    expert_agent = HardcodedPolicy(10, crafting_goal)
    # Create an instance of the HardcodedPolicy
    #policy = PolicyFromAgentModel(n=10,agent_model=agent_model)
    
    trajectories = []
    # Reset the environment
    
    expert_rewards = []
    agent_rewards = []

    for _ in range(20):
        expert_rewards.append([])
        agent_rewards.append([])
        agent_model = AgentModel(len(actions),possible_predicates,0.1,0.1, cluster_class=StateDependentStateCluster)
        for ep in range(100):
            expert_r,agent_r = run_ep(env, agent_model, expert_agent, trajectories)
            expert_rewards[-1].append(expert_r)
            agent_rewards[-1].append(agent_r)
        #agent_model.reset()


    expert_rewards = np.array(expert_rewards)
    agent_rewards = np.array(agent_rewards)

    with open(f'rewards_{crafting_goal}.pkl', 'wb') as f:
        pkl.dump((expert_rewards,agent_rewards), f)
    
    agent_model.plot_transition_graph(actions,f'agent_model_plots/file_{crafting_goal}.dot')


def agent_model_experiment_change_goal():
    crafting_goal = 'decoration'
    
    
     
    actions = ['up', 'down', 'left', 'right', 'craft_planks', 'craft_chair_parts', 'craft_chair', 'craft_decoration']
    possible_predicates = ['(next_to wood)', '(has wood)', '(has planks)', '(has chair_parts)', '(has chair)', '(has decoration)']
    
    
    # Create an instance of the HardcodedPolicy
    #policy = PolicyFromAgentModel(n=10,agent_model=agent_model)
    
    trajectories = []
    # Reset the environment
    
    expert_rewards = []
    agent_rewards = []

    for _ in range(20):
        expert_rewards.append([])
        agent_rewards.append([])
        agent_model = AgentModel(len(actions),possible_predicates,0.1,0.01, cluster_class=StateDependentStateCluster)
        for ep in range(100):
            crafting_goal = 'decoration'
            expert_agent = HardcodedPolicy(10, crafting_goal)
            env = GridWorldEnv(n=10, crafting_goal=crafting_goal, max_timesteps=20)
            expert_r,agent_r = run_ep(env, agent_model, expert_agent, trajectories)
            expert_rewards[-1].append(expert_r)
            agent_rewards[-1].append(agent_r)

        for ep in range(100):
            crafting_goal = 'chair'
            expert_agent = HardcodedPolicy(10, crafting_goal)
            env = GridWorldEnv(n=10, crafting_goal=crafting_goal, max_timesteps=20)
            expert_r,agent_r = run_ep(env, agent_model, expert_agent, trajectories)

            expert_rewards[-1].append(expert_r)
            agent_rewards[-1].append(agent_r)
        #agent_model.reset()

    expert_rewards = np.array(expert_rewards)
    agent_rewards = np.array(agent_rewards)

    with open(f'rewards_decoration_chair.pkl', 'wb') as f:
        pkl.dump((expert_rewards,agent_rewards), f)
    
    agent_model.plot_transition_graph(actions,f'agent_model_plots/file_decoration_chair.dot')

if __name__=='__main__':
    
    agent_model_experiment()
    #agent_model_experiment_change_goal()
    


    

