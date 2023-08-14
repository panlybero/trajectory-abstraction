import pickle
import gym
import numpy as np
import torch
from PolicyFromAgentModel import PlanningPolicyFromAgentModelV2
from env import GridWorldEnv

from hardcoded_policy import HardcodedPolicy
from planning_agent import PlanningAgent
from stable_baselines3.ppo import MlpPolicy
from baselines import SBPolicyWrapper
import json
import tqdm


def run_episode(agent, env):
    rewards = []
    obs = env.reset()
    done = False
    while not done:
        # print(obs)
        action = agent.act(
            obs['distance_to_wood'], obs['inventory'], obs['distance_to_trader'])
        # print(action)
        # input()
        obs, reward, done, _ = env.step(action)
        rewards.append(reward)
    # print("Done")
    return sum(rewards)


if __name__ == "__main__":
    possible_predicates = ['(next_to wood)', '(has wood)', '(has planks)',
                           '(has chair_parts)', '(has chair)', '(has decoration)', '(has stick)', '(next_to trader)']
    init_seed = 19
    crafting_goal = 'decoration'
    env = GridWorldEnv(
        n=10, crafting_goal=crafting_goal, max_timesteps=20, success_reward=0, spawn_wood=False, spawn_traders=True)
    expert = HardcodedPolicy(env.n, crafting_goal)
    # agent = PlanningAgent(
    #     env.n, crafting_goal, possible_predicates, pddl_path='pddl/with_trades')

    with open('results/agent_model_decoration_stick_invent=True.pkl', 'rb') as f:
        agent_model = pickle.load(f)
        print(agent_model.inferred_invented_predicates)
        agent_model.inferred_invented_predicates = set(
            ["(invented_1)"]) if crafting_goal == "decoration" else set(["(not (invented_1))"])

    with open('results/agent_model_decoration_stick_invent=False.pkl', 'rb') as f:
        agent_model_noinvent = pickle.load(f)

    agent_model_agent = PlanningPolicyFromAgentModelV2(
        env.n, agent_model, pddl_path='pddl/with_trades')

    agent_model_agent_noinvent = PlanningPolicyFromAgentModelV2(
        env.n, agent_model_noinvent, pddl_path='pddl/with_trades')

    # agent = PlanningPolicyFromAgentModelV2(
    #     env.n, agent_model)

    policy = MlpPolicy.load(
        "/home/plymper/trajectory-abstraction/results/bc_decoration_stick_policy.pkl", 'cpu')

    bc_agent = SBPolicyWrapper(policy)
    # Run the expert agent in the environment

    rewards = {"expert": [], "bc_agent": [],
               "agent_model_agent": [], "agent_model_agent_noinvent": []}
    for _ in tqdm.tqdm(range(20)):
        rewards["expert"].append([])
        rewards["bc_agent"].append([])
        rewards["agent_model_agent"].append([])
        rewards["agent_model_agent_noinvent"].append([])
        for i in range(10):
            np.random.seed(init_seed+i)
            rewards["expert"][-1].append(run_episode(expert, env))
            np.random.seed(init_seed+i)
            rewards["bc_agent"][-1].append(run_episode(bc_agent, env))
            np.random.seed(init_seed+i)
            rewards["agent_model_agent"][-1].append(
                run_episode(agent_model_agent, env))
            agent_model_agent.reset()
            np.random.seed(init_seed+i)
            rewards["agent_model_agent_noinvent"][-1].append(
                run_episode(agent_model_agent_noinvent, env))
            agent_model_agent_noinvent.reset()

    with open(f'results/rewards_novelty_task_{crafting_goal}_4_agents.json', 'w') as f:
        json.dump(rewards, f)
