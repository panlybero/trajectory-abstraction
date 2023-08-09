import gym
from env import GridWorldEnv
import numpy as np
from hardcoded_policy import HardcodedPolicy
from PolicyFromAgentModel import PolicyFromAgentModel, PlanningPolicyFromAgentModel, PlanningPolicyFromAgentModelV2
from AgentModel import AgentModel, prepare_step
from StateCluster import CategoricalStateCluster, StateDependentStateCluster
import tqdm
import pickle as pkl
actions = ['up', 'down', 'left', 'right', 'craft_planks',
           'craft_chair_parts', 'craft_chair', 'craft_decoration', 'craft_stick']


def run_ep(env, agent_model: AgentModel, expert_agent, trajectories):
    obs = env.reset()
    expert_agent.reset()
    done = False
    expert_reward = 0
    agent_reward = 0
    trajectory = []
    agent_policy = PlanningPolicyFromAgentModelV2(
        n=env.n, agent_model=agent_model)
    done = False
    # print('Assuming this:', agent_model.inferred_invented_predicates)

    obs = env.reset()
    prev_step = (None, None, None)
    done = False
    agent_model.reset()
    while not done:
        action = expert_agent.act(obs['distance_to_wood'], obs['inventory'])

        new_obs, reward, done, _ = env.step(action)
        expert_reward += reward
        trajectory.append((obs, action, reward, new_obs))
        if prev_step[0] is not None:
            step = prepare_step(
                agent_model, (prev_step[0], prev_step[1], prev_step[2], obs, action))

            agent_model.process_step(step)
            # print('Model size', agent_model.n_clusters)
        prev_step = (obs, action, reward)
        obs = new_obs

    step = prepare_step(
        agent_model, (prev_step[0], prev_step[1], prev_step[2], obs, None))
    agent_model.process_step(step)
    # print('Model size', agent_model.n_clusters)

    agent_actions = []
    done = False
    obs = env.reset()
    observs = []
    while not done:
        action = agent_policy.act(obs['distance_to_wood'], obs['inventory'])
        new_obs, reward, done, _ = env.step(action)
        agent_reward += reward
        agent_actions.append(action)
        obs = new_obs
        observs.append((obs, action))
        # print(action)

    # if agent_reward == -20:
    #     # print()
    #     # print(agent_model.goals)
    #     # print(env.crafting_goal)
    #     # print(observs)
    #     # print(agent_actions)
    #     print(agent_model.n_clusters)

        # agent_model.plot_transition_graph(
        #     actions=actions, fname=f'agent_model_plots/file_tmp.dot')
        # input()

    # if agent_model.n_clusters>6:
    #     agent_model.merge_threshold = 0.9*agent_model.merge_threshold
    #     print(f"Lowering Merge threshold to: {agent_model.merge_threshold}, n_clusters: {agent_model.n_clusters}")
    # else:
    #     agent_model.merge_threshold = 1.1*agent_model.merge_threshold
    #     print(f"Raising Merge threshold to: {agent_model.merge_threshold}, n_clusters: {agent_model.n_clusters}")
    # agent_model.plot_transition_graph(actions,f'agent_model_plots/file{ep}.dot')

    # print('Infered this:', agent_model.inferred_invented_predicates)

    # print(f"Total Expert reward: {expert_reward}")

    trajectories.append(trajectory)
    return expert_reward, agent_reward


def agent_model_experiment():
    crafting_goal = 'stick'
    invent = True
    actions = ['up', 'down', 'left', 'right', 'craft_planks',
               'craft_chair_parts', 'craft_chair', 'craft_decoration', 'craft_stick']
    possible_predicates = ['(next_to wood)', '(has wood)', '(has planks)',
                           '(has chair_parts)', '(has chair)', '(has decoration)', '(has stick)']
    expert_agent = HardcodedPolicy(10, crafting_goal)
    # Create an instance of the HardcodedPolicy
    # policy = PolicyFromAgentModel(n=10,agent_model=agent_model)

    trajectories = []
    # Reset the environment

    expert_rewards = []
    agent_rewards = []

    for _ in tqdm.tqdm(range(20)):
        expert_rewards.append([])
        agent_rewards.append([])
        agent_model = AgentModel(len(
            actions), possible_predicates, 0.01, 0.1, cluster_class=CategoricalStateCluster, max_n_clusters=8, invent_predicates=invent)

        env = GridWorldEnv(n=10, crafting_goal=crafting_goal,
                           max_timesteps=20, success_reward=0)
        for ep in range(100):
            expert_r, agent_r = run_ep(
                env, agent_model, expert_agent, trajectories)
            expert_rewards[-1].append(expert_r)
            agent_rewards[-1].append(agent_r)
        # agent_model.reset()

    expert_rewards = np.array(expert_rewards)
    agent_rewards = np.array(agent_rewards)

    with open(f'results/rewards_{crafting_goal}_invent={invent}.pkl', 'wb') as f:
        pkl.dump((expert_rewards, agent_rewards), f)

    agent_model.plot_transition_graph(
        actions, f'agent_model_plots/file_{crafting_goal}_invent={invent}.dot')

    from plot_reward import plot
    plot(f'results/rewards_{crafting_goal}_invent={invent}.pkl')


def agent_model_experiment_change_goal():
    crafting_goal = 'decoration'
    invent = False
    actions = ['up', 'down', 'left', 'right', 'craft_planks',
               'craft_chair_parts', 'craft_chair', 'craft_decoration', 'craft_stick']
    possible_predicates = ['(next_to wood)', '(has wood)', '(has planks)',
                           '(has chair_parts)', '(has chair)', '(has decoration)', '(has stick)']
    # Create an instance of the HardcodedPolicy
    # policy = PolicyFromAgentModel(n=10,agent_model=agent_model)

    trajectories = []
    # Reset the environment

    expert_rewards = []
    agent_rewards = []

    for _ in tqdm.tqdm(range(20)):
        expert_rewards.append([])
        agent_rewards.append([])
        agent_model = AgentModel(len(actions), possible_predicates, 0.01, 0.1,
                                 cluster_class=CategoricalStateCluster, invent_predicates=invent)
        for crafting_goal in ['chair', 'decoration']*2:
            for ep in range(50):
                # crafting_goal = 'decoration'
                expert_agent = HardcodedPolicy(10, crafting_goal)
                env = GridWorldEnv(
                    n=10, crafting_goal=crafting_goal, max_timesteps=20, success_reward=0)
                expert_r, agent_r = run_ep(
                    env, agent_model, expert_agent, trajectories)
                expert_rewards[-1].append(expert_r)
                agent_rewards[-1].append(agent_r)
                # agent_model.cluster_transitions._apply_linear_decay(1)

        # agent_model.reset()

    expert_rewards = np.array(expert_rewards)
    agent_rewards = np.array(agent_rewards)

    with open(f'results/rewards_decoration_chair_invent={invent}.pkl', 'wb') as f:
        pkl.dump((expert_rewards, agent_rewards), f)

    agent_model.plot_transition_graph(
        actions, f'agent_model_plots/file_decoration_chair_invent={invent}.dot')

    from plot_reward import plot
    plot(f'results/rewards_decoration_chair_invent={invent}.pkl')


def agent_model_experiment_invent_predicate():
    crafting_goal = 'decoration'
    invent = False

    actions = ['up', 'down', 'left', 'right', 'craft_planks',
               'craft_chair_parts', 'craft_chair', 'craft_decoration', 'craft_stick']
    possible_predicates = ['(next_to wood)', '(has wood)', '(has planks)',
                           '(has chair_parts)', '(has chair)', '(has decoration)', '(has stick)']

    # Create an instance of the HardcodedPolicy
    # policy = PolicyFromAgentModel(n=10,agent_model=agent_model)

    trajectories = []
    # Reset the environment

    expert_rewards = []
    agent_rewards = []

    for _ in tqdm.tqdm(range(20)):
        expert_rewards.append([])
        agent_rewards.append([])
        agent_model = AgentModel(len(actions), possible_predicates,
                                 0.01, 0.1, cluster_class=CategoricalStateCluster, invent_predicates=invent, max_n_clusters=8)
        for ep in range(400):
            if ep % 100 == 0:
                crafting_goal = 'decoration' if crafting_goal == 'stick' else 'stick'
                # agent_model.plot_transition_graph(
                #    actions, 'agent_model_plots/file_tmp.dot')
                # input("here")

            env = GridWorldEnv(
                n=10, crafting_goal=crafting_goal, max_timesteps=20, success_reward=0)

            expert_agent = HardcodedPolicy(env.n, crafting_goal)

            expert_r, agent_r = run_ep(
                env, agent_model, expert_agent, trajectories)

            expert_rewards[-1].append(expert_r)
            agent_rewards[-1].append(agent_r)
        for ep in range(200):
            if ep % 2 == 0:
                crafting_goal = 'decoration' if crafting_goal == 'stick' else 'stick'
                # agent_model.plot_transition_graph(
                #    actions, 'agent_model_plots/file_tmp.dot')
                # input("here")

            env = GridWorldEnv(
                n=10, crafting_goal=crafting_goal, max_timesteps=20, success_reward=0)

            expert_agent = HardcodedPolicy(env.n, crafting_goal)

            expert_r, agent_r = run_ep(
                env, agent_model, expert_agent, trajectories)

            expert_rewards[-1].append(expert_r)
            agent_rewards[-1].append(agent_r)
            # print(agent_model.n_clusters)
        # agent_model.plot_transition_graph(
            # actions=actions, fname=f'agent_model_plots/file_tmp.dot')
        # input("Done longs")

        # for ep in range(100):
        #     if ep % 5 == 0:
        #         crafting_goal = 'decoration' if crafting_goal == 'stick' else 'stick'

        #     env = GridWorldEnv(
        #         n=10, crafting_goal=crafting_goal, max_timesteps=20, success_reward=0)
        #     expert_agent = HardcodedPolicy(env.n, crafting_goal)
        #     expert_r, agent_r = run_ep(
        #         env, agent_model, expert_agent, trajectories)
        #     expert_rewards[-1].append(expert_r)
        #     agent_rewards[-1].append(agent_r)

        # print("Changing goal")

        # agent_model.reset()

        # agent_model.plot_transition_graph(
        #    actions=actions, fname=f'agent_model_plots/file_tmp.dot')
        # input("Done short")

    expert_rewards = np.array(expert_rewards)
    agent_rewards = np.array(agent_rewards)

    with open(f'results/rewards_decoration_stick_invent={invent}.pkl', 'wb') as f:
        pkl.dump((expert_rewards, agent_rewards), f)

    agent_model.plot_transition_graph(
        actions, f'agent_model_plots/file_decoration_stick_invent={invent}.dot')

    from plot_reward import plot

    plot(f'results/rewards_decoration_stick_invent={invent}.pkl')


def agent_model_experiment_open_world():
    crafting_goal = 'decoration'
    invent = True
    decay = 1

    actions = ['up', 'down', 'left', 'right', 'craft_planks',
               'craft_chair_parts', 'craft_chair', 'craft_decoration', 'craft_stick']
    possible_predicates = ['(next_to wood)', '(has wood)', '(has planks)',
                           '(has chair_parts)', '(has chair)', '(has decoration)', '(has stick)']

    # Create an instance of the HardcodedPolicy
    # policy = PolicyFromAgentModel(n=10,agent_model=agent_model)

    trajectories = []
    # Reset the environment

    expert_rewards = []
    agent_rewards = []
    possible_goals = ['decoration', 'chair',
                      'stick', 'planks', 'chair_parts']
    goals = ['decoration', 'chair',
             'stick', 'planks', 'chair_parts', "decoration", 'chair']  # np.random.choice(possible_goals, size=10, replace=True)
    print(goals)
    import tqdm

    for _ in tqdm.tqdm(range(20)):
        expert_rewards.append([])
        agent_rewards.append([])
        agent_model = AgentModel(len(actions), possible_predicates,
                                 0.05, 0.1, cluster_class=CategoricalStateCluster, invent_predicates=invent)
        for goal in goals:
            crafting_goal = goal
            for ep in range(10):

                expert_agent = HardcodedPolicy(10, crafting_goal)
                env = GridWorldEnv(
                    n=10, crafting_goal=crafting_goal, max_timesteps=20, success_reward=0)
                expert_r, agent_r = run_ep(
                    env, agent_model, expert_agent, trajectories)
                expert_rewards[-1].append(expert_r)
                agent_rewards[-1].append(agent_r)
                # print("Changing goal")
                agent_model.cluster_transitions._apply_linear_decay(decay)

            agent_model.cluster_transitions.get_transition_matrix(agent_model.clusters,
                                                                  get_counts=True)

        # agent_model.reset()

    expert_rewards = np.array(expert_rewards)
    agent_rewards = np.array(agent_rewards)

    with open(f'results/rewards_open_world_invent={invent}_decay={decay}.pkl', 'wb') as f:
        pkl.dump((expert_rewards, agent_rewards), f)

    agent_model.plot_transition_graph(
        actions, f'agent_model_plots/file_open_world_invent={invent}_decay={decay}.dot')

    from plot_reward import plot
    plot(f'results/rewards_open_world_invent={invent}_decay={decay}.pkl')


if __name__ == '__main__':

    agent_model_experiment()
    # agent_model_experiment_change_goal()
    # agent_model_experiment_invent_predicate()

    # agent_model_experiment_open_world()
