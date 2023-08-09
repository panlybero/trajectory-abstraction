import imitation
from imitation.data.types import Transitions
import gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.ppo import MlpPolicy

from imitation.algorithms import bc
from imitation.data import rollout
from imitation.data.wrappers import RolloutInfoWrapper
from imitation.util.networks import RunningNorm
from env import GridWorldEnv
from hardcoded_policy import SBHardCodedPolicy
import datetime
import tqdm
from imitation.rewards.reward_nets import BasicShapedRewardNet
from imitation.util.networks import RunningNorm
from imitation.algorithms.adversarial.airl import AIRL
from imitation.util.util import make_vec_env


def sample_expert_transitions(env, n_new_eps=1):
    expert = SBHardCodedPolicy(env.observation_space, env.action_space,
                               device='cpu', n=env.n, crafting_goal=env.crafting_goal)  # train_expert()

    print("Sampling expert transitions.")
    rollouts = rollout.rollout(
        expert,
        DummyVecEnv([lambda: RolloutInfoWrapper(env)]),
        rollout.make_sample_until(min_timesteps=None, min_episodes=n_new_eps),
        rng=np.random.RandomState(0)

    )

    return rollout.flatten_trajectories(rollouts)


def combine_datasets(d1, d2):
    return Transitions(
        obs=np.concatenate((d1.obs, d2.obs)),
        acts=np.concatenate((d1.acts, d2.acts)),
        next_obs=np.concatenate((d1.next_obs, d2.next_obs)),
        dones=np.concatenate((d1.dones, d2.dones)),
        infos=np.concatenate((d1.infos, d2.infos)),

    )


def airl_on_env(env, n_new_eps=1, policy=None, replay_buffer=None, buffer_size=100, epochs=10):

    gym.envs.registration.register(
        id='MyCraftWorld-v0',
        entry_point=GridWorldEnv,
        max_episode_steps=20,  # Customize to your needs.
        reward_threshold=500  # Customize to your needs.
    )

    expert = SBHardCodedPolicy(env.observation_space, env.action_space,
                               device='cpu', n=env.n, crafting_goal=env.crafting_goal)  # train_expert()

    transitions = sample_expert_transitions(env, n_new_eps=n_new_eps)
    if replay_buffer is not None:
        data = combine_datasets(transitions, replay_buffer)
    else:
        data = transitions

    if replay_buffer is None:
        replay_buffer = transitions[:buffer_size]
    else:

        replay_buffer = combine_datasets(
            transitions, replay_buffer)[:buffer_size]

    if policy is None:
        policy = PPO(env=env, policy=MlpPolicy)
    # learner = PPO(env=env, policy=MlpPolicy)
    reward_net = BasicShapedRewardNet(
        env.observation_space,
        env.action_space,
        normalize_input_layer=RunningNorm,
    )
    rng = np.random.default_rng(0)
    venv = make_vec_env("MyCraftWorld-v0", rng=rng,
                        n_envs=1, env_make_kwargs={"n": env.n, 'crafting_goal': env.crafting_goal, 'max_timesteps': 20, 'success_reward': 0, 'dict_obs': False})

    airl_trainer = AIRL(
        demonstrations=data,
        demo_batch_size=min(len(data), 32),
        gen_replay_buffer_capacity=buffer_size,
        n_disc_updates_per_round=4,
        venv=venv,
        gen_algo=policy,
        reward_net=reward_net,
        gen_train_timesteps=min(len(data), 32),
        allow_variable_horizon=True,


    )
    try:
        airl_trainer.train(min(len(data), 32))
    except AssertionError as e:
        print(e)
        print(len(data))
        raise e
        return policy, replay_buffer, 0, 0, 0
    # reward, _ = evaluate_policy(
    #     bc_trainer.policy,  # type: ignore[arg-type]
    #     env,
    #     n_eval_episodes=10,
    #     render=False,
    # )
    # print(f"Reward before training: {reward}")

    print("Training a policy using AIRL")
    # bc_trainer.train(n_epochs=epochs, progress_bar=False, log_interval=1000)
    expert_reward, reward_std = evaluate_policy(
        expert,  # type: ignore[arg-type]
        env,
        n_eval_episodes=1,
        render=False,

    )

    reward, reward_std = evaluate_policy(
        policy,  # type: ignore[arg-type]
        env,
        n_eval_episodes=1,
        render=False,
    )
    print(f"Reward after training: {reward}")

    return policy, replay_buffer, reward, reward_std, expert_reward


def bc_on_env(env, n_new_eps=1, policy=None, replay_buffer=None, buffer_size=100, epochs=10):
    expert = SBHardCodedPolicy(env.observation_space, env.action_space,
                               device='cpu', n=env.n, crafting_goal=env.crafting_goal)  # train_expert()

    transitions = sample_expert_transitions(env, n_new_eps=n_new_eps)
    if replay_buffer is not None:
        data = combine_datasets(transitions, replay_buffer)
    else:
        data = transitions

    if replay_buffer is None:
        replay_buffer = transitions[:buffer_size]
    else:

        replay_buffer = combine_datasets(
            transitions, replay_buffer)[:buffer_size]

    print(f"Replay buffer size: {len(replay_buffer)}")
    print("data size", len(data))
    bc_trainer = bc.BC(
        observation_space=env.observation_space,
        action_space=env.action_space,
        demonstrations=data,
        policy=policy,
        batch_size=min(len(data), 32),
        rng=np.random.RandomState(0)
    )

    # reward, _ = evaluate_policy(
    #     bc_trainer.policy,  # type: ignore[arg-type]
    #     env,
    #     n_eval_episodes=10,
    #     render=False,
    # )
    # print(f"Reward before training: {reward}")

    print("Training a policy using Behavior Cloning")
    bc_trainer.train(n_epochs=epochs, progress_bar=False, log_interval=1000)
    expert_reward, reward_std = evaluate_policy(
        expert,  # type: ignore[arg-type]
        env,
        n_eval_episodes=1,
        render=False,

    )

    reward, reward_std = evaluate_policy(
        bc_trainer.policy,  # type: ignore[arg-type]
        env,
        n_eval_episodes=1,
        render=False,
    )
    print(f"Reward after training: {reward}")

    return bc_trainer.policy, replay_buffer, reward, reward_std, expert_reward


def bc_decoration_chair():

    env = GridWorldEnv(10, crafting_goal="chair",
                       max_timesteps=100, success_reward=0, dict_obs=False,)

    rewards = []
    expert_rewards = []
    times = []
    for repeat in range(20):
        policy = None
        buffer = None
        rewards.append([])
        expert_rewards.append([])
        now = datetime.datetime.now()
        for goal in ['chair', 'decoration']*2:
            for i in range(50):
                env = GridWorldEnv(10, crafting_goal=goal,
                                   max_timesteps=20, success_reward=0, dict_obs=False,)
                policy, buffer, reward, reward_std, expert_reward = bc_on_env(
                    env, n_new_eps=1, buffer_size=500, policy=policy, replay_buffer=buffer)
                rewards[-1].append(reward)
                expert_rewards[-1].append(expert_reward)
        delta = datetime.datetime.now() - now
        times.append(delta.total_seconds())
    expert_reward = np.array(expert_rewards)

    rewards = np.array(rewards)

    import pickle as pkl
    path = "results/bc_chair_decoration_rewards.pkl"
    with open(path, "wb") as f:
        pkl.dump((expert_reward, rewards), f)

    from plot_reward import plot

    plot(path)
    txtpath = path.replace('pkl', 'txt')
    with open(txtpath, "w") as f:
        f.write(
            f"Expert Reward: {expert_reward.mean()} +- {expert_reward.std()}\n")
        f.write(f"Agent Reward: {rewards.mean()} +- {rewards.std()}\n")
        f.write(f"Mean time per run {np.array(times).mean()}")

    from slack_message import send_message
    send_message(f"BC results: {txtpath}")


def bc_single_task():
    goal = 'stick'
    env = GridWorldEnv(10, crafting_goal=goal,
                       max_timesteps=20, success_reward=0, dict_obs=False,)

    rewards = []
    expert_rewards = []
    times = []
    for repeat in tqdm.tqdm(range(20)):
        policy = None
        buffer = None
        rewards.append([])
        expert_rewards.append([])
        now = datetime.datetime.now()
        for i in range(100):

            policy, buffer, reward, reward_std, expert_reward = bc_on_env(
                env, n_new_eps=1, buffer_size=200, policy=policy, replay_buffer=buffer)
            rewards[-1].append(reward)
            expert_rewards[-1].append(expert_reward)

        delta = datetime.datetime.now() - now
        times.append(delta.total_seconds())
    expert_reward = np.array(expert_rewards)

    rewards = np.array(rewards)

    import pickle as pkl
    path = f"results/bc_{goal}_rewards.pkl"
    with open(path, "wb") as f:
        pkl.dump((expert_reward, rewards), f)

    from plot_reward import plot

    plot(path)
    txtpath = path.replace('pkl', 'txt')
    with open(txtpath, "w") as f:
        f.write(
            f"Expert Reward: {expert_reward.mean()} +- {expert_reward.std()}\n")
        f.write(f"Agent Reward: {rewards.mean()} +- {rewards.std()}\n")
        f.write(f"Mean time per run {np.array(times).mean()}")


def bc_decoration_stick():

    env = GridWorldEnv(10, crafting_goal="decoration",
                       max_timesteps=100, success_reward=0, dict_obs=False,)

    rewards = []
    expert_rewards = []
    times = []
    crafting_goal = 'decoration'
    for repeat in range(20):
        policy = None
        buffer = None
        rewards.append([])
        expert_rewards.append([])
        now = datetime.datetime.now()
        for ep in range(400):
            if ep % 100 == 0:
                crafting_goal = 'decoration' if crafting_goal == 'stick' else 'stick'
            env = GridWorldEnv(10, crafting_goal=crafting_goal,
                               max_timesteps=20, success_reward=0, dict_obs=False,)
            policy, buffer, reward, reward_std, expert_reward = bc_on_env(
                env, n_new_eps=1, buffer_size=200, policy=policy, replay_buffer=buffer, epochs=2)
            rewards[-1].append(reward)
            expert_rewards[-1].append(expert_reward)

        for ep in range(200):
            if ep % 2 == 0:
                crafting_goal = 'decoration' if crafting_goal == 'stick' else 'stick'
            env = GridWorldEnv(10, crafting_goal=crafting_goal,
                               max_timesteps=20, success_reward=0, dict_obs=False,)
            policy, buffer, reward, reward_std, expert_reward = bc_on_env(
                env, n_new_eps=1, buffer_size=200, policy=policy, replay_buffer=buffer, epochs=2)
            rewards[-1].append(reward)
            expert_rewards[-1].append(expert_reward)

        delta = datetime.datetime.now() - now
        times.append(delta.total_seconds())
    expert_reward = np.array(expert_rewards)

    rewards = np.array(rewards)

    import pickle as pkl
    path = "results/bc_decoration_stick_rewards.pkl"
    with open(path, "wb") as f:
        pkl.dump((expert_reward, rewards), f)

    from plot_reward import plot

    plot(path)
    txtpath = path.replace('pkl', 'txt')
    with open(txtpath, "w") as f:
        f.write(
            f"Expert Reward: {expert_reward.mean()} +- {expert_reward.std()}\n")
        f.write(f"Agent Reward: {rewards.mean()} +- {rewards.std()}\n")
        f.write(f"Mean time per run {np.array(times).mean()}")

    from slack_message import send_message
    send_message(f"BC results: {txtpath}")


def airl_decoration_stick():

    env = GridWorldEnv(10, crafting_goal="decoration",
                       max_timesteps=100, success_reward=0, dict_obs=False,)

    rewards = []
    expert_rewards = []
    times = []
    crafting_goal = 'decoration'
    for repeat in range(20):
        policy = None
        buffer = None
        rewards.append([])
        expert_rewards.append([])
        now = datetime.datetime.now()
        for ep in range(400):
            if ep % 100 == 0:
                crafting_goal = 'decoration' if crafting_goal == 'stick' else 'stick'
            env = GridWorldEnv(10, crafting_goal=crafting_goal,
                               max_timesteps=20, success_reward=0, dict_obs=False,)
            policy, buffer, reward, reward_std, expert_reward = airl_on_env(
                env, n_new_eps=1, buffer_size=200, policy=policy, replay_buffer=buffer, epochs=2)
            rewards[-1].append(reward)
            expert_rewards[-1].append(expert_reward)

        for ep in range(200):
            if ep % 2 == 0:
                crafting_goal = 'decoration' if crafting_goal == 'stick' else 'stick'
            env = GridWorldEnv(10, crafting_goal=crafting_goal,
                               max_timesteps=20, success_reward=0, dict_obs=False,)
            policy, buffer, reward, reward_std, expert_reward = airl_on_env(
                env, n_new_eps=1, buffer_size=200, policy=policy, replay_buffer=buffer, epochs=2)
            rewards[-1].append(reward)
            expert_rewards[-1].append(expert_reward)

        delta = datetime.datetime.now() - now
        times.append(delta.total_seconds())
    expert_reward = np.array(expert_rewards)

    rewards = np.array(rewards)

    import pickle as pkl
    path = "results/airl_decoration_stick_rewards.pkl"
    with open(path, "wb") as f:
        pkl.dump((expert_reward, rewards), f)

    from plot_reward import plot

    plot(path)
    txtpath = path.replace('pkl', 'txt')
    with open(txtpath, "w") as f:
        f.write(
            f"Expert Reward: {expert_reward.mean()} +- {expert_reward.std()}\n")
        f.write(f"Agent Reward: {rewards.mean()} +- {rewards.std()}\n")
        f.write(f"Mean time per run {np.array(times).mean()}")


if __name__ == "__main__":
    np.random.seed(0)
    # bc_decoration_stick()
    bc_single_task()
    # airl_decoration_stick()
