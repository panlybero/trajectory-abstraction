import numpy as np
import gym
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.ppo import MlpPolicy

from imitation.algorithms.adversarial.gail import GAIL
from imitation.data import rollout
from imitation.data.wrappers import RolloutInfoWrapper
from imitation.rewards.reward_nets import BasicRewardNet
from imitation.util.networks import RunningNorm
from imitation.util.util import make_vec_env
from baselines import combine_datasets
from hardcoded_policy import SBHardCodedPolicy
from env import GridWorldEnv
import torch

import pickle as pkl


def make_dataset(goal='decoration', n_eps=100):

    rng = np.random.default_rng(0)
    env = GridWorldEnv(10, crafting_goal=goal,
                       max_timesteps=100, success_reward=0, dict_obs=False,)
    expert = SBHardCodedPolicy(env.observation_space, env.action_space,
                               device='cpu', n=env.n, crafting_goal=env.crafting_goal)  # train_expert()
    rollouts = rollout.rollout(
        expert,
        make_vec_env(
            "MyCraftWorld-v0",
            n_envs=1,
            post_wrappers=[lambda env, _: RolloutInfoWrapper(env)],
            rng=rng,
            env_make_kwargs={"n": env.n, 'crafting_goal': env.crafting_goal,
                             'max_timesteps': 20, 'success_reward': 0, 'dict_obs': False}
        ),
        rollout.make_sample_until(min_timesteps=None, min_episodes=n_eps),
        rng=rng,
    )
    return rollouts


def make_task_dataset():
    dataset = []
    d1 = make_dataset('decoration', n_eps=100)
    d2 = make_dataset('stick', n_eps=100)
    dataset += d1
    dataset += d2
    for i in range(200):
        d1 = make_dataset('decoration', n_eps=1)
        d2 = make_dataset('stick', n_eps=1)
        dataset += d1
        dataset += d2

    return dataset


def train():
    gym.envs.registration.register(
        id='MyCraftWorld-v0',
        entry_point=GridWorldEnv,
        max_episode_steps=20,  # Customize to your needs.
        reward_threshold=500  # Customize to your needs.
    )

    env = GridWorldEnv(10, crafting_goal="decoration",
                       max_timesteps=100, success_reward=0, dict_obs=False,)
    rng = np.random.default_rng(0)

    expert = SBHardCodedPolicy(env.observation_space, env.action_space,
                               device='cpu', n=env.n, crafting_goal=env.crafting_goal)  # train_expert()
    rollouts = make_task_dataset()
    # env = gym.make("MyCraftWorld-v0")
    # expert = PPO(policy=MlpPolicy, env=env, n_steps=64)
    # expert.learn(1000)

    venv = make_vec_env(
        "MyCraftWorld-v0",
        n_envs=1,
        post_wrappers=[lambda env, _: RolloutInfoWrapper(env)],
        rng=rng,
        env_make_kwargs={"n": env.n, 'crafting_goal': env.crafting_goal,
                         'max_timesteps': 20, 'success_reward': 0, 'dict_obs': False}
    )
    learner = PPO(env=venv, policy=MlpPolicy)
    reward_net = BasicRewardNet(
        venv.observation_space,
        venv.action_space,
        use_action=True

        # normalize_input_layer=RunningNorm,
    )
    gail_trainer = GAIL(
        demonstrations=rollouts,
        demo_batch_size=100,
        gen_replay_buffer_capacity=100,
        n_disc_updates_per_round=4,
        venv=venv,
        gen_algo=learner,
        reward_net=reward_net,
        allow_variable_horizon=True,
    )

    gail_trainer.train(200000)
    rewards, _ = evaluate_policy(
        learner, venv, 100, return_episode_rewards=True)
    print("Rewards:", rewards)

    torch.save(reward_net.state_dict(), 'reward_net.pt')
    return reward_net


# def measure_reward_at_last_state():

def irl_run():
    env = GridWorldEnv(10, crafting_goal="decoration",
                       max_timesteps=20, success_reward=0, dict_obs=False,)
    rewardnet = train()

    obs = env.reset()
    done = False
    expert = SBHardCodedPolicy(env.observation_space, env.action_space,
                               'cpu', n=env.n, crafting_goal=env.crafting_goal)  # train_expert()
    while not done:
        action, _ = expert.predict(obs, deterministic=True)
        newobs, reward, done, _ = env.step(action)
        # print(obs.shape, action, newobs.shape, done)
        rew = rewardnet(
            torch.tensor(obs).reshape(1, -1).float(), onehot_tensor(action, 9), torch.tensor(newobs).reshape(1, -1).float(), done)
        obs = newobs
    decoration_rew = rew

    env = GridWorldEnv(10, crafting_goal="stick",
                       max_timesteps=20, success_reward=0, dict_obs=False,)

    obs = env.reset()
    done = False
    expert = SBHardCodedPolicy(env.observation_space, env.action_space,
                               'cpu', n=env.n, crafting_goal=env.crafting_goal)  # train_expert()
    while not done:
        action, _ = expert.predict(obs, deterministic=True)
        newobs, reward, done, _ = env.step(action)
        # print(obs.shape, action, newobs.shape, done)
        rew = rewardnet(
            torch.tensor(obs).reshape(1, -1).float(), onehot_tensor(action, 9), torch.tensor(newobs).reshape(1, -1).float(), done)
        obs = newobs
    stick_rew = rew

    return decoration_rew.detach().cpu().numpy(), stick_rew.detach().cpu().numpy()


def irl_experiment():
    decoration_rews = []
    stick_rews = []
    for i in range(10):
        decoration_rew, stick_rew = irl_run()
        decoration_rews.append(decoration_rew)
        stick_rews.append(stick_rew)
    print(np.mean(decoration_rews), np.std(decoration_rews))
    print(np.mean(stick_rews), np.std(stick_rews))

    result = {'decoration': decoration_rews, 'stick': stick_rews}
    with open('irl_results.pkl', 'wb') as f:
        pkl.dump(result, f)

    return decoration_rews, stick_rews


def onehot_tensor(x, n):
    x = x
    return torch.eye(n)[x].float().reshape(1, -1)


if __name__ == '__main__':

    irl_experiment()
    exit()
    env = GridWorldEnv(10, crafting_goal="decoration",
                       max_timesteps=20, success_reward=0, dict_obs=False,)

    # rewardnet = train()
    rewardnet = BasicRewardNet(
        env.observation_space,
        env.action_space,
        use_action=True

        # normalize_input_layer=RunningNorm,
    )
    rewardnet.load_state_dict(torch.load('reward_net.pt'))

    obs = env.reset()
    done = False
    expert = SBHardCodedPolicy(env.observation_space, env.action_space,
                               'cpu', n=env.n, crafting_goal=env.crafting_goal)  # train_expert()
    while not done:
        action, _ = expert.predict(obs, deterministic=True)
        newobs, reward, done, _ = env.step(action)
        # print(obs.shape, action, newobs.shape, done)
        rew = rewardnet(
            torch.tensor(obs).reshape(1, -1).float(), onehot_tensor(action, 9), torch.tensor(newobs).reshape(1, -1).float(), done)
        print(obs, reward, rew)
        obs = newobs

    env = GridWorldEnv(10, crafting_goal="stick",
                       max_timesteps=20, success_reward=0, dict_obs=False,)

    obs = env.reset()
    done = False
    expert = SBHardCodedPolicy(env.observation_space, env.action_space,
                               'cpu', n=env.n, crafting_goal=env.crafting_goal)  # train_expert()
    while not done:
        action, _ = expert.predict(obs, deterministic=True)
        newobs, reward, done, _ = env.step(action)
        # print(obs.shape, action, newobs.shape, done)
        rew = rewardnet(
            torch.tensor(obs).reshape(1, -1).float(), onehot_tensor(action, 9), torch.tensor(newobs).reshape(1, -1).float(), done)
        print(obs, reward, rew)
        obs = newobs
