import numpy as np
import matplotlib.pyplot as plt
import pickle as pkl


if __name__=='__main__':

    crafting_goal = 'chair'
    with open(f"rewards_{crafting_goal}.pkl", "rb") as f:
        rewards = pkl.load(f)
    
    expert,agent = rewards
    print(expert.shape, agent.shape)
    plt.errorbar(x=range(expert.shape[1]),y=expert.mean(axis=0),yerr=expert.std(axis=0)/np.sqrt(expert.shape[0]), label='Expert')
    plt.errorbar(x=range(expert.shape[1]),y=agent.mean(axis=0),yerr=agent.std(axis=0)/np.sqrt(expert.shape[0]), label='Agent Model')
    plt.legend()
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title("Performance of Policy Derived from Agent Model")

    plt.savefig(f'rewards_{crafting_goal}.png')