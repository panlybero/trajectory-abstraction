import json
import numpy as np
import matplotlib.pyplot as plt
import pickle as pkl


def make_table(lines):
    pre = "\\begin{tabular}{lll} \\hline"
    suf = "\\end{tabular}"
    def make_line(x): return " & ".join(x) + " \\\\ \\hline"

    lines = [make_line(line) for line in lines]
    lines = [pre] + lines + [suf]
    return "\n".join(lines)


def plot(data_path):

    # crafting_goal = 'open_world_invent=True'
    with open(data_path, "rb") as f:
        rewards = pkl.load(f)

    expert, agent = rewards
    print(expert.shape, agent.shape)
    plt.errorbar(x=range(expert.shape[1]), y=expert.mean(
        axis=0), yerr=expert.std(axis=0)/np.sqrt(expert.shape[0]), label='Expert')
    plt.errorbar(x=range(expert.shape[1]), y=agent.mean(axis=0), yerr=agent.std(
        axis=0)/np.sqrt(expert.shape[0]), label='Agent Model')
    plt.legend()
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title("Performance of Policy Derived from Agent Model")

    plotpath = data_path.replace('pkl', 'png')
    plt.savefig(plotpath)


def plot_invent_decoration_stick():
    import matplotlib
    SMALL_SIZE = 14
    MEDIUM_SIZE = 16
    BIGGER_SIZE = 20

    plt.rc('font', size=BIGGER_SIZE)          # controls default text sizes
    plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
    plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
    plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

    with open("results/rewards_decoration_stick_invent=False.pkl", "rb") as f:
        no_invent = pkl.load(f)
        _, no_invent = no_invent

    with open("results/rewards_decoration_stick_invent=True.pkl", "rb") as f:
        invent = pkl.load(f)
        expert, invent = invent

    with open("results/bc_decoration_stick_rewards.pkl", "rb") as f:
        bc = pkl.load(f)
        _, bc = bc

    plt.figure(figsize=(10, 5))
    plt.errorbar(x=range(expert.shape[1]), y=expert.mean(
        axis=0), yerr=expert.std(axis=0)/np.sqrt(expert.shape[0]), label='Expert', elinewidth=0.1)
    plt.errorbar(x=range(expert.shape[1]), y=invent.mean(axis=0), yerr=invent.std(
        axis=0)/np.sqrt(expert.shape[0]), label='Full Agent Model', elinewidth=0.1)
    plt.errorbar(x=range(expert.shape[1]), y=no_invent.mean(axis=0), yerr=no_invent.std(
        axis=0)/np.sqrt(expert.shape[0]), label='Base Agent Model', elinewidth=0.1)

    plt.errorbar(x=range(expert.shape[1]), y=bc.mean(axis=0), yerr=bc.std(
        axis=0)/np.sqrt(expert.shape[0]), label='Online Behavioral cloning', elinewidth=0.1)
    plt.legend()
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.ylim(-20, -1)
    plt.xlim(left=0, right=600)
    # plt.title("Average Reward per Episode for Decoration and Stick task")
    plt.suptitle("Average Reward per Episode for Decoration and Stick task")
    plotpath = 'results/comparison_decoration_stick_new.png'
    # plt.tight_layout()
    plt.savefig(plotpath, transparent=True, dpi=150)


def time_to_mean(data):
    vals = data.mean(axis=0)
    mean_val = data.mean()
    for i, val in enumerate(vals):
        if val > mean_val:
            return i


def plot_chair():

    with open("results/rewards_chair_invent=False.pkl", "rb") as f:
        no_invent = pkl.load(f)
        _, no_invent = no_invent

    with open("results/rewards_chair_invent=True.pkl", "rb") as f:
        invent = pkl.load(f)
        expert, invent = invent

    with open("results/bc_chair_rewards.pkl", "rb") as f:
        bc = pkl.load(f)
        _, bc = bc

    plt.errorbar(x=range(expert.shape[1]), y=expert.mean(
        axis=0), yerr=expert.std(axis=0)/np.sqrt(expert.shape[0]), label='Expert', elinewidth=0.1)
    plt.errorbar(x=range(expert.shape[1]), y=invent.mean(axis=0), yerr=invent.std(
        axis=0)/np.sqrt(expert.shape[0]), label='Agent Model w/ Invention', elinewidth=0.1)
    plt.errorbar(x=range(expert.shape[1]), y=no_invent.mean(axis=0), yerr=no_invent.std(
        axis=0)/np.sqrt(expert.shape[0]), label='Agent Model w/out Invention', elinewidth=0.1)

    plt.errorbar(x=range(expert.shape[1]), y=bc.mean(axis=0), yerr=bc.std(
        axis=0)/np.sqrt(expert.shape[0]), label='Online Behavioral cloning', elinewidth=0.1)
    plt.legend()
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title("Average Reward per Episode for Chair task")

    plotpath = 'results/comparison_chair.png'
    plt.savefig(plotpath)


def single_task_table():

    res = {}
    for goal in ['chair', 'stick']:
        res[goal] = {}
        with open(f"results/rewards_{goal}_invent=False.pkl", "rb") as f:
            no_invent = pkl.load(f)
            _, no_invent = no_invent

        with open(f"results/rewards_{goal}_invent=True.pkl", "rb") as f:
            invent = pkl.load(f)
            expert, invent = invent

        with open(f"results/bc_{goal}_rewards.pkl", "rb") as f:
            bc = pkl.load(f)
            _, bc = bc

        res[goal]['Base Agent Model'] = {
            "Rewards": f"{no_invent[:,-1].mean():.2f} $\pm$ {no_invent[:,-1].std()/np.sqrt(no_invent.shape[0]):.2f}", "Time to Mean": time_to_mean(no_invent)}
        res[goal]['Agent Model w/ Invention'] = {
            "Rewards": f"{invent[:,-1].mean():.2f} $\pm$ {invent[:,-1].std()/np.sqrt(invent.shape[0]):.2f}", "Time to Mean": time_to_mean(invent)}
        res[goal]['Online Behavioral Cloning'] = {
            "Rewards": f"{bc[:,-1].mean():.2f} $\pm$ {bc[:,-1].std()/np.sqrt(bc.shape[0]):.2f}", "Time to Mean": time_to_mean(bc)}
        res[goal]['Expert'] = {"Rewards": f"{expert[:,-1].mean():.2f} $\pm$ {expert[:,-1].std()/np.sqrt(expert.shape[0]):.2f}",
                               "Time to Mean": time_to_mean(expert)}

    import json
    with open('results/single_task_table.json', 'w') as f:
        json.dump(res, f, indent=4)


def plot_novelty_task():
    with open("results/rewards_novelty_task_3_agents.json", "r") as f:
        rewards = json.load(f)

    names = {"expert": "Expert", "bc_agent": "Online BC",
             "agent_model_agent": "Agent Model + IM", "agent_model_agent_noinvent": "Agent Model"}

    for agent in rewards:
        mean = np.array(rewards[agent]).mean(axis=0)
        std = np.array(rewards[agent]).std(axis=0)/np.sqrt(len(rewards[agent]))
        plt.errorbar(x=range(len(mean)), y=mean, yerr=std, label=names[agent])

    plt.legend()
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title("Average Reward per Episode for Novelty Task")
    plotpath = 'results/comparison_novelty_task.png'
    plt.savefig(plotpath)


def novelty_task_table():
    crafting_goal = 'stick'
    with open(f"results/rewards_novelty_task_{crafting_goal}_4_agents.json", "r") as f:
        rewards = json.load(f)

    names = {"expert": "Expert", "bc_agent": "Online BC",
             "agent_model_agent": "Agent Model + IM", "agent_model_agent_noinvent": "Agent Model"}

    for agent in rewards:
        mean = np.array(rewards[agent]).mean()
        std = np.array(rewards[agent]).std()/np.sqrt(len(rewards[agent][0]))
        print(f"{names[agent]}: {mean:.2f} $\pm$ {std:.2f}")


if __name__ == '__main__':

    # path = "results/bc_chair_decoration_rewards.pkl"
    # path = "results/rewards_chair.pkl"
    # plot(path)
    plot_invent_decoration_stick()
    # plot_chair()
    # single_task_table()
    # plot_novelty_task()
    # novelty_task_table()
