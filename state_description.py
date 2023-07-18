import pickle as pkl
import numpy as np

from MarkovProcessModel import MarkovProcessModel
from StateDescription import StateDescription, StateDescriptionFactory
from sklearn.cluster import AgglomerativeClustering, KMeans, SpectralClustering

from scipy.stats import entropy


def describe_state(state, as_num=False):
    distances = state['distance_to_wood']
    inventory = state['inventory']
    def cast(x): return int(x) if as_num else bool(x)

    is_next_to_wood = cast(np.any(distances == 0))
    has_wood = cast(inventory[0] > 0)
    has_planks = cast(inventory[1] > 0)
    has_chair_parts = cast(inventory[2] > 0)
    has_chair = cast(inventory[3] > 0)
    has_decoration = cast(inventory[4] > 0)
    has_stick = cast(inventory[5] > 0)
    predicate_dict = {"(next_to wood)": is_next_to_wood, "(has wood)": has_wood, "(has planks)": has_planks,
                      "(has chair_parts)": has_chair_parts, "(has chair)": has_chair, "(has decoration)": has_decoration, "(has stick)": has_stick}

    return predicate_dict


def describe_array_state(state, as_num=False):
    distances = state[:8]
    inventory = state[8:]
    return describe_state({"distance_to_wood": distances, "inventory": inventory}, as_num=as_num)


def get_truth_fractions_per_group(states, groups):
    tmp = describe_array_state(states[0], as_num=True)
    predicate_dict = {k: 0 for k in tmp}
    group_descriptions = {i: predicate_dict.copy()
                          for i in range(len(set(groups)))}
    group_counts = {i: 0 for i in range(len(set(groups)))}

    for state, group in zip(states, groups):
        state_description = describe_array_state(state, as_num=True)
        group_counts[group] += 1
        group_descriptions[group] = {k: v+state_description[k]
                                     for k, v in group_descriptions[group].items()}

    for group in group_descriptions:
        group_descriptions[group] = {k: v/group_counts[group]
                                     for k, v in group_descriptions[group].items()}

    print(group_descriptions)
    print(group_counts)


def description_to_set(description):
    return frozenset(set([k for k, v in description.items() if v == True]).union(set([f"(not {k})" for k, v in description.items() if v == False])))


def subsumes(d1, d2):
    if d1.issubset(d2):
        return True
    return False


def get_least_general_generalization(descriptions):
    all_preds = set()
    for d in descriptions:
        for p in list(d):
            all_preds.add(p)

    for p in list(all_preds):
        for d in descriptions:
            if p not in d:
                all_preds.remove(p)
                break

    return all_preds


def compute_generalizations(descriptions, labels):
    groups = {i: [] for i in set(labels)}
    for d, l in zip(descriptions, labels):
        groups[l].append(d)
    generalized_states = []
    for g in groups:
        generalized_states.append(get_least_general_generalization(groups[g]))

    return generalized_states


def get_most_specific_generalization(generalizations, description):

    overlap = []

    for g in generalizations:
        if subsumes(g, description):
            overlap.append(len(g.intersection(description)))
        else:
            overlap.append(-1)
    if np.max(overlap) == -1:
        raise ValueError("No generalization found")

    return generalizations[np.argmax(overlap)]


def fit_transition_model(gen_trajectories, generalizations):
    model = MarkovProcessModel(num_states=len(generalizations))

    numbered_trajectories = []
    for ep in gen_trajectories:
        numbered_ep = []
        for step in ep:
            numbered_ep.append((generalizations.index(step[0])))
        numbered_trajectories.append(numbered_ep)

    model.fit(numbered_trajectories)
    return model


def get_per_step_action_probs(describe_state, data, n_actions=7):
    description_to_steps = {}

    for ep in data:
        for step in ep:
            d = describe_state(step[0])

            tmp = description_to_steps.get(d, [])
            tmp.append(step)
            description_to_steps[d] = tmp

    per_set_action_probs = {s: np.zeros(n_actions)
                            for s in description_to_steps}

    for s in description_to_steps:
        for step in description_to_steps[s]:
            per_set_action_probs[s][step[1]] += 1

    # for s in per_set_action_probs:
    #    per_set_action_probs[s] /= np.sum(per_set_action_probs[s])
    return per_set_action_probs


def generalized_description_of_state(generalizations):
    def f(state):
        return frozenset(get_most_specific_generalization(generalizations, description_to_set(describe_state(state))))
    return f


def plot_model(data, generalizations, model, fname='file.dot', actions=['up', 'down', 'left', 'right', 'craft_planks', 'craft_chair_parts', 'craft_chair'], from_data=True, clustering=None):
    if from_data:
        per_set_action_probs = get_per_step_action_probs(
            lambda x: x, data, n_actions=len(actions))
    else:
        cluster_counts = clustering.compute_cluster_counts()
        generalizations = list(clustering.generalizations)
        per_set_action_probs = {
            k: cluster_counts[i]/np.sum(cluster_counts[i]) for i, k in enumerate(generalizations)}

    per_set_action_probs = {k: [(actions[i], f"{v:.2f}") for i, v in enumerate(
        per_set_action_probs[k])] for k in per_set_action_probs}
    print(per_set_action_probs)

    node_labels = {i: str(
        g)+'\n'+str(per_set_action_probs[g]) for i, g in enumerate(generalizations)}
    model.plot_transition_graph(node_labels=node_labels, fname=fname)


def get_per_generalization_data(data, generalizations):
    per_generalization_data_states = {
        frozenset(i): [] for i in generalizations}
    per_generalization_data_actions = {
        frozenset(i): [] for i in generalizations}
    per_generalization_samples = {frozenset(i): [] for i in generalizations}

    for i, ep in enumerate(data):
        for d in ep:
            state = frozenset(description_to_set(describe_state(d[0])))
            d_gen = frozenset(
                get_most_specific_generalization(generalizations, state))
            per_generalization_data_states[d_gen].append(state)
            per_generalization_data_actions[d_gen].append(d[1])
            print(d)
            per_generalization_samples[d_gen].append(d)

    per_generalization_action_probs = {g: np.zeros(
        7) for g in per_generalization_data_actions}
    for g in per_generalization_data_actions:
        for a in per_generalization_data_actions[g]:
            per_generalization_action_probs[g][a] += 1
        per_generalization_action_probs[g] /= np.sum(
            per_generalization_action_probs[g])
    return per_generalization_action_probs, per_generalization_data_states, per_generalization_data_actions, per_generalization_samples


def invent_symbol(state_description, action_dist):
    new_predicate = "_invented_symbol_"

    new_description1 = frozenset(state_description.union(set([new_predicate])))
    new_description2 = frozenset(
        state_description.union(set([f"not({new_predicate})"])))

    # split active actions between the two new states

    action_dist1 = np.zeros(7)
    action_dist2 = np.zeros(7)

    active_actions = np.where(action_dist > 0)[0]
    np.random.shuffle(active_actions)

    for i in range(len(active_actions)//2):
        action_dist1[active_actions[i]] = action_dist[active_actions[i]]
    for i in range(len(active_actions)//2, len(active_actions)):
        action_dist2[active_actions[i]] = action_dist[active_actions[i]]

    action_dist1 /= np.sum(action_dist1)
    action_dist2 /= np.sum(action_dist2)

    new_generalizations = {
        new_description1: action_dist1, new_description2: action_dist2}
    return new_generalizations


def split_cluster(per_generalization_data):
    # print(per_generalization_data)
    per_state_action_probs = get_per_step_action_probs(
        lambda state: description_to_set(describe_state(state)), [per_generalization_data])

    distances = compute_cross_entropies(per_state_action_probs)

    try:
        clustering = AgglomerativeClustering(
            n_clusters=2, affinity='precomputed', linkage='single').fit(distances)
        all_states = list(per_state_action_probs.keys())
        generalizations = compute_generalizations(
            all_states, clustering.labels_)
    except ValueError as e:
        if distances.shape != (1, 1):
            raise e
        print("Splitting 1 state is not possible, inventing a symbol instead")
        item = next(iter(per_state_action_probs.items()))
        generalization = invent_symbol(*item)
        return generalization

    # print(clustering.labels_)

    # print(generalizations)

    return generalizations


def get_abstract_trajectories(data, generalizations, factory):
    new_data = []
    for ep in data:
        new_ep = []
        for step in ep:
            state_desc = factory.create_state_description(
                list(description_to_set(describe_state(step[0]))))
            applicable_generalizations = state_desc.get_applicable_generalizations(
                generalizations)
            if applicable_generalizations is None:
                raise ValueError("No applicable generalization found")
            # use most specific generalization
            new_ep.append((applicable_generalizations[0], step[1]))
        new_data.append(new_ep)

    return new_data


if __name__ == '__main__':

    from StateDescriptionClustering import StateDescriptionClustering

    factory = StateDescriptionFactory(
        {"is_next_to_wood", "has_wood", "has_planks", "has_chair_parts", "has_chair"})

    actions = ['up', 'down', 'left', 'right', 'craft_planks',
               'craft_chair_parts', 'craft_chair', 'craft_decoration']

    data = pkl.load(open("trajectories_decoration.pkl", "rb"))
    per_state_action_probs = get_per_step_action_probs(lambda state: factory.create_state_description(
        list(description_to_set(describe_state(state)))), data, n_actions=len(actions))

    state_descriptions = list(per_state_action_probs.keys())
    action_probability_vectors = list(per_state_action_probs.values())

    clustering = StateDescriptionClustering(
        state_descriptions, action_probability_vectors)
    clusters = clustering.cluster_state_descriptions(4)

    generalizations = clustering.get_generalizations()
    print(generalizations)

    abstract_data = get_abstract_trajectories(data, generalizations, factory)

    # distances = compute_cross_entropies(per_state_action_probs)

    # print(distances.shape)

    # clustering = AgglomerativeClustering(n_clusters=4, affinity='precomputed', linkage='single').fit(distances)

    # print(clustering.labels_)

    # all_states = list(per_state_action_probs.keys())
    # generalizations = compute_generalizations(all_states,clustering.labels_)
    # print(generalizations)

    # for ep in data[:1]:
    #     for step in ep:
    #         d = description_to_set(describe_state(step[0]))

    #         d_gen = get_most_specific_generalization(generalizations,d)
    #         print(d,d_gen)

    model = fit_transition_model(abstract_data, generalizations)

    print(model.predict(0, 10))

    plot_model(abstract_data, generalizations, model,
               actions=actions, fname='abstract_model.dot')

    clustering.merge_lowest_entropy_clusters(2)
    generalizations = clustering.get_generalizations()
    print('New # of clusters', len(generalizations))
    abstract_data = get_abstract_trajectories(data, generalizations, factory)
    model = fit_transition_model(abstract_data, generalizations)
    plot_model(abstract_data, generalizations, model,
               actions=actions, fname='abstract_model1.dot')

    novelty = (abstract_data[0][-1][0], 6)
    print("-------------------")

    no_novelty = abstract_data[0][-1]
    clustering.add_step(no_novelty[0], no_novelty[1])

    model = fit_transition_model(abstract_data, generalizations)
    plot_model(abstract_data, generalizations, model, actions=actions,
               fname='abstract_model1.dot', from_data=False, clustering=clustering)

    probs = clustering.get_action_probability(novelty[0], novelty[1])

    if all([p < 0.05 for k, p in probs.items()]):
        print("Novelty Detected")
        try:
            clustering.add_cluster(novelty[0])
        except ValueError as e:
            print("nott adding cluster")
            pass
        clustering.add_step(novelty[0], novelty[1])

    probs = clustering.get_action_probability(novelty[0], novelty[1])
    print(probs)

    model = fit_transition_model(abstract_data, generalizations)
    plot_model(abstract_data, generalizations, model,
               actions=actions, fname='abstract_model1.dot')
