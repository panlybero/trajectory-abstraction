from StateDescription import StateDescription, StateDescriptionFactory
from StateCluster import CategoricalStateCluster, StateDependentStateCluster
from scipy.spatial.distance import pdist as pairwise_distance, squareform
import numpy as np
import pickle as pkl
import networkx as nx
from networkx.drawing.nx_agraph import to_agraph
from ClusterCollection import ClusterCollection
from ClusterTransitions import ClusterTransitions
from state_description import describe_state


class AgentModel:
    def __init__(self, n_actions, possible_predicates, prob_threshold, merge_threshold, max_n_clusters=None, cluster_class=CategoricalStateCluster):
        self.clusters = ClusterCollection()
        self.state_description_factory = StateDescriptionFactory(
            possible_predicates)
        self.n_actions = n_actions
        self.prob_threshold = prob_threshold
        self.merge_threshold = merge_threshold
        self.max_n_clusters = max_n_clusters

        self.cluster_transitions = ClusterTransitions()
        self.cluster_class = cluster_class

        self.n_invented = 0

    def get_current_best_cluster_description(self, step):
        x, s, a, next_x, next_s, next_a = step
        generalizations = self.clusters.keys()
        applicable_generalizations = s.get_applicable_generalizations(
            generalizations)

        if applicable_generalizations is None:
            return None
        # check if first fe generalizations differ only in invented predicates
        if len(applicable_generalizations) > 1 and a is not None:
            for i in range(1, len(applicable_generalizations)):
                if not applicable_generalizations[0].compare_predicates(applicable_generalizations[i]):
                    break
            candidates = applicable_generalizations[:i]
            # check probabilities
            # print("Candidates:", candidates, len(candidates),
            # len(applicable_generalizations))
            probs = [self.clusters[c].calculate_step_probability(
                step) for c in candidates]
            # print(probs, step)
            best_cluster = candidates[np.argmax(probs)]
        else:  # len(applicable_generalizations) == 1:
            best_cluster = applicable_generalizations[0]

        return best_cluster

    def _invent_predicate(self):
        invented_predicate = f"(invented_{self.n_invented})"
        invented_not_pred = f"(not {invented_predicate})"
        self.n_invented += 1

        return invented_predicate, invented_not_pred

    def process_step(self, step):
        x, s, a, next_x, next_s, next_a = step

        generalizations = self.clusters.keys()

        most_specific = self.get_current_best_cluster_description(step)

        if most_specific is None:
            prob = -float('inf')
            most_specific = self.state_description_factory.create_state_description([
            ])
        else:
            prob = self.clusters[most_specific].calculate_step_probability(
                step)

        if prob < self.prob_threshold:

            if most_specific.compare_predicates(s):
                if self.clusters[most_specific].is_stable:
                    invented_predicate, invented_not_pred = self._invent_predicate()
                    clust = self.clusters[most_specific]
                    clust.add_invented_predicate(  # this updates the key as well
                        invented_not_pred)

                    # for c in self.clusters.keys():
                    #     self.cluster_transitions[(
                    #         c, clust.state_description)] = self.cluster_transitions.get((c, most_specific), 0)

                    #     # self.cluster_transitions.pop((c, most_specific), None)

                    #     self.cluster_transitions[(
                    #         clust.state_description, c)] = self.cluster_transitions.get((most_specific, c), 0)

                    # self.cluster_transitions.pop((most_specific, c), None)

                    # self.clusters.pop(most_specific)
                    # clust.add_invented_predicate(
                    #     invented_not_pred)
                    # self.clusters[clust.state_description] = clust

                    s.add_invented_predicate(invented_predicate)

                    new_cluster = self.cluster_class(s, self.n_actions)
                    if a is not None:
                        new_cluster.update_distribution(step)

                    self.clusters[s] = new_cluster
                    most_specific = s

                    print("Created new cluster with invented predicate:",
                          self.clusters[s])

                else:
                    self.clusters[most_specific].update_distribution(step)
                    # print("Updating cluster:", most_specific, self.clusters[most_specific].counts)

            else:

                new_cluster = self.cluster_class(s, self.n_actions)
                if a is not None:
                    new_cluster.update_distribution(step)

                self.clusters[s] = new_cluster
                most_specific = s
                # print("Updating cluster:", s, self.clusters[s].counts)
        else:
            self.clusters[most_specific].update_distribution(step)
            # print("Updating cluster:", most_specific, self.clusters[most_specific].counts)

        next_step = (next_x, next_s, next_a, None, None, None)
        most_specific_next = self.get_current_best_cluster_description(
            next_step)

        if most_specific_next is None:
            print("No applicable generalizations for next state", next_s)

            new_cluster = self.cluster_class(next_s, self.n_actions)
            self.clusters[next_s] = new_cluster
            self.cluster_transitions[(most_specific, next_s)] = 1
        else:

            self.cluster_transitions[(most_specific, most_specific_next)] = self.cluster_transitions.get(
                (most_specific, most_specific_next), 0) + 1

        self.merge_clusters()

    def get_generalizations(self, state_description):
        generalizations = [k for k in self.clusters]
        applicable_generalizations = state_description.get_applicable_generalizations(
            generalizations)
        return applicable_generalizations

    def _clean_invented_predicates(self):

        mapping = {}
        invented = []
        for cluster in self.clusters.values():
            invented.extend(cluster.state_description.invented_predicates)
        invented = list(set(invented))

        for p in invented:
            if p.startswith('(not'):
                notp = p[5:-1]
            else:
                notp = f'(not {p})'

            if p in invented and notp in invented:
                invented.remove(p)
                invented.remove(notp)

        mapping = {p: f'(invented_{i})' for i, p in enumerate(invented)}

        for cluster in self.clusters.values():
            inventeds = cluster.state_description.invented_predicates
            new_inventeds = set()
            for p in inventeds:
                if p in mapping:
                    new_inventeds.add(mapping[p])

            cluster.state_description.invented_predicates = new_inventeds

        self.n_invented = len(invented)

    def merge_transitions(self, s1, s2, new_s):
        # print(self.get_cluster_transition_matrix(get_counts=True))
        prev_sum = self.get_cluster_transition_matrix(get_counts=True).sum()

        new_edges = ClusterTransitions()

        for cluster in self.clusters.keys():

            new_edges[(new_s, cluster)] = self.cluster_transitions.get(
                (s1, cluster), 0) + self.cluster_transitions.get((s2, cluster), 0)
            new_edges[(cluster, new_s)] = self.cluster_transitions.get(
                (cluster, s1), 0) + self.cluster_transitions.get((cluster, s2), 0)

        for cluster in self.clusters.keys():
            if (cluster, s1) in self.cluster_transitions:
                self.cluster_transitions.pop((cluster, s1))
            if (cluster, s2) in self.cluster_transitions:
                self.cluster_transitions.pop((cluster, s2))

            if (s1, cluster) in self.cluster_transitions:
                self.cluster_transitions.pop((s1, cluster))
            if (s2, cluster) in self.cluster_transitions:
                self.cluster_transitions.pop((s2, cluster))

        self.cluster_transitions.update(new_edges)

        # print(self.get_cluster_transition_matrix(get_counts=True))
        curr_sum = self.get_cluster_transition_matrix(get_counts=True).sum()

        print('Sums', prev_sum, curr_sum)

        # assert prev_sum == curr_sum

    def merge_clusters(self):
        # compute distance between clusters
        if len(self.clusters) < 2:
            return
        # if not self.max_n_clusters is None:
        #     if self.max_n_clusters < len(self.clusters):
        #         merge_threshold = self.merge_threshold
        #     else:
        merge_threshold = self.merge_threshold

        # cluster_distributions = [v.counts for k,v in self.clusters.items()]
        cluster_idx = {i: k for i, k in enumerate(self.clusters.keys())}
        clusters = list(self.clusters.values())

        dist_matrix = self.cluster_class.pairwise_distances(clusters)

        # make diagonal elements inf
        np.fill_diagonal(dist_matrix, np.inf)

        # do not merge clusters with no actions
        for i, k in enumerate(self.clusters.keys()):
            if not self.clusters[k].is_fitted:
                dist_matrix[i, :] = np.inf
                dist_matrix[:, i] = np.inf

        # print(dist_matrix)
        # find closest clusters
        min_i, min_j = np.unravel_index(
            np.argmin(dist_matrix), dist_matrix.shape)
        min_dist = dist_matrix[min_i, min_j]
        # print(dist_matrix)
        # print("max dist:",max_dist)

        if min_dist < merge_threshold:
            print(self.get_cluster_transition_matrix(
                get_counts=True).sum(), 'n_clusters', len(self.clusters))
            print("Merging clusters with distance:", min_dist)

            # merge clusters
            cluster_i = self.clusters[cluster_idx[min_i]]
            cluster_j = self.clusters[cluster_idx[min_j]]
            print("Cluster i:", cluster_i.state_description)
            print("Cluster j:", cluster_j.state_description)

            trans_mat = self.get_cluster_transition_matrix(
                get_counts=True)
            # print("i")
            # print(trans_mat[min_i, :].sum(), trans_mat[:, min_i].sum())
            # print("j")
            # print(trans_mat[min_j, :].sum(), trans_mat[:, min_j].sum())

            new_cluster = self.cluster_class.merge([cluster_i, cluster_j])
            new_cluster_description = new_cluster.state_description

            # and not new_cluster.state_description in [cluster_i.state_description, cluster_j.state_description]:
            if new_cluster.state_description in self.clusters.keys():

                invented_predicate, invented_not_pred = self._invent_predicate()
                og_cluster_description = new_cluster.state_description.copy()
                print(self.cluster_transitions.get_transitions(
                    og_cluster_description))
                existing_cluster = self.clusters[new_cluster.state_description]
                existing_cluster.add_invented_predicate(
                    invented_not_pred)
                new_cluster.add_invented_predicate(
                    invented_predicate)

                new_cluster_description = new_cluster.state_description
                print("OG", og_cluster_description)
                print("Existing_modded", existing_cluster.state_description)
                print("New", new_cluster_description)

                print(self.cluster_transitions.get_transitions(
                    existing_cluster.state_description))

                # for c in self.clusters.keys():
                #     self.cluster_transitions[(
                #         c, existing_cluster.state_description)] = self.cluster_transitions.get((c, og_cluster_description), 0)
                #     self.cluster_transitions[(
                #         existing_cluster.state_description, c)] = self.cluster_transitions.get((og_cluster_description, c), 0)

                #     print(self.cluster_transitions.get((
                #         c, existing_cluster.state_description)), self.cluster_transitions.get((
                #             c, og_cluster_description)))

                #     print(self.cluster_transitions.get((
                #         existing_cluster.state_description, c)), self.cluster_transitions.get((
                #             og_cluster_description, c)))

                # self.cluster_transitions.pop(
                #     (c, og_cluster_description), None)
                # self.cluster_transitions.pop(
                #     (og_cluster_description, c), None)

                print(
                    "######################## Merged cluster already exists, inventing predicate", invented_predicate)

            self.clusters[new_cluster_description] = new_cluster
            print("added cluster:", new_cluster.state_description)

            self.merge_transitions(
                cluster_i.state_description, cluster_j.state_description, new_cluster_description)
            print(self.cluster_transitions.get_transitions(
                new_cluster_description))

            print("Merged cluster:", new_cluster.state_description)
            if cluster_i.state_description != new_cluster_description:
                print('popping', cluster_i.state_description)
                self.clusters.pop(cluster_i.state_description)
            if cluster_j.state_description != new_cluster_description:
                print('popping', cluster_j.state_description)
                self.clusters.pop(cluster_j.state_description)

            # input()

            self.merge_clusters()  # recursively merge clusters
        # self._clean_invented_predicates()
        else:
            print("No clusters to merge")

    def relabel_trajectory(self, trajectory):
        new_trajectory = []
        for x, s, a, next_x, next_s in trajectory:
            generalizations = [v.state_description for k,
                               v in self.clusters.items()]
            applicable_generalizations = s.get_applicable_generalizations(
                generalizations)
            most_specific = applicable_generalizations[0]
            new_trajectory.append((x, most_specific, a))
        return new_trajectory

    def get_state_sequence(self, trajectory, as_int=True):
        seq = []
        indeces = {k: i for i, k in enumerate(self.clusters)}
        for x, s, a in trajectory:
            if as_int:
                seq.append(indeces[s])
            else:
                seq.append(s)
        return seq

    @property
    def n_clusters(self):
        return len(self.clusters)

    def get_cluster_counts(self):
        return {k: v.counts for k, v in self.clusters.items()}

    def get_cluster_transition_matrix(self, get_counts=False):
        matrix = np.zeros((self.n_clusters, self.n_clusters))
        descriptions = self.clusters.keys()
        for i in range(self.n_clusters):
            for j in range(self.n_clusters):
                matrix[i, j] = self.cluster_transitions.get(
                    (descriptions[i], descriptions[j]), 0)
        # normalize
        if not get_counts:
            matrix = matrix + 1
            matrix = matrix / np.sum(matrix, axis=1, keepdims=True)

        return matrix

    def compute_step_likelihood(self, step):
        trans_mat = self.get_cluster_transition_matrix()

        x, s, a, next_x, next_s, next_a = step
        most_specific = self.get_current_best_cluster_description(step)
        next_step = (next_x, next_s, next_a, None, None, None)
        most_specific_next = self.get_current_best_cluster_description(
            next_step)

        curr_clst_ind = list(self.clusters.keys()).index(most_specific)
        next_clst_ind = list(self.clusters.keys()).index(
            most_specific_next)
        trans_prob = trans_mat[curr_clst_ind, next_clst_ind]
        trans_lik = np.log(trans_prob)

        emission_prob = self.clusters[most_specific].calculate_step_probability(
            step)
        emission_lik = np.log(emission_prob)

        step_lik = trans_lik + emission_lik
        return step_lik

    def compute_trajectory_likelihood(self, trajectory):
        trans_mat = self.get_cluster_transition_matrix()
        traj_lik = 0
        for step in trajectory:

            traj_lik += self.compute_step_likelihood(step)
        return traj_lik

    def get_next_state(self, s):

        generalizations = [k for k in self.clusters]
        # print(s)
        # print(generalizations)
        applicable_generalizations = s.get_applicable_generalizations(
            generalizations)
        if applicable_generalizations is None:
            raise ValueError(
                "Cannot predict next state if current state has no applicable generalizations")

        most_specific = applicable_generalizations[0]

        indx = generalizations.index(most_specific)

        trans_matrix = self.get_cluster_transition_matrix(get_counts=True)
        counts = trans_matrix[indx]+1
        counts[indx] = 0
        probs = counts / np.sum(counts)

        next_state_idx = np.random.choice(np.arange(self.n_clusters), p=probs)
        next_state = generalizations[next_state_idx]

        return next_state

    def get_cluster_trajectory(self, start_state, n_steps):
        trajectory = [start_state]
        for i in range(n_steps):
            state = self.get_next_state(trajectory[-1])
            trajectory.append(state)
        return trajectory

    def plot_transition_graph(self, actions, fname='file.dot'):

        # Create a directed graph
        G = nx.DiGraph()

        # Add nodes to the graph
        for i in range(self.n_clusters):
            G.add_node(i)

        count_matrix = self.get_cluster_transition_matrix(get_counts=True)
        trans_matrix = self.get_cluster_transition_matrix()

        # Add edges to the graph with transition probabilities as labels
        for i in range(self.n_clusters):
            for j in range(self.n_clusters):

                count = count_matrix[i, j]
                prob = trans_matrix[i, j]
                prob = np.round(prob, decimals=3)
                if count > 0:
                    G.add_edge(i, j, weight=prob)

        for u, v, d in G.edges(data=True):
            d['label'] = f"{d.get('weight',''):.3f}"

        action_strs = {}

        for k, v in self.clusters.items():
            action_strs[k] = v.cluster_str(actions)

        # make node_labels
        node_labels = {
            i: str(g)+'\n'+"".join(action_strs[g]) for i, g in enumerate(self.clusters.keys())}

        count_matrix = self.get_cluster_transition_matrix(get_counts=True)

        sink_states = count_matrix.sum(axis=1) == 0

        for i in range(self.n_clusters):
            G.nodes[i]['label'] = node_labels[i]
            if sink_states[i]:
                G.nodes[i]['color'] = 'red'
                G.remove_edges_from([(i, j) for j in range(self.n_clusters)])

        pos = nx.circular_layout(G)

        A = to_agraph(G)
        A.layout('dot')

        A.draw('multi.png')
        A.write(fname)


def plot_model(agent_model, markov_model, fname='file.dot', actions=['up', 'down', 'left', 'right', 'craft_planks', 'craft_chair_parts', 'craft_chair']):

    cluster_counts = agent_model.get_cluster_counts()
    generalizations = list(agent_model.clusters.keys())
    per_set_action_probs = {
        k: v.counts/np.sum(v.counts) for k, v in agent_model.clusters.items()}

    per_set_action_probs = {k: [(actions[i], f"{v:.2f}") for i, v in enumerate(
        per_set_action_probs[k])] for k in per_set_action_probs}
    print(per_set_action_probs)

    node_labels = {i: str(
        g)+'\n'+str(per_set_action_probs[g]) for i, g in enumerate(generalizations)}
    markov_model.plot_transition_graph(node_labels=node_labels, fname=fname)


def prepare_step(model, step):

    obs = step[0]
    action = step[1]
    new_obs = step[3]
    new_action = step[4]
    described = describe_state(obs)
    new_described = describe_state(new_obs)

    # print(described)
    state = model.state_description_factory.create_state_description_from_dict(
        described)
    next_state = model.state_description_factory.create_state_description_from_dict(
        new_described)

    step = (obs, state, action, new_obs, next_state, new_action)
    return step


if __name__ == '__main__':

    actions = ['up', 'down', 'left', 'right', 'craft_planks',
               'craft_chair_parts', 'craft_chair', 'craft_decoration', 'craft_stick']
    possible_predicates = ['(next_to wood)', '(has wood)', '(has planks)',
                           '(has chair_parts)', '(has chair)', '(has decoration)', '(has stick)']
    model = AgentModel(len(actions), possible_predicates, 0.05,
                       0.1, cluster_class=CategoricalStateCluster)

    # data = pkl.load(open("trajectories_decoration.pkl","rb"))
    # for trajectory in data:
    #     described_trajectory = []
    #     for step in trajectory:
    #         step = prepare_step(model, step)

    #         model.process_step(step)
    #         print("N of clusters",model.n_clusters)

    # pkl.dump(model,open("agent_model_decoration.pkl","wb"))

    # model.plot_transition_graph(actions=actions,fname='abstract_model1.dot')

    data = pkl.load(open("results/trajectories_decoration_or_stick.pkl", "rb"))

    # described_data = []
    for i, trajectory in enumerate(data):
        described_trajectory = []
        for step in trajectory:
            step = prepare_step(model, step)
            model.process_step(step)

        if i == 51:
            print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")
            model.plot_transition_graph(
                actions=actions, fname='results/abstract_model0.dot')
        if i == 52:
            model.plot_transition_graph(
                actions=actions, fname='results/abstract_model1.dot')
            print(model.get_cluster_transition_matrix(get_counts=True))
            exit()

    model.merge_clusters()

    traj_liks = []
    for trajectory in data:
        lik = 0
        for step in trajectory:
            step = prepare_step(model, step)
            lik += model.compute_step_likelihood(step)
        traj_liks.append(lik)

    print("Avg lik", np.mean(traj_liks))

    model.plot_transition_graph(
        actions=actions, fname='results/abstract_model2.dot')

    print(model.get_cluster_transition_matrix(get_counts=True))
    print(model.clusters.keys())
