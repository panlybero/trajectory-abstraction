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
    def __init__(self, n_actions, possible_predicates, prob_threshold, merge_threshold, max_n_clusters=None, cluster_class=CategoricalStateCluster, invent_predicates=True):
        self.clusters = ClusterCollection()
        self.state_description_factory = StateDescriptionFactory(
            possible_predicates)
        self.n_actions = n_actions
        self.prob_threshold = prob_threshold
        self.merge_threshold = merge_threshold
        self.max_n_clusters = max_n_clusters

        self.cluster_transitions = ClusterTransitions()
        self.cluster_class = cluster_class
        self.invent_predicates = invent_predicates
        self.n_invented = 0

        self.inferred_invented_predicates = set()

        self.goals = []

    def reset_invented_inference(self):
        self.inferred_invented_predicates = set()

    def reset(self):
        self.reset_invented_inference()

       # print(self.goals)
        self.goals = []

    def get_current_best_cluster_description(self, step, invented_predicates=set()):
        x, s, a, next_x, next_s, next_a = step
        generalizations = self.clusters.keys()
        applicable_generalizations = s.get_applicable_generalizations(
            generalizations, invented_predicates=invented_predicates)

        if applicable_generalizations is None:
            return None
        # check if first fe generalizations differ only in invented predicates
        if len(applicable_generalizations) > 1 and a:
            for i in range(1, len(applicable_generalizations)):
                if not applicable_generalizations[0].compare_predicates(applicable_generalizations[i]):
                    break
            candidates = applicable_generalizations[:i+1]
            # check probabilities
            # #print("Candidates:", candidates, len(candidates),
            # len(applicable_generalizations))
            if a is not None:
                probs = [self.clusters[c].calculate_step_probability(
                    step) for c in candidates]
                # #print(probs, step)
                best_cluster = candidates[np.argmax(probs)]
            else:
                best_cluster = candidates[0]

        else:
            best_cluster = applicable_generalizations[0]

        return best_cluster

    def _invent_predicate(self):
        invented_predicate = f"(invented_{self.n_invented})"
        invented_not_pred = f"(not {invented_predicate})"
        self.n_invented += 1
        return invented_predicate, invented_not_pred

    def infer_invented(self, step):
        most_specific = self.get_current_best_cluster_description(step)
        curr_invented = most_specific.invented_predicates

        for inv in curr_invented:
            if inv.startswith('(not'):

                notinv = inv[5:-1]
            else:
                notinv = f'(not {inv})'

            if notinv in self.inferred_invented_predicates:
                self.inferred_invented_predicates.remove(notinv)

            self.inferred_invented_predicates.add(inv)

        if self.inferred_invented_predicates and curr_invented:
            pass
            # print("Inferred invented predicates:",
            # self.inferred_invented_predicates, 'at', most_specific)

    def process_state_action(self, step, update_probs=True):
        x, s, a, next_x, next_s, next_a = step
        most_specific = self.get_current_best_cluster_description(
            step, self.inferred_invented_predicates)

        if most_specific is None:

            prob = -float('inf')
            most_specific = self.state_description_factory.create_state_description([
            ])

        elif a is not None:
            prob = self.clusters[most_specific].calculate_step_probability(
                step)
        else:
            return most_specific

        # If we can compute the probability, we can identify the cluster better and update it.
        cluster_to_update = None

        if prob < self.prob_threshold:
            # are the same observable predicates

            if most_specific.compare_predicates(s):
                if self.clusters[most_specific].is_stable and self.invent_predicates:
                    invented_predicate, invented_not_pred = self._invent_predicate()
                    clust = self.clusters[most_specific]
                    clust.add_invented_predicate(  # this updates the key as well
                        invented_not_pred)

                    s.add_invented_predicate(invented_predicate)

                    new_cluster = self.cluster_class(s, self.n_actions)
                    cluster_to_update = new_cluster

                    self.clusters[s] = new_cluster
                    most_specific = s

                    # print("Created new cluster with invented predicate:",
                    #      self.clusters[s])

                else:
                    cluster_to_update = self.clusters[most_specific]

                    # #print("Updating cluster:", most_specific, self.clusters[most_specific].counts)

            else:

                new_cluster = self.cluster_class(s, self.n_actions)
                self.clusters[s] = new_cluster
                most_specific = s

                cluster_to_update = new_cluster

                # print("Created new cluster:", s)

            # if a is not None:
            #     cluster_to_update.update_distribution(step)
        else:
            cluster_to_update = self.clusters[most_specific]

        if a is not None and update_probs:
            cluster_to_update.update_distribution(step)

            # #print("Updating cluster:", most_specific, self.clusters[most_specific].counts)

        return most_specific

    def process_step(self, step):
        x, s, a, next_x, next_s, next_a = step
        created_new = False
        generalizations = self.clusters.keys()

        current_state_most_specific = self.process_state_action(step)
        next_step = (next_x, next_s, next_a, None, None, None)
        next_state_most_specific = self.process_state_action(
            next_step, update_probs=False)

        '''next_step = (next_x, next_s, next_a, None, None, None)
        most_specific_next = self.get_current_best_cluster_description(
            next_step, self.inferred_invented_predicates)

        # check anomaly in next step. If so, make a new cluster for it
        if not most_specific_next is None and not next_a is None:
            prob = self.clusters[most_specific_next].calculate_step_probability(
                next_step)

            if prob < self.prob_threshold:
                #print(next_s)
                #print(most_specific_next, prob)
                #print('Generalization too unlikely, creating new cluster',
                      most_specific_next)
                most_specific_next = None

        if most_specific_next is None:
            #print("No applicable generalizations for next state", next_s)

            new_cluster = self.cluster_class(next_s, self.n_actions)
            self.clusters[next_s] = new_cluster
            self.cluster_transitions[(most_specific, next_s)] = 1
        else:'''

        self.cluster_transitions[(current_state_most_specific, next_state_most_specific)] = self.cluster_transitions.get(
            (current_state_most_specific, next_state_most_specific), 0) + 1

        # if created_new:
        #     model.plot_transition_graph(
        #         actions=actions, fname='results/abstract_model0.dot')

        self.merge_clusters()
        self.cluster_transitions.drop_deprecated_transitions(
            cluster_descriptions=self.clusters.keys())

        if self.invent_predicates:
            self.infer_invented(step)
            self._clean_invented_predicates()

            # #print("Inferred invented predicates:",
            #       self.inferred_invented_predicates)

    def get_generalizations(self, state_description):
        generalizations = [k for k in self.clusters.keys()]
        applicable_generalizations = state_description.get_applicable_generalizations(
            generalizations, invented_predicates=self.inferred_invented_predicates)
        return applicable_generalizations

    def _clean_invented_predicates(self):

        mapping = {}
        invented = []
        for cluster in self.clusters.values():
            invented.extend(cluster.state_description.invented_predicates)
        invented = list(set(invented))

        c = 0
        for p in invented:
            if p.startswith('(not'):
                negp = p
                p = p[5:-1]
            else:
                negp = f'(not {p})'

            if p in invented and negp in invented:

                mapping[p] = f'(invented_{c})'
                mapping[negp] = f'(not (invented_{c}))'
                c += 1

        # for c1 in self.clusters.values():
        #     for c2 in self.clusters.values():
        #         if not c1.state_description.compare_predicates(c2.state_description):
        #             continue

        #         inv1 = c1.state_description.invented_predicates
        #         inv2 = c2.state_description.invented_predicates
        #         neginv1 = set([f'(not {p})' if not p.startswith(
        #             '(not') else p[5:-1] for p in list(inv1)])
        #         if inv2 == neginv1:
        #             for p in inv1:
        #                 if p in mapping:
        #                     negp = f'(not {p})' if not p.startswith(
        #                         '(not') else p[5:-1]

        #                     mapped_p = mapping[p]
        #                     mapped_neg_p = mapping[negp]
        #                     break
        #             for p in inv1:
        #                 if p in mapping:
        #                     negp = f'(not {p})' if not p.startswith(
        #                         '(not') else p[5:-1]

        #                     mapping[p] = mapped_p
        #                     mapping[negp] = mapped_neg_p
        #                    # c += 1

        for cluster in self.clusters.values():
            inventeds = cluster.state_description.invented_predicates
            new_inventeds = set()
            for p in inventeds:
                if p in mapping:
                    new_inventeds.add(mapping[p])

            cluster.state_description.invented_predicates = new_inventeds

        self.n_invented = len(invented)

        new_inferred = set()
        for p in self.inferred_invented_predicates:
            if p in mapping:
                new_inferred.add(mapping[p])
        self.inferred_invented_predicates = new_inferred

    def merge_transitions(self, s1, s2, new_s):
        # #print(self.cluster_transitions.get_transition_matrix(get_counts=True))
        prev_sum = self.cluster_transitions.get_transition_matrix(self.clusters,
                                                                  get_counts=True).sum()

        # new_edges = ClusterTransitions()

        for cluster in self.clusters.keys():
           # #print("Outgoing")
            a = self.cluster_transitions.pop(
                (s1, cluster), 0)
            b = self.cluster_transitions.pop((s2, cluster), 0)
            c = self.cluster_transitions.pop((new_s, cluster), 0)
            self.cluster_transitions[(new_s, cluster)] = a + b + c

            # #print(a, b, c, '=', a+b+c)
            # #print("Incoming")
            a = self.cluster_transitions.pop(
                (cluster, s1), 0)
            b = self.cluster_transitions.pop((cluster, s2), 0)
            c = self.cluster_transitions.pop((cluster, new_s), 0)
            self.cluster_transitions[(cluster, new_s)] = a + b + c
            # #print(a, b, c, '=', a+b+c)

        # self.cluster_transitions.update(new_edges)

        # #print(self.cluster_transitions.get_transition_matrix(get_counts=True))
        curr_sum = self.cluster_transitions.get_transition_matrix(self.clusters,
                                                                  get_counts=True).sum()

        # print('Sums', prev_sum, curr_sum)

        # assert prev_sum == curr_sum
    def remove_cluster_and_cleanup(self, cluster_description):

        clust = self.clusters.pop(cluster_description)

        # remove transitions to and from cluster
        for pair in list(self.cluster_transitions.keys()):
            if cluster_description in pair:
                self.cluster_transitions.pop(pair)

    def merge_clusters(self):
        # compute distance between clusters
        if len(self.clusters) < 2:
            return

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

        # #print(dist_matrix)
        # find closest clusters
        min_i, min_j = np.unravel_index(
            np.argmin(dist_matrix), dist_matrix.shape)
        min_dist = dist_matrix[min_i, min_j]
        # #print(dist_matrix)
        # #print("max dist:",max_dist)

        if not self.max_n_clusters is None:
            if self.max_n_clusters < len(self.clusters):
                merge_threshold = min_dist
            else:
                merge_threshold = self.merge_threshold

        if min_dist <= merge_threshold:
            # print(self.cluster_transitions.get_transition_matrix(self.clusters,
            #                                                     get_counts=True).sum(), 'n_clusters', len(self.clusters))
            # print("Merging clusters with distance:", min_dist)

            # merge clusters
            cluster_i = self.clusters[cluster_idx[min_i]]
            cluster_j = self.clusters[cluster_idx[min_j]]
            # print("Cluster i:", cluster_i.state_description)
            # print("Cluster j:", cluster_j.state_description)

            new_cluster = self.cluster_class.merge([cluster_i, cluster_j])
            new_cluster_description = new_cluster.state_description

            # and not new_cluster.state_description in [cluster_i.state_description, cluster_j.state_description]:
            cluster_description_already_exists = any([new_cluster.state_description ==
                                                      c.state_description for c in self.clusters.values()])
            cluster_description_already_exists_barring_invented = any(
                [new_cluster.state_description.compare_predicates(c.state_description) for c in self.clusters.values()])

            if cluster_description_already_exists and not new_cluster.state_description in [cluster_i.state_description, cluster_j.state_description] and self.invent_predicates and self.clusters[new_cluster.state_description].is_stable:

                invented_predicate, invented_not_pred = self._invent_predicate()
                og_cluster_description = new_cluster.state_description.copy()

                existing_cluster = self.clusters[new_cluster.state_description]
                existing_cluster.add_invented_predicate(
                    invented_not_pred)
                new_cluster.add_invented_predicate(
                    invented_predicate)

                new_cluster_description = new_cluster.state_description
                # print("OG", og_cluster_description)
                # print("Existing_modded", existing_cluster.state_description)
                # print("New", new_cluster_description)
                # print(
                # "######################## Merged cluster already exists, inventing predicate", invented_predicate)
                # input()

            # cluster description exists barring invented predicates
            elif cluster_description_already_exists_barring_invented:

                # print("Cluster description already exists barring invented predicates")

                # Two cases: if we are not using invented predicates, we can simply merge the clusters.
                # if we are using invented predicates, treat this as the 'default' behavior. It will either be merged into
                # one of the other clusters with invented later, or it will remain as is.
                if not self.invent_predicates:
                    # Merge the new clsuter with the existing one. Since they have the same description, transitions will be merged as normal
                    clust_i = self.clusters[new_cluster_description]
                    clust_j = new_cluster
                    new_cluster = self.cluster_class.merge([clust_i, clust_j])

            self.clusters[new_cluster_description] = new_cluster
            #    #print("added cluster:", new_cluster.state_description)
            # print("n incoming", len(
            # self.cluster_transitions.get_transitions_in(new_cluster_description)))
            # print("n outgoing", len(
            # self.cluster_transitions.get_transitions_out(new_cluster_description)))

            # print("n incoming", len(self.cluster_transitions.get_transitions_in(cluster_i.state_description)),
            # len(self.cluster_transitions.get_transitions_in(cluster_j.state_description)))
            # print("n outgoing", len(self.cluster_transitions.get_transitions_out(cluster_i.state_description)), len(
            # self.cluster_transitions.get_transitions_out(cluster_j.state_description)))

            self.merge_transitions(
                cluster_i.state_description, cluster_j.state_description, new_cluster_description)
            # #print(self.cluster_transitions.get_transitions_in(
            #     new_cluster_description))
            # print("n incoming", len(
            #    self.cluster_transitions.get_transitions_in(new_cluster_description)))
            # print("n outgoing", len(
            #    self.cluster_transitions.get_transitions_out(new_cluster_description)))
            names = self.clusters._cluster_names()
            # #print(names[new_cluster_description])
            # #print(self.cluster_transitions._named_transitions(names))

            # print("Merged cluster:", new_cluster.state_description)
            if cluster_i.state_description != new_cluster_description:
                # print('popping', cluster_i.state_description)
                self.remove_cluster_and_cleanup(cluster_i.state_description)
            if cluster_j.state_description != new_cluster_description:
                # print('popping', cluster_j.state_description)
                self.remove_cluster_and_cleanup(cluster_j.state_description)

            # input()

            self.merge_clusters()  # recursively merge clusters

        else:
            pass  # print("No clusters to merge")

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

    def compute_step_likelihood(self, step):
        trans_mat = self.cluster_transitions.get_transition_matrix(
            self.clusters)

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
        trans_mat = self.cluster_transitions.get_transition_matrix(
            self.clusters)
        traj_lik = 0
        for step in trajectory:

            traj_lik += self.compute_step_likelihood(step)
        return traj_lik

    def get_next_state(self, s, use_invented=False):

        generalizations = [k for k in self.clusters.keys()]
        # #print(s)
        # #print(generalizations)
        applicable_generalizations = s.get_applicable_generalizations(
            generalizations)
        if applicable_generalizations is None:
            raise ValueError(
                "Cannot predict next state if current state has no applicable generalizations")

        if use_invented:
            including_invented = []
            for g in applicable_generalizations:
                if not g._check_contradiction_in_invented(self.inferred_invented_predicates):
                    including_invented.append(g)

            applicable_generalizations = including_invented

        most_specific = applicable_generalizations[0]
        curr_state = most_specific

        indx = generalizations.index(curr_state)

        trans_matrix = self.cluster_transitions.get_transition_matrix(self.clusters,
                                                                      get_counts=True)
        counts = trans_matrix[indx]+1
        new_counts = np.zeros_like(counts)
        for i, c in enumerate(self.clusters):
            if len(c.state_description.invented_predicates) > 0 and not c.state_description._check_contradiction_in_invented(self.inferred_invented_predicates):
                new_counts[i] = counts[i]

            if c.state_description._check_contradiction_in_invented(self.inferred_invented_predicates):
                counts[i] = 0

                # print(self.inferred_invented_predicates,
                #      'zeros out', c.state_description)
            if c.state_description.subsumes(s):
                counts[i] = 0
                # print(s, 'subsumes', c.state_description)
        # if sum(new_counts) > 0:
        #    counts = new_counts
        indx = generalizations.index(curr_state)
        counts[indx] = 0

        if counts.sum() == 0:
            self.goals.append("randomizing next state")
            probs = np.ones(len(counts))/len(counts)
        else:

            probs = counts / np.sum(counts)

        next_state_idx = np.random.choice(
            np.arange(self.n_clusters), p=probs)

        next_state = generalizations[next_state_idx]
        self.goals.append(next_state)

        return next_state

    def get_stationary_distribution(self):
        transition_matrix = self.cluster_transitions.get_transition_matrix(
            self.clusters)
        transition_matrix_transp = transition_matrix.T
        eigenvals, eigenvects = np.linalg.eig(transition_matrix_transp)
        '''
        Find the indexes of the eigenvalues that are close to one.
        Use them to select the target eigen vectors. Flatten the result.
        '''
        close_to_1_idx = np.isclose(eigenvals, 1)
        target_eigenvect = eigenvects[:, close_to_1_idx]
        target_eigenvect = target_eigenvect[:, 0]
        # Turn the eigenvector elements into probabilites
        stationary_distrib = target_eigenvect / sum(target_eigenvect)

        d = {k: stationary_distrib[i]
             for i, k in enumerate(self.clusters.keys())}

        print(d)
        return d

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

        count_matrix = self.cluster_transitions.get_transition_matrix(self.clusters,
                                                                      get_counts=True)
        trans_matrix = self.cluster_transitions.get_transition_matrix(
            self.clusters)

        # Add edges to the graph with transition probabilities as labels
        for i in range(self.n_clusters):
            for j in range(self.n_clusters):

                count = count_matrix[i, j]
                prob = trans_matrix[i, j]
                prob = np.round(prob, decimals=5)
                if count > 0:
                    G.add_edge(i, j, weight=prob)

        for u, v, d in G.edges(data=True):
            d['label'] = f"{d.get('weight',''):.3f}"

        action_strs = {}

        for k, v in self.clusters.items():
            action_strs[k] = v.cluster_str(actions)
        names = self.clusters._cluster_names()
        # make node_labels
        node_labels = {
            i: names[g]+"\n"+str(g)+'\n'+"".join(action_strs[g]) for i, g in enumerate(self.clusters.keys())}

        count_matrix = self.cluster_transitions.get_transition_matrix(self.clusters,
                                                                      get_counts=True)

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
    # print(per_set_action_probs)

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

    # #print(described)
    state = model.state_description_factory.create_state_description_from_dict(
        described)
    next_state = model.state_description_factory.create_state_description_from_dict(
        new_described)

    step = (obs, state, action, new_obs, next_state, new_action)
    return step


actions = ['up', 'down', 'left', 'right', 'craft_planks',
           'craft_chair_parts', 'craft_chair', 'craft_decoration', 'craft_stick']
if __name__ == '__main__':

    possible_predicates = ['(next_to wood)', '(has wood)', '(has planks)',
                           '(has chair_parts)', '(has chair)', '(has decoration)', '(has stick)']
    model = AgentModel(len(actions), possible_predicates, 0.05,
                       0.1, cluster_class=CategoricalStateCluster, invent_predicates=True)

    # data = pkl.load(open("trajectories_decoration.pkl","rb"))
    # for trajectory in data:
    #     described_trajectory = []
    #     for step in trajectory:
    #         step = prepare_step(model, step)

    #         model.process_step(step)
    #         #print("N of clusters",model.n_clusters)

    # pkl.dump(model,open("agent_model_decoration.pkl","wb"))

    # model.plot_transition_graph(actions=actions,fname='abstract_model1.dot')

    data = pkl.load(open("results/trajectories_decoration_or_stick.pkl", "rb"))

    # described_data = []
    for i, trajectory in enumerate(data):
        described_trajectory = []
        for j, step in enumerate(trajectory):
            step = prepare_step(model, step)
            model.process_step(step)

        if i == 49:
            # print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")
            model.plot_transition_graph(
                actions=actions, fname='results/abstract_model0.dot')
        if i == 50:
            model.plot_transition_graph(
                actions=actions, fname='results/abstract_model1.dot')
            # print(model.get_transition_matrix(
            #    model.clusters, get_counts=True))
            # print(model.cluster_transitions._named_transitions(
            #    model.clusters._cluster_names()))

    # model.merge_clusters()
    # model._clean_invented_predicates()

    model.get_stationary_distribution()
    traj_liks = []
    for trajectory in data:
        lik = 0
        for step in trajectory:
            step = prepare_step(model, step)
            lik += model.compute_step_likelihood(step)
        traj_liks.append(lik)

    # print("Avg lik", np.mean(traj_liks))

    model.plot_transition_graph(
        actions=actions, fname='results/abstract_model2.dot')

    pkl.dump(model, open("agent_model_decoration_or_stick.pkl", "wb"))
