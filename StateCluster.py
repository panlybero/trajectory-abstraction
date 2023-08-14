import numpy as np
from abc import ABC, abstractmethod
from StateDescription import StateDescription
import scipy.spatial as spatial
from OnlineNearestCentroid import OnlineNearestCentroid


class StateCluster(ABC):
    def __init__(self, state_description):
        self.state_description = state_description
        self._is_stable = True
        self.consistent_with_infered = set()

    def is_consistent_with_infered(self, infered):
        for i in infered:
            if i not in self.consistent_with_infered:
                return False
        return True

    @abstractmethod
    def calculate_step_probability(self, step):
        pass

    @abstractmethod
    def get_next_action(self, step):
        pass

    @abstractmethod
    def update_distribution(self, step):
        pass

    @abstractmethod
    def cluster_str(self, actions):
        pass

    def add_invented_predicate(self, predicate):
        self.state_description.add_invented_predicate(predicate)

    def add_consistent_with_infered(self, predicates):
        for predicate in predicates:
            self.consistent_with_infered.add(predicate)

    @property
    def is_stable(self):
        return self._is_stable


class CategoricalStateCluster(StateCluster):
    def __init__(self, state_description, n_actions, stability_count=10, smoothing=1):
        super().__init__(state_description)
        self.n_actions = n_actions
        self.counts = np.zeros(n_actions)
        self._is_stable = False
        self.stability_count = stability_count
        self.smoothing = smoothing

    def calculate_step_probability(self, step):
        _, _, action_index, _, _, _ = step

        counts = self.counts + self.smoothing
        action_probabilities = counts / np.sum(counts)

        return action_probabilities[action_index]

    def get_next_action(self, info):

        counts = self.counts + self.smoothing
        action_probabilities = counts / np.sum(counts)
        next_action = np.random.choice(self.n_actions, p=action_probabilities)
        return next_action

    def update_distribution(self, step):
        _, _, action_index, _, _, _ = step
        self.counts[action_index] += 1

    @classmethod
    def merge(cls, state_cluster_list):
        n_actions = state_cluster_list[0].n_actions
        descriptions = [
            state_cluster.state_description for state_cluster in state_cluster_list]
        new_cluster_description = StateDescription.generalize(
            descriptions)
        new_cluster = cls(new_cluster_description, n_actions)

        for state_cluster in state_cluster_list:
            new_cluster.counts += state_cluster.counts

        union = set()
        for state_cluster in state_cluster_list:
            union = union.union(state_cluster.consistent_with_infered)
        new_cluster.consistent_with_infered = union
        # print(state_cluster.counts)
        # print(new_cluster.counts)
        return new_cluster

    @classmethod
    def compute_distance(self, c1, c2):
        count1 = c1.counts + c1.smoothing
        count1 = count1/np.sum(count1)
        count2 = c2.counts + c2.smoothing
        c2 = count2/np.sum(count2)
        dist = spatial.distance.jensenshannon(count1, count2)
        return dist

    @classmethod
    def pairwise_distances(self, clist):
        distances = np.zeros((len(clist), len(clist)))
        for i in range(len(clist)):
            for j in range(i+1, len(clist)):
                distances[i, j] = self.compute_distance(clist[i], clist[j])
                distances[j, i] = distances[i, j]
        return distances

    @property
    def is_fitted(self):
        return np.sum(self.counts) > 0

    @property
    def dist(self):
        c = self.counts+1
        return c/np.sum(c)

    @property
    def is_stable(self):
        if np.sum(self.counts) > self.stability_count:
            self._is_stable = True
        return self._is_stable

    def cluster_str(self, actions):
        d = self.dist
        vals = {actions[i]: f"{d[i]:0.2f}" for i in range(len(actions))}
        return f"CategoricalCluster({vals})"


class StateDependentStateCluster(StateCluster):
    def __init__(self, state_description, n_actions, memory_size=100, classifier_class=OnlineNearestCentroid, stability_count=10):
        super().__init__(state_description)
        self.n_actions = n_actions
        self.classifier = classifier_class(
            n_actions) if classifier_class is not None else None
        self._classifier_class = classifier_class
        self.memory_bank = []
        self.memory_size = memory_size

    def calculate_step_probability(self, step):
        x, s, action_index, _, _ = step
        if type(x) is dict:
            x = np.array([np.concatenate(list(x.values()))])
        action_probabilities = self.classifier.predict_proba(x)[0]

        return action_probabilities[action_index]

    def get_next_action(self, info):
        x = info
        if type(x) is dict:
            x = np.array([np.concatenate(list(x.values()))])
        action_probabilities = self.classifier.predict_proba(x)[0]
        next_action = np.random.choice(self.n_actions, p=action_probabilities)
        return next_action

    def update_distribution(self, step):
        x, s, action_index, _, _ = step
        if type(x) is dict:
            x = np.concatenate(list(x.values()))
        self.memory_bank = [step] + self.memory_bank[:self.memory_size-1]

        X = np.array([x])
        y = np.array([action_index])
        self.classifier.partial_fit(X, y)

    def _compute_probabilities(self, X):

        if type(X[0]) is dict:
            X = np.array([np.concatenate(list(x.values())) for x in X])
        return self.classifier.predict_proba(X)

    @classmethod
    def compute_distance(self, c1, c2):
        samples = c1.memory_bank + c2.memory_bank
        distances = []

        xs = np.array([np.concatenate(list(x.values()))
                      for x, _, _, _, _ in samples])
        ys = [y for _, _, y, _, _ in samples]

        if not c1.classifier.is_fitted or not c2.classifier.is_fitted:
            return np.inf

        dists1 = c1._compute_probabilities(xs)
        dists2 = c2._compute_probabilities(xs)

        for i in range(len(xs)):
            dist = spatial.distance.jensenshannon(dists1[i], dists2[i])
            distances.append(dist)

        mean_dist = np.mean(distances)

        return mean_dist

    @classmethod
    def pairwise_distances(self, clist):
        distances = np.zeros((len(clist), len(clist)))
        for i in range(len(clist)):
            for j in range(i+1, len(clist)):
                distances[i, j] = self.compute_distance(clist[i], clist[j])
                distances[j, i] = distances[i, j]
        return distances

    @classmethod
    def merge(cls, state_cluster_list):
        n_actions = state_cluster_list[0].n_actions
        descriptions = [
            state_cluster.state_description for state_cluster in state_cluster_list]
        new_cluster_description = StateDescription.least_general_generalization(
            descriptions)
        new_cluster = cls(new_cluster_description, n_actions)

        _classifier_class = state_cluster_list[0]._classifier_class
        if len(state_cluster_list) > 2:
            raise NotImplementedError(
                "Merging more than 2 clusters not implemented")

        new_cluster.classifier = _classifier_class.merge(
            *[state_cluster.classifier for state_cluster in state_cluster_list])

        memory_bank = state_cluster_list[0].memory_bank + \
            state_cluster_list[1].memory_bank

        random_indices = np.random.choice(len(memory_bank), size=min(
            len(memory_bank), new_cluster.memory_size), replace=False)
        new_cluster.memory_bank = [memory_bank[i] for i in random_indices]

        return new_cluster

    @property
    def is_fitted(self):
        return self.classifier.is_fitted

    @property
    def dist(self):
        return self.classifier

    def cluster_str(self, actions):
        d = self.classifier.class_counts
        vals = {actions[i]: d.get(i, 0) for i in range(len(actions))}

        return f"StateDependentCluster({vals})"

    def __repr__(self) -> str:
        return f"StateDependentCluster({self.state_description},{self.classifier})"

    @property
    def is_stable(self):
        vals = list(self.classifier.class_counts.values())
        if sum(vals) > 0:
            self._is_stable = True

        return self._is_stable
