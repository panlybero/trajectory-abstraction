import numpy as np


class ClusterTransitions:
    def __init__(self):
        self.transitions = {}
        self.updated = {}
        pass

    def drop_deprecated_transitions(self, cluster_descriptions):
        for pair in list(self.transitions.keys()):
            if pair[0] not in cluster_descriptions or pair[1] not in cluster_descriptions:
                self.transitions.pop(pair, None)

    def _named_transitions(self, cluster_names):
        named_transitions = {}
        # print(cluster_names)
        for k, v in self.transitions.items():
            # print("*****", v)
            # print(k[0])
            # print(k[1])
            try:
                named_transitions[(cluster_names[k[0]], cluster_names[k[1]])
                                  ] = v
            except:
                # some outdated transitions.
                pass

        return named_transitions

    def _apply_linear_decay(self, decay):
        # print("HERE")
        for k in self.transitions:
            v = self.transitions.get(k, 0)

            if v > 0 and not self.updated.get(k, False):
                self.transitions[k] = max(0, v-decay)

        self.updated = {}

    def keys(self):
        return self.transitions.keys()

    def __getitem__(self, pair):
        return self.transitions[pair]

    def __setitem__(self, pair, value):
        self.transitions[pair] = value
        self.updated[pair] = True

    def __iter__(self):
        return iter(self.transitions)

    def get(self, pair, default=None):
        return self.transitions.get(pair, default)

    def pop(self, pair, default=None):
        return self.transitions.pop(pair, default)

    def update(self, other):
        self.transitions.update(other.transitions)

    def update_state_description(self, old, new):
        from_old = []
        to_old = []
        for pair in self.transitions:
            if pair[0] == old:
                from_old.append(pair)
            elif pair[1] == old:
                to_old.append(pair)

        for pair in from_old:
            self.transitions[(new, pair[1])] = self.transitions.pop(pair)

        for pair in to_old:
            self.transitions[(pair[0], new)] = self.transitions.pop(pair)

    def get_transitions_out(self, state):
        transitions = []
        for pair in self.transitions:
            if pair[0] == state:
                transitions.append(pair)

        return transitions

    def get_transitions_in(self, state):
        transitions = []
        for pair in self.transitions:
            if pair[1] == state:
                transitions.append(pair)

        return transitions

    def get_transitions(self, state):
        _in = self.get_transitions_in(state)
        _out = self.get_transitions_out(state)

        return _in, _out

    def items(self):
        return self.transitions.items()

    def get_transition_matrix(self, clusters, get_counts=False):
        n_clusters = len(clusters)

        matrix = np.zeros((n_clusters, n_clusters))
        descriptions = clusters.keys()

        for t, v in self.items():
            try:
                i = descriptions.index(t[0])
                j = descriptions.index(t[1])
            except ValueError as e:
                # outdated transition, will be cleaned up later
                continue
            matrix[i, j] = v
        # normalize
        if not get_counts:
            matrix = matrix + 1
            matrix = matrix / np.sum(matrix, axis=1, keepdims=True)

        return matrix
