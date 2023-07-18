class ClusterTransitions:
    def __init__(self):
        self.transitions = {}
        pass

    def __getitem__(self, pair):
        return self.transitions[pair]

    def __setitem__(self, pair, value):
        self.transitions[pair] = value

    def __iter__(self):
        return iter(self.transitions)

    def get(self, pair, default=None):
        return self.transitions.get(pair, default)

    def pop(self, pair, default=None):
        return self.transitions.pop(pair, default)

    def update(self, other):
        self.transitions.update(other.transitions)

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
