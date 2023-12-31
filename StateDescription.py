class StateDescription:
    def __init__(self, predicates, invented_predicates=None):
        self.predicates = predicates
        if invented_predicates is None:
            self.invented_predicates = set()

    def subsumes(self, other):
        return set(self.predicates).issubset(other.predicates)

    def __eq__(self, other):
        return set(self.predicates) == set(other.predicates)

    def to_conjunction_str(self):
        conj = f"(and {' '.join(self.predicates)})"

        return conj

    def add_invented_predicate(self, predicate):
        self.invented_predicates.add(predicate)

    def to_list_str(self):
        s = '\n'.join(self.predicates)
        return s

    @classmethod
    def least_general_generalization(cls, state_descriptions):
        if not state_descriptions:
            return None

        predicates_intersection = set(state_descriptions[0].predicates)
        invented_predicates_intersection = set(
            state_descriptions[0].invented_predicates)
        for state_desc in state_descriptions[1:]:
            predicates_intersection.intersection_update(state_desc.predicates)
            invented_predicates_intersection.intersection_update(
                state_desc.invented_predicates)

        return cls(list(predicates_intersection))

    def get_applicable_generalizations(self, generalizations):
        applicable_generalizations = []

        for generalization in generalizations:
            if generalization.subsumes(self):
                applicable_generalizations.append(generalization)

        if len(applicable_generalizations) == 0:
            return None

        applicable_generalizations.sort(
            key=lambda x: len(x.predicates), reverse=True)

        return applicable_generalizations

    def holds(self, predicate):
        return predicate in self.predicates or predicate in self.invented_predicates

    def __hash__(self):
        return hash(tuple(sorted(self.predicates + list(self.invented_predicates))))

    def __repr__(self):
        return f"StateDescription(pred={self.predicates}, invented={self.invented_predicates})"


class StateDescriptionFactory:
    def __init__(self, predicates):
        self.possible_predicates = predicates

    def create_state_description(self, predicates, invented=None):
        parsed_predicates = []
        negated_predicates = []

        for predicate in predicates:
            if predicate.startswith("(not") and predicate.endswith(")"):
                negated_predicates.append(predicate[5:-1])
            else:
                parsed_predicates.append(predicate)

        invalid_predicates = set(
            parsed_predicates + negated_predicates) - set(self.possible_predicates)
        if invalid_predicates:
            raise ValueError(
                f"Invalid predicates: {invalid_predicates}", 'Possible predicates:', self.possible_predicates)

        if invented is None:
            invented = []
        return StateDescription(parsed_predicates + [f"(not {neg_pred})" for neg_pred in negated_predicates] + invented)

    def create_state_description_from_dict(self, predicates_dict):
        predicates = []
        for k, v in predicates_dict.items():
            if v:
                predicates.append(k)
            else:
                predicates.append(f"(not {k})")

        return self.create_state_description(predicates)


# Example usage
if __name__ == '__main__':
    # Create the StateDescriptionFactory with possible predicates
    predicate_set = {"is_next_to_wood", "has_wood",
                     "has_planks", "has_chair_parts", "has_chair"}

    factory = StateDescriptionFactory(predicate_set)

    # Create StateDescription objects using the factory
    state1 = factory.create_state_description(['is_next_to_wood', 'has_wood'])
    state2 = factory.create_state_description(
        ['is_next_to_wood', 'has_planks'])
    state3 = factory.create_state_description(['has_wood', 'has_chair_parts'])

    # Check if state1 is more general than state2
    print(state1.subsumes(state2))  # Output: True

    # Check if state2 is more general than state1
    print(state2.subsumes(state1))  # Output: False

    # Compute the least general generalization
    least_general = StateDescription.least_general_generalization(
        [state1, state2, state3])
    print(least_general.predicates)  # Output: ['predicate1']

    # Use as keys in a dictionary
    state_dict = {state1: 'value1', state2: 'value2'}
    # Output: {StateDescription(predicates=['predicate1', 'predicate2']): 'value1',
    print(state_dict)
    #          StateDescription(predicates=['predicate1', 'predicate3']): 'value2'}

    state1 = factory.create_state_description(['is_next_to_wood', 'has_wood'])
    state2 = factory.create_state_description(['is_next_to_wood', 'has_wood'])
    a = {}
    a[state1] = 1
    a[state2] = 2
    print(a)
