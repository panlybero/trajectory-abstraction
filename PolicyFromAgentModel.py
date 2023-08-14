from hardcoded_policy import HardcodedPolicy
from planning_agent import PlanningAgent
from state_description import describe_state
from StateDescription import StateDescriptionFactory, StateDescription
from AgentModel import AgentModel
from Planner import Planner
import numpy as np


class PolicyFromAgentModel:
    def __init__(self, n, agent_model: AgentModel):
        self.agent_model = agent_model

    def reset(self):
        pass

    def act(self, distance_to_wood, inventory):
        dict_inventory = {k: v for k, v in zip(
            ['wood', 'planks', 'chair_parts', 'chair', 'decoration'], inventory)}

        state = {'inventory': inventory, 'distance_to_wood': distance_to_wood}

        state_description = self.agent_model.state_description_factory.create_state_description_from_dict(
            describe_state(state))

        generalizations = self.agent_model.get_generalizations(
            state_description)
        if generalizations is None:
            # print("No knowledge, acting randomly")
            return np.random.randint(8)

        else:
            return self.agent_model.clusters[generalizations[0]].get_next_action(state)


class PlanningPolicyFromAgentModel(PlanningAgent):
    def __init__(self, n, agent_model: AgentModel, pddl_path='pddl/simple_crafting'):
        self.hardcoded = HardcodedPolicy(n)
        self.agent_model = agent_model
        self.curr_goal = None
        self.curr_plan = None
        self.planner = Planner(path=pddl_path,
                               exec_path='/home/plymper/trajectory-abstraction/pddlgym_planners/FF-v2.3/ff')

        self.goals = []

    def reset(self):
        self.curr_goal = None
        self.curr_plan = None
        self.hardcoded.reset()

    def _current_goal_subsumes_all(self):
        return len(self.agent_model.get_generalizations(self.curr_goal)) == 1

    def act(self, distance_to_wood, inventory):
        dict_inventory = {k: v for k, v in zip(
            ['wood', 'planks', 'chair_parts', 'chair', 'decoration', 'stick'], inventory)}

        state = {'inventory': inventory, 'distance_to_wood': distance_to_wood}

        state_description = self.agent_model.state_description_factory.create_state_description_from_dict(
            describe_state(state))

        while True:
            try:
                if len(self.curr_plan) > 0:
                    return next(self._run_plan(self.curr_plan, state_description, dict_inventory, distance_to_wood))
            except:
                pass

            generalizations = self.agent_model.get_generalizations(
                state_description)
            most_applicable = self.agent_model.get_current_best_cluster_description(
                (state, state_description, None, None, None, None), invented_predicates=self.agent_model.inferred_invented_predicates)  # generalizations[0]

            if generalizations is None:
                # print("No knowledge, acting randomly", state_description)
                return np.random.randint(8)

            if self.agent_model.n_clusters == 1:
                k = next(iter(self.agent_model.clusters.items()))[0]
                return self.agent_model.clusters[k].get_next_action(state)

            # if self.curr_goal is None:
            #     self.curr_goal = self.agent_model.get_next_state(
            #         state_description, use_invented=self.agent_model.invent_predicates)
            #     # print("New goal has stick:", self.curr_goal.holds("(has stick)"), "has decoration", self.curr_goal.holds(
            #     #     "(has decoration)"), "has wood", self.curr_goal.holds("(has wood)"), "has planks", self.curr_goal.holds("(has planks)"))

            # if self.curr_plan is None:
            #     try:
            #         self.curr_plan = self.planner.plan(
            #             state_description.to_list_str(), self.curr_goal.to_conjunction_str())
            #     except ValueError:
            #         self.curr_plan = []
            #         pass
            while self.curr_goal is None or self.curr_plan is None:
                self.curr_goal = self.agent_model.get_next_state(
                    state_description, use_invented=self.agent_model.invent_predicates)

                try:
                    self.curr_plan = self.planner.plan(
                        state_description.to_list_str(), self.curr_goal.to_conjunction_str())
                except ValueError:
                    self.curr_plan = None
                    pass

                    # print("No plan actiing randomly")
                    # return np.random.randint(8)

            while self.curr_goal.subsumes(most_applicable):
                # print("Subsumed", self.curr_goal, most_applicable)
                self.curr_goal = self.agent_model.get_next_state(
                    state_description, use_invented=self.agent_model.invent_predicates)
                if self._current_goal_subsumes_all():
                    # current goal subsumes all other states
                    self.curr_goal = None
                    self.curr_plan = None

                    return self.agent_model.clusters[most_applicable].get_next_action(state)

                try:
                    self.curr_plan = self.planner.plan(
                        state_description.to_list_str(), self.curr_goal.to_conjunction_str())
                except ValueError:
                    self.curr_plan = []
                    # print("Nothing to do, acting randomly1")

                    return self.agent_model.clusters[most_applicable].get_next_action(state)

            try:
                # print("Executing plan", self.curr_plan)
                return next(self._run_plan(self.curr_plan, state_description, dict_inventory, distance_to_wood))
            except StopIteration:

                # print(f"Plan finished for {self.curr_goal}, acting randomly")
                self.curr_goal = None
                self.curr_plan = None
                act = self.agent_model.clusters[most_applicable].get_next_action(
                    state)
                return act

                continue


class PlanningPolicyFromAgentModelV2(PlanningPolicyFromAgentModel):

    def act(self, distance_to_wood, inventory, distance_to_trader):

        dict_inventory = {k: v for k, v in zip(
            ['wood', 'planks', 'chair_parts', 'chair', 'decoration', 'stick'], inventory)}

        state = {'inventory': inventory, 'distance_to_wood': distance_to_wood,
                 'distance_to_trader': distance_to_trader}

        state_description = self.agent_model.state_description_factory.create_state_description_from_dict(
            describe_state(state))

        # print("currently at", state_description)
        while True:
           # print("plan", self.curr_plan)
            if self.curr_plan is not None:
                try:
                    if len(self.curr_plan) > 0:
                        # print('Executing plan -- ', self.curr_plan)

                        act = next(self._run_plan(
                            self.curr_plan, state_description, dict_inventory, distance_to_wood, distance_to_trader))
                        self.agent_model.goals.append(act)
                        # print('act', act)
                        return act
                except StopIteration:
                    # print("Couldnt execute plan",
                    #       self.curr_plan, 'stopiteration')
                    pass

            generalizations = self.agent_model.get_generalizations(
                state_description)
            most_applicable = self.agent_model.get_current_best_cluster_description(
                (state, state_description, None, None, None, None), invented_predicates=self.agent_model.inferred_invented_predicates)  # generalizations[0]

            # if generalizations is None:
            #     # print("No knowledge, acting randomly", state_description)

            #     if self.curr_goal is None:3
            #         self.agent_model.goals.append(f"rand")
            #         return np.random.randint(8)
            #     else:
            #         most_applicable = self.curr_goal
            # print("most applicable", most_applicable)
            # print("curr goal", self.curr_goal)
            # print("curr plan", self.curr_plan)

            if most_applicable is None or generalizations is None:
                # print(state_description)
                # print("No knowledge, attempting to plan to known state",
                # 'n clusters:', self.agent_model.n_clusters)
                for s in self.agent_model.clusters.keys():
                    if not self.agent_model.clusters[s].is_consistent_with_infered(self.agent_model.inferred_invented_predicates):
                        continue
                    self.curr_goal = s
                    try:
                        # print("planning for", s, 'from', state_description)
                        self.curr_plan = self.planner.plan(
                            state_description.to_list_str(), self.curr_goal.to_conjunction_str())
                    except ValueError:
                        self.curr_plan = []
                        pass

                    if len(self.curr_plan) > 0:
                        break

                if len(self.curr_plan) > 0:
                    # print('planned to', self.curr_goal)
                    # print("got one, acting with plan", self.curr_plan)
                    try:
                        act = next(self._run_plan(
                            self.curr_plan, state_description, dict_inventory, distance_to_wood))
                        # print("Doing,", act)
                        self.agent_model.goals.append(act)
                        # print('act', act)
                        return act
                    except:
                        self.curr_plan = None
                        self.curr_goal = None
                        # print("Bad plan acting randomly")
                        return np.random.randint(4)
                else:
                    self.curr_goal = None
                    self.curr_plan = None

                    return np.random.randint(4)

            if self.agent_model.n_clusters == 1:
                k = next(iter(self.agent_model.clusters.items()))[0]
                act = self.agent_model.clusters[k].get_next_action(state)
                return act

            # if self.curr_goal is None:
            #     self.curr_goal = self.agent_model.get_next_state(
            #         state_description, use_invented=self.agent_model.invent_predicates)
            #     # print("New goal has stick:", self.curr_goal.holds("(has stick)"), "has decoration", self.curr_goal.holds(
            #     #     "(has decoration)"), "has wood", self.curr_goal.holds("(has wood)"), "has planks", self.curr_goal.holds("(has planks)"))

            # if self.curr_plan is None:
            #     try:
            #         self.curr_plan = self.planner.plan(
            #             state_description.to_list_str(), self.curr_goal.to_conjunction_str())
            #     except ValueError:
            #         self.curr_plan = []
            #         pass
            attempts = self.agent_model.n_clusters
            exclude = set()
            while self.curr_goal is None or self.curr_plan is None:
                # print(self.curr_goal, self.curr_plan)
                if self.curr_goal is None:
                    try:

                        self.curr_goal = self.agent_model.get_next_state(
                            state_description, use_invented=self.agent_model.invent_predicates, exclude=exclude)
                        # print("new goal", self.curr_goal)
                    except:
                        act = np.random.randint(9)
                        return act

                try:
                    # print('planning')
                    self.curr_plan = self.planner.plan(
                        state_description.to_list_str(), self.curr_goal.to_conjunction_str())
                    # print("from", state_description.to_list_str())
                    # print("to", self.curr_goal.to_conjunction_str())
                    # print("Got plan", self.curr_plan)
                    # print(self.curr_plan, self.curr_goal)
                    if len(self.curr_plan) == 0:
                        raise ValueError
                except ValueError:
                    # print("Excluding", self.curr_goal, self.curr_plan)
                    exclude.add(self.curr_goal)
                    self.curr_plan = None
                    self.curr_goal = None
                    attempts -= 1

                    if attempts == 0:
                        try:
                            act = self.agent_model.clusters[most_applicable].get_next_action(
                                state)
                        except:
                            act = np.random.randint(4)
                        # print('out of options')
                        return act

                    # print("No plan actiing randomly")
                    # return np.random.randint(8)

            try:
                if self.curr_plan is not None:
                    # print('Executing plan --', self.curr_plan)
                    act = next(self._run_plan(
                        self.curr_plan, state_description, dict_inventory, distance_to_wood, distance_to_trader))
                    self.agent_model.goals.append(act)

                    return act
            except StopIteration:
                self.curr_plan = None
                self.curr_goal = None
