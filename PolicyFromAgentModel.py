from hardcoded_policy import HardcodedPolicy
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


class PlanningPolicyFromAgentModel:
    def __init__(self, n, agent_model: AgentModel):
        self.hardcoded = HardcodedPolicy(n)
        self.agent_model = agent_model
        self.curr_goal = None
        self.curr_plan = None
        self.planner = Planner(path='pddl/simple_crafting',
                               exec_path='/home/plymper/trajectory-abstraction/pddlgym_planners/FF-v2.3/ff')

        self.goals = []

    def reset(self):
        self.curr_goal = None
        self.curr_plan = None
        self.hardcoded.reset()

    def _approach_wood(self, distance_to_wood, state_description):
        while not state_description.holds("(next_to wood)"):
            yield next(self.hardcoded._get_wood(distance_to_wood))
        else:
            yield 'DONE'

    def _break_wood(self, distance_to_wood, state_description):
        while not state_description.holds("(has wood)"):
            yield next(self.hardcoded._get_wood(distance_to_wood))
        else:
            yield 'DONE'

    def _run_plan(self, plan, state_description, dict_inventory, distance_to_wood):

        while len(plan) > 0:

            # print('Executing ',plan)
            high_level_action = plan[0]

            if high_level_action == 'APPROACH WOOD':
                action = next(self._approach_wood(
                    distance_to_wood, state_description))

            if high_level_action == 'BREAK WOOD':
                action = next(self._break_wood(
                    distance_to_wood, state_description))

            if high_level_action == 'CRAFT_PLANKS':
                if dict_inventory['planks'] == 0:
                    action = 4
                else:
                    action = 'DONE'

            if high_level_action == 'CRAFT_CHAIR_PARTS':
                if dict_inventory['chair_parts'] == 0:
                    action = 5
                else:
                    action = 'DONE'

            if high_level_action == 'CRAFT_CHAIR':
                if dict_inventory['chair'] == 0:
                    action = 6
                else:
                    action = 'DONE'

            if high_level_action == 'CRAFT_DECORATION':
                if dict_inventory['decoration'] == 0:
                    action = 7
                else:
                    action = 'DONE'
            if high_level_action == 'CRAFT_STICK':
                if dict_inventory['stick'] == 0:
                    action = 8
                else:
                    action = 'DONE'

            if action == 'DONE':
                plan.pop(0)
                continue
            else:
                yield action

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

    def act(self, distance_to_wood, inventory):
        dict_inventory = {k: v for k, v in zip(
            ['wood', 'planks', 'chair_parts', 'chair', 'decoration', 'stick'], inventory)}

        state = {'inventory': inventory, 'distance_to_wood': distance_to_wood}

        state_description = self.agent_model.state_description_factory.create_state_description_from_dict(
            describe_state(state))

        while True:
            try:
                if len(self.curr_plan) > 0:
                    # print('Executing plan -- ', self.curr_plan)

                    act = next(self._run_plan(
                        self.curr_plan, state_description, dict_inventory, distance_to_wood))
                    self.agent_model.goals.append(act)
                    # print('act', act)
                    return act
            except:
                pass

            generalizations = self.agent_model.get_generalizations(
                state_description)
            most_applicable = self.agent_model.get_current_best_cluster_description(
                (state, state_description, None, None, None, None), invented_predicates=self.agent_model.inferred_invented_predicates)  # generalizations[0]

            if generalizations is None:
                # print("No knowledge, acting randomly", state_description)

                if self.curr_goal is None:
                    self.agent_model.goals.append(f"rand")
                    return np.random.randint(8)
                else:
                    most_applicable = self.curr_goal
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
            attempts = 5
            while self.curr_goal is None or self.curr_plan is None:
                # print(self.curr_goal, self.curr_plan)
                if self.curr_goal is None:
                    try:

                        self.curr_goal = self.agent_model.get_next_state(
                            state_description, use_invented=self.agent_model.invent_predicates)
                    except:
                        act = np.random.randint(8)
                        return act

                try:
                    # print('planning')
                    self.curr_plan = self.planner.plan(
                        state_description.to_list_str(), self.curr_goal.to_conjunction_str())
                    # print(self.curr_plan, self.curr_goal)
                    if len(self.curr_plan) == 0:
                        raise ValueError
                except ValueError:
                    self.curr_plan = None
                    self.curr_goal = None
                    attempts -= 1

                    if attempts == 0:
                        try:
                            act = self.agent_model.clusters[most_applicable].get_next_action(
                                state)
                        except:
                            act = np.random.randint(8)
                        return act

                    # print("No plan actiing randomly")
                    # return np.random.randint(8)

            try:
                if self.curr_plan is not None:
                    # print('Executing plan --', self.curr_plan)
                    act = next(self._run_plan(
                        self.curr_plan, state_description, dict_inventory, distance_to_wood))
                    self.agent_model.goals.append(act)

                    return act
            except:
                self.curr_plan = None
                self.curr_goal = None
