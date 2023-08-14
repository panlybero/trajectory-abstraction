
from Planner import Planner
from hardcoded_policy import HardcodedPolicy
from StateDescription import StateDescription, StateDescriptionFactory
from state_description import describe_state
import numpy as np


class PlanningAgent:
    def __init__(self, n, crafting_goal, predicates, pddl_path='pddl/simple_crafting'):
        self.hardcoded = HardcodedPolicy(n)
        self.crafting_goal = crafting_goal
        self.curr_goal = None
        self.curr_plan = None
        self.planner = Planner(path=pddl_path,
                               exec_path='/home/plymper/trajectory-abstraction/pddlgym_planners/FF-v2.3/ff')

        self.goals = []

        self.state_description_factory = StateDescriptionFactory(predicates)

        self.goal = self.state_description_factory.create_state_description_from_dict(
            {f"(has {crafting_goal})": True, "(next_to trader)": False})
        self.curr_plan = []

    def reset(self):
        self.curr_goal = None
        self.curr_plan = None
        self.hardcoded.reset()

    def _approach_wood(self, distance_to_wood, state_description):
        while not state_description.holds("(next_to wood)"):
            yield next(self.hardcoded._get_wood(distance_to_wood))
        else:
            yield 'DONE'

    def _leave_wood(self, distance_to_wood, state_description):
        while state_description.holds("(next_to wood)"):
            yield next(self.hardcoded._leave_wood(distance_to_wood))
        else:
            yield 'DONE'

    def _approach_trader(self, inventory, distance_to_trader, state_description):
        while not state_description.holds("(next_to trader)"):
            yield next(self.hardcoded._goto_trader(inventory, distance_to_trader))
        else:
            yield 'DONE'

    def _leave_trader(self, distance_to_trader, state_description):
        while state_description.holds("(next_to trader)"):
            yield next(self.hardcoded._leave_wood(distance_to_trader))
        else:
            yield 'DONE'

    def _trade(self, inventory, distance_to_trader, state_description):
        while not state_description.holds("(has wood)"):
            yield next(self.hardcoded._do_trade(inventory, distance_to_trader))
        else:
            yield 'DONE'

    def _break_wood(self, distance_to_wood, state_description):
        while not state_description.holds("(has wood)"):
            yield next(self.hardcoded._get_wood(distance_to_wood))
        else:
            yield 'DONE'

    def _run_plan(self, plan, state_description, dict_inventory, distance_to_wood, distance_to_trader):

        while len(plan) > 0:

            # print('Executing ',plan)
            high_level_action = plan[0]

            if high_level_action == 'APPROACH WOOD':
                action = next(self._approach_wood(
                    distance_to_wood, state_description))

            if high_level_action == 'APPROACH TRADER':
                action = next(self._approach_trader(dict_inventory,
                                                    distance_to_trader, state_description))

            if high_level_action == 'TRADE':
                if dict_inventory['planks'] == 0:
                    action = 9
                else:
                    action = 'DONE'

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
            # print(high_level_action, action)
            if action == 'DONE':
                plan.pop(0)
                continue
            else:
                yield action

    def _current_goal_subsumes_all(self):
        return len(self.agent_model.get_generalizations(self.curr_goal)) == 1

    def act(self, distance_to_wood, inventory, distance_to_trader):
        dict_inventory = {k: v for k, v in zip(
            ['wood', 'planks', 'chair_parts', 'chair', 'decoration', 'stick'], inventory)}

        state = {'inventory': inventory, 'distance_to_wood': distance_to_wood,
                 'distance_to_trader': distance_to_trader}

        state_description = self.state_description_factory.create_state_description_from_dict(
            describe_state(state))

        while True:

            try:
                return next(self._run_plan(self.curr_plan, state_description, dict_inventory, distance_to_wood, distance_to_trader))
            except StopIteration:
                self.curr_plan = self.planner.plan(
                    state_description.to_list_str(), self.goal.to_conjunction_str())
                print(self.curr_plan)
