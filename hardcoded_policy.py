import numpy as np


class HardcodedPolicy:
    def __init__(self, n, crafting_goal = 'chair'):
        self.n = n

        self.made_chair = False
        self.made_planks = False
        self.made_parts = False
        self.crafting_goal = crafting_goal
        
    def reset(self):
        self.made_chair = False
        self.made_planks = False
        self.made_parts = False
        self.inventory = None


    def _get_wood(self, distance_to_wood):
        min_distance_idx = np.argmin(distance_to_wood)
        if min_distance_idx in [4,5]:
            yield np.random.choice([0,2])
        if min_distance_idx in [6,7]:
            yield np.random.choice([1,3])

        if all(np.array(distance_to_wood) == 1):
            yield np.random.randint(4)
        yield min_distance_idx

    def _get_planks(self, inventory, distance_to_wood):
        while inventory['planks'] ==0:
            if inventory['planks'] < 1 and inventory['wood'] >= 1:
                yield 4
            elif inventory['planks'] < 1 and inventory['wood'] ==0:
                while inventory['wood'] ==0:
                    yield next(self._get_wood(distance_to_wood))

    def _get_chair_parts(self, inventory, distance_to_wood):
        while inventory['chair_parts'] < 1:
            if inventory['planks'] >= 1 and inventory['wood'] >= 1:
                yield 5
            elif inventory['planks'] ==0:
                yield next(self._get_planks(inventory, distance_to_wood))
            elif inventory['wood'] ==0:
                while inventory['wood'] ==0:
                    yield next(self._get_wood(distance_to_wood))
        
    def _get_chair(self, inventory, distance_to_wood):
        while inventory['chair'] < 1:
            if inventory['chair_parts'] >= 1:
                yield 6
            elif inventory['chair_parts'] ==0:
                yield next(self._get_chair_parts(inventory, distance_to_wood))


    def _get_decoration(self, inventory, distance_to_wood):
        while inventory['decoration'] < 1:
            if inventory['planks'] >= 1:
                yield 7
            elif inventory['planks'] ==0:
                yield next(self._get_planks(inventory, distance_to_wood))

    def act(self, distance_to_wood, inventory):
        inventory = {k:v for k,v in zip(['wood', 'planks', 'chair_parts', 'chair','decoration'], inventory)}
        self.inventory = inventory
       # print(inventory)
        if self.crafting_goal == 'planks':
            return next(self._get_planks(inventory, distance_to_wood))
        elif self.crafting_goal == 'chair_parts':
            return next(self._get_chair_parts(inventory, distance_to_wood))
        
        elif self.crafting_goal == 'chair':
            return next(self._get_chair(inventory, distance_to_wood))
        elif self.crafting_goal == 'decoration':
    
            return next(self._get_decoration(inventory, distance_to_wood))
    
    