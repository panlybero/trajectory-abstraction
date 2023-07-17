import numpy as np

def describe_simple_gridworld_state(state, as_num=False):
    distances = state['distance_to_wood']
    inventory = state['inventory']
    cast = lambda x: int(x) if as_num else bool(x)

    is_next_to_wood = cast(np.any(distances==0))
    has_wood = cast(inventory[0]>0)
    has_planks = cast(inventory[1]>0)
    has_chair_parts = cast(inventory[2]>0)
    has_chair = cast(inventory[3]>0)
    has_decoration = cast(inventory[4]>0)
    predicate_dict = {"(next_to wood)":is_next_to_wood, "(has wood)":has_wood, "(has planks)":has_planks, "(has chair_parts)":has_chair_parts, "(has chair)":has_chair, "(has decoration)":has_decoration}

    return predicate_dict