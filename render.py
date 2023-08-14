import numpy as np
import cv2


def render(env, size):
    world = env.world.copy()
    world = np.array([[str(int(x)) for x in row] for row in world])

    agent_pos = env.agent_pos.copy()

    world[agent_pos[0], agent_pos[1]] = 'A'
    object_sprites = {
        '0': "sprites/grass.png",  # Empty space
        '1': 'sprites/tree_ongrass.png',  # Wood
        'A': 'sprites/agent_ongrass.png',  # Agent
        '2': 'sprites/trader_ongrass.png',  # Trader
    }

    img = np.zeros((size[0]*32, size[1]*32, 3))
    out = []
    for i in range(env.n):
        out.append([])
        for j in range(env.n):
            out[-1].append(cv2.imread(object_sprites[world[i, j]]))

    out = [np.concatenate(out[i], axis=1) for i in range(env.n)]
    out = np.concatenate(out, axis=0)
    img = out
    img = cv2.resize(img, size)
    return img
