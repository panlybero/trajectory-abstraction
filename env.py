import gym
import numpy as np


class GridWorldEnv(gym.Env):
    def __init__(self, n, crafting_goal='chair', max_timesteps=100, success_reward=100, dict_obs=True):
        super(GridWorldEnv, self).__init__()

        self.n = n
        self.world = np.zeros((n, n))
        self.agent_pos = np.random.randint(0, n, size=2)
        self.wood_positions = self._scatter_wood()
        self.max_timesteps = max_timesteps
        self.timesteps = 0
        self.crafting_goal = crafting_goal
        self.success_reward = success_reward
        self.inventory = {
            'wood': 0,
            'planks': 0,
            'chair_parts': 0,
            'chair': 0,
            'decoration': 0,
            'stick': 0
        }
        self.dict_obs = dict_obs
        if dict_obs:
            self.observation_space = gym.spaces.Dict({
                'distance_to_wood': gym.spaces.Box(low=0.0, high=1, shape=(8,), dtype=np.float32),
                'inventory': gym.spaces.Box(low=0, high=5, shape=(len(self.inventory),), dtype=np.float32)
            })
        else:
            self.observation_space = gym.spaces.Box(
                low=0, high=5, shape=(8+len(self.inventory)+5,), dtype=np.float32)

        # 4 navigation actions, 4 crafting action
        self.action_space = gym.spaces.Discrete(9)

    def _scatter_wood(self):
        num_wood = np.random.randint(4, max(5, self.n))
        wood_positions = []
        while len(wood_positions) < num_wood:
            pos = tuple(np.random.randint(0, self.n, size=2))
            if pos != tuple(self.agent_pos) and pos not in wood_positions:
                wood_positions.append(pos)
                self.world[pos[0], pos[1]] = 1

        return wood_positions

    def _get_observation(self):
        obs = {
            'distance_to_wood': self._calculate_distance_to_wood()/self.n,
            'inventory': np.array(list(self.inventory.values()))
        }

        if not self.dict_obs:
            obs = np.concatenate(
                [obs['distance_to_wood'], obs['inventory'], np.random.rand(5)])

        return obs

    def _calculate_distance_to_wood(self):
        distances = np.zeros(8) + self.n
        agent_x, agent_y = self.agent_pos

        for i, (dx, dy) in enumerate([(0, -1), (0, 1), (-1, 0), (1, 0), (-1, -1), (-1, 1), (1, -1), (1, 1)]):
            x = agent_x + dx
            y = agent_y + dy
            distance = 0

            while 0 <= x < self.n and 0 <= y < self.n:
                if self.world[x, y] == 1:  # Wood found in this direction
                    distances[i] = distance
                    break
                distance += 1
                x += dx
                y += dy

        return distances

    def _craft_planks(self):
        if self.inventory['wood'] >= 1:
            self.inventory['wood'] -= 1
            self.inventory['planks'] += 1
            return True
        return False

    def _craft_stick(self):
        if self.inventory['planks'] >= 1:
            self.inventory['planks'] -= 1
            self.inventory['stick'] += 1
            return True
        return False

    def _craft_decoration(self):
        if self.inventory['planks'] >= 1:
            self.inventory['planks'] -= 1
            self.inventory['decoration'] += 1
            return True
        return False

    def _craft_chair_parts(self):
        if self.inventory['wood'] >= 1 and self.inventory['planks'] >= 1:
            self.inventory['wood'] -= 1
            self.inventory['planks'] -= 1
            self.inventory['chair_parts'] += 1
            return True
        return False

    def _craft_chair(self):
        if self.inventory['chair_parts'] >= 1:
            self.inventory['chair_parts'] -= 1
            self.inventory['chair'] += 1
            return True
        return False

    def _navigate(self, action):
        dx, dy = [(0, -1), (0, 1), (-1, 0), (1, 0)][action]
        new_pos = self.agent_pos + np.array([dx, dy])

        if 0 <= new_pos[0] < self.n and 0 <= new_pos[1] < self.n:
            self.agent_pos = new_pos
            new_pos = tuple(new_pos)

            # Check if there is an item to collect at the new position
            if new_pos in self.wood_positions:

                self.wood_positions.remove(new_pos)
                self.inventory['wood'] += 1
                self.world[new_pos[0], new_pos[1]] = 0

            return True
        return False

    def render(self, mode="human"):
        return

    def render_ascii(self):
        n = self.n
        world = self.world.copy()
        world = np.array([[str(int(x)) for x in row] for row in world])

        agent_pos = self.agent_pos.copy()

        # Set the agent's position in the world grid
        world[agent_pos[0], agent_pos[1]] = 'A'

        # Define the ASCII characters for different objects
        object_chars = {
            '0': '.',  # Empty space
            '1': 'W',  # Wood
            'A': 'A',  # Agent
            'P': 'P',  # Planks
            'C': 'C',  # Chair Parts
            'H': 'H'  # Chair
        }

        # Print the ASCII representation of the world grid
        for row in range(n):
            line = ''
            for col in range(n):
                line += object_chars[world[row, col]] + ' '
            print(line)

    def step(self, action):
        self.timesteps += 1
        reward = -1
        if action < 4:  # Navigation actions (forward, back, left, right)
            success = self._navigate(action)

        else:  # Crafting action
            if action == 4:  # Craft planks
                success = self._craft_planks()
            elif action == 5:  # Craft chair_parts
                success = self._craft_chair_parts()
            elif action == 6:  # Craft chair
                success = self._craft_chair()
            elif action == 7:  # Craft decoration
                success = self._craft_decoration()
            elif action == 8:  # Craft stick
                success = self._craft_stick()

        if self.inventory[self.crafting_goal] >= 1 or self.timesteps >= self.max_timesteps:
            done = True  # Episode termination is not implemented in this example
            reward = self.success_reward if self.inventory[self.crafting_goal] >= 1 else -1
        else:
            done = False

        obs = self._get_observation()

        return obs, float(reward), done, {}

    def reset(self):
        self.world = np.zeros((self.n, self.n))
        self.agent_pos = np.random.randint(0, self.n, size=2)
        self.wood_positions = self._scatter_wood()
        self.timesteps = 0
        self.inventory = {
            'wood': 0,
            'planks': 0,
            'chair_parts': 0,
            'chair': 0,
            'decoration': 0,
            'stick': 0
        }

        obs = self._get_observation()

        return obs
