from matplotlib import pyplot as plt
import numpy as np
import gym


class MAGRID(gym.Env):
    def __init__(self, num_agents, grid, gamma):
        self.num_agents = num_agents
        self.grid = grid
        self.gamma = gamma
        self.p_grid, self.p_start, self.p_end = self.get_grid()
        self.num_steps = 0
        self.action = {0: (0, 0, 1), 1: (0, 1, 0), 2: (1, 0, 0), 3: (0, 0, -1), 4: (0, -1, 0), 5: (-1, 0, 0)}
        self.num_states = self.get_observation().size
        self.num_actions = num_agents * 3

    def get_action_space_sample(self):
        sample = list()
        for a in range(self.num_agents):
            sample.append(self.action[np.random.randint(0, 6)])
        return np.array(sample).ravel()

    def get_grid(self):
        p_grid = np.random.choice(a=[0, 1], size=(self.grid, self.grid, self.grid), p=[1. - self.gamma, self.gamma])
        p_start = list()
        p_end = list()
        start_end = [False for _ in range(self.num_agents + 1)]
        while not all(start_end):
            point = tuple(np.random.randint((0, 0, 0), (self.grid, self.grid, self.grid)))
            if not p_grid[point] and not any([all(np.array(p) == point) for p in p_start]) if not len(p_start) else True:
                if start_end[0]:
                    p_start.append(point)
                    start_end[len(p_start)] = True
                else:
                    p_end.append(point)
                    start_end[0] = True
        return p_grid, p_start, p_end

    def get_observation(self):
        obs = np.array([])
        for a in range(self.num_agents):
            for i in range(6):
                obs = np.hstack([obs, self.p_grid[self.p_start[a]]])
            obs = np.hstack([obs, np.array(list(map(lambda t1, t2: t1 - t2, tuple(self.p_end), self.p_start[a]))).squeeze().tolist()])
        return obs.ravel()

    def get_done(self):
        done = False
        if all([p == self.p_end for p in self.p_start]):
            done = True
        return done

    def get_reward(self, action):
        old_distance = list()
        for a in range(self.num_agents):
            old_distance.append(abs(np.array(list(map(lambda t1, t2: t1 - t2, tuple(self.p_end), self.p_start[a]))).squeeze()))

        reward = 0
        for a in range(self.num_agents):
            position = tuple(self.p_start[a] + action[a])
            if not any([p >= self.grid or p < 0 for p in position]):
                if not self.p_grid[position] and not any([p == self.p_end for p in self.p_start]):
                    self.p_start[a] = position
                if position == self.p_end:
                    reward += 1
                else:
                    new_distance = sum(abs(np.array(list(map(lambda t1, t2: t1 - t2, tuple(self.p_end), self.p_start[a]))).squeeze()))
                    if new_distance < sum(old_distance[a]):
                        reward += 0.1
        return reward

    def step(self, action):
        self.num_steps += 1
        reward = self.get_reward(action)
        done = self.get_done()
        obs = self.get_observation()
        return obs, reward, done

    def reset(self):
        self.num_steps = 0
        self.p_grid, self.p_start, self.p_end = self.get_grid()
        obs = self.get_observation()
        return obs

    def render(self, close=False):
        pass
