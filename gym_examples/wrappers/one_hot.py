import gymnasium as gym
from gymnasium.spaces import Box
import numpy as np


class OneHot(gym.ObservationWrapper):
    def __init__(self, env,):
        super().__init__(env)
        self.observation_space = Box(shape=(self.env.size*self.env.size,), low=0, high=1, dtype=int)

    def observation(self, obs):
        one_hot = np.zeros(self.env.size*self.env.size)
        one_hot[self.env.size*obs["agent"][0]+obs["agent"][1]] = 1
        return one_hot