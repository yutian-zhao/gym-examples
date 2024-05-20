import gymnasium as gym
from gymnasium.spaces import Box
import numpy as np


class FullFrame(gym.ObservationWrapper):
    def __init__(self, env,):
        super().__init__(env)
        self.observation_space = Box(shape=(env.size,env.size,1), low=-1, high=1, dtype=int)

    def observation(self, obs):
        map = np.zeros((self.env.size,self.env.size,1))
        map[obs["agent"][0], obs["agent"][1]] = 1
        # map[obs["target"][0], obs["target"][1]] = -1
        return map