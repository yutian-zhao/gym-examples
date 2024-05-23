import gymnasium as gym
from gymnasium.spaces import Box
import numpy as np


class AgentLocation(gym.ObservationWrapper):
    def __init__(self, env,):
        super().__init__(env)
        self.observation_space = Box(shape=(2,), low=0, high=self.env.unwrapped.size, dtype=int)

    def observation(self, obs):
        return obs["agent"]
