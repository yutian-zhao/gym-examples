from gymnasium.envs.registration import register
from gym_examples.envs import *
from gym_examples.wrappers import *

register(
    id="gym_examples/GridWorld-v0",
    entry_point="gym_examples:GridWorldEnv", # gym_examples.envs:
)
