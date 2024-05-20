import gymnasium as gym
import numpy as np
# import torch
import torch.nn.functional as F

class UpsideWrapper(gym.Wrapper):
    # non-terminal, with truncation
    def __init__(self, env, skill, discriminator, mapping, reward_start, reward_end):
        super().__init__(env)
        self.skill = skill
        self.discriminator = discriminator
        self.mapping = mapping
        self.reward_start = reward_start
        self.reward_end = reward_end
        self.timestep = 0

    def step(self, action):
        observation, _, _, _, info = super().step(action)

        self.timestep += 1
        # reward = int(self.mapping[self.skill]==self.discriminator.predict(observation))
        if self.discriminator and self.timestep>self.reward_start and self.timestep<=self.reward_end:
            reward = F.log_softmax(self.discriminator([observation]), dim=1)[:, self.mapping[self.skill]].item()
        else:
            reward = 0

        return observation, reward, False, self.timestep==self.reward_end, info
    
    def reset(self, seed=None, options=None):
        # We need the following line to seed self.np_random
        observation, info = super().reset(seed=seed, options=[self.size//2, self.size//2])
        # self.set_location()
        self.timestep = 0
        # observation = self.get_obs()
        # info = self.get_info()

        # if self.render_mode == "human":
        #     self._render_frame()
        
        return observation, info