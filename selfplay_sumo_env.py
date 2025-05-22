import os
import numpy as np
import gymnasium as gym
import pybullet as p
import pybullet_data
from gymnasium import spaces
from math import sqrt, pi
# at top of your script
import cv2

from envs.sumo_env import SumoEnv

class SumoEnvSelfPlay(SumoEnv):
    def __init__(self, opponent_policy=None, **kwargs):
        super().__init__(**kwargs)
        self.opponent_policy = opponent_policy  # function: obs → action

    def step(self, action_a):
        # Get obs for Bot B
        obs = self._get_obs()
        obs_b = self._extract_obs_for_botB(obs)

        # Decide action for Bot B
        if self.opponent_policy:
            action_b = self.opponent_policy(obs_b)
        else:
            action_b = self.np_random.uniform(-1, 1, size=2)  # random

        # Combine actions
        action = np.concatenate([action_a, action_b])
        return super().step(action)
    
    def _extract_obs_for_botB(self, obs):
    # If symmetric, just return same
        return obs.copy()

