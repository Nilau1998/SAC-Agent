from gym import Env
from gym.spaces import Discrete, Box
import numpy as np
import random


# https://www.youtube.com/watch?v=bD6V3rcr_54
class ShowerEnv(Env):
    def __init__(self):
        self.action_space = Discrete(3)
        self.observation_space = Box(low=np.array([0]), high=np.array([100]))
        self.state = 38 + random.randint(-3, 3)
        self.shower_length = 60

    def step(self, action):
        # Apply action
        self.state += action - 1
        self.shower_length -= 1

        # Reward calculation
        if self.state >= 37 and self.state <=39:
            reward = 1
        else:
            reward = -1

        # Check if shower time is over
        done = False
        if self.shower_length <= 0:
            done = True

        # Apply temperature noise
        self.state +=random.randint(-1, 1)
        info = {}

        return self.state, reward, done, info

    def render(self):
        pass

    def reset(self):
        # Reset states to default
        self.state = 38 + random.randint(-3, 3)
        self.shower_length = 60
        return self.state