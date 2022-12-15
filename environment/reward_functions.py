import math
import numpy as np


class RewardFunction:
    """
    Class that implements a bunch of reward functions that can be used to calculate the rewards for the agent. The reward functions need a track_width which describes the area the boat should stay in. Additionally the reward functions have a x_pos_multiplier that gives extra reward the further the boat gets this parameter however can be turned of by just not setting it and leaving it in its default being 0.
    """

    def __init__(self, config, a=1, b=1):
        self.track_width = config.boat_env.track_width
        self.track_length = config.boat_env.goal_line
        self.a = a
        self.b = b

    def linear_reward(self, position, x_pos_multiplier=0):
        x = position[0] * x_pos_multiplier
        y = np.abs(position[1])
        linear_function = y * self.a
        norm_linear_function = 1 - (linear_function / self.track_width)
        norm_x = x / self.track_length
        return norm_linear_function + norm_x

    def quadratic_reward(self, position, x_pos_multiplier=0):
        x = position[0] * x_pos_multiplier
        y = np.abs(position[1])
        quadratic_function = math.pow(y, 2) * self.a + y * self.b
        return (self.track_width - quadratic_function + x) / (self.track_width + self.track_length)

    def exponential_reward(self, position, x_pos_multiplier=0):
        x = position[0] * x_pos_multiplier
        y = np.abs(position[1])
        exponential_function = self.a * math.exp(y) + x
        return (self.track_width - exponential_function) / (self.track_width + self.track_length)
