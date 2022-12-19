import math
import numpy as np


class RewardFunction:
    """
    Class that implements a bunch of reward functions that can be used to calculate the rewards for the agent. The reward functions need a track_width which describes the area the boat should stay in on the y axis. Additionally the reward functions have a x_pos_multiplier that gives extra reward the further the boat gets this parameter however can be turned of by just not setting it and leaving it in its default being 0. The linear, quadratic.. only matters for the y axis. Linear would mean that the middle of the track gives max points and it linearely decreases towards the edge to 0. Additional points are given the futher the agent drives the boat towards the goal line.
    """

    def __init__(self, config, a=1, b=1):
        self.track_width = config.boat_env.track_width
        self.track_length = config.boat_env.goal_line
        self.a = a
        self.b = b
        self.buffer = 0

    def linear_reward(self, position):
        x = position[0]
        if x > self.buffer:
            self.buffer = x
            y = np.abs(position[1])
            linear_function = y * self.a
            norm_linear_function = linear_function / self.track_width
            return norm_linear_function
        else:
            return 0

    def quadratic_reward(self, position):
        x = position[0]
        y = np.abs(position[1])
        quadratic_function = math.pow(y, 2) * self.a + y * self.b
        return (self.track_width - quadratic_function + x) / (self.track_width + self.track_length)

    def exponential_reward(self, position):
        x = position[0]
        y = np.abs(position[1])
        exponential_function = self.a * math.exp(y) + x
        return (self.track_width - exponential_function) / (self.track_width + self.track_length)
