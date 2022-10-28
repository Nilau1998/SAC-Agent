from turtle import position
from typing import Tuple
from gym import Env
from gym.spaces import Box, Dict
import numpy as np
import random
import math

class BoatEnv(Env):
    def __init__(self, config):
        self.config = config
        self.current_step = 0
        self.boat = Boat()
        self.reward_function = RewardFunction()

        # Define action space
        # The agent can turn the boats angle by a certain amount, nothing more
        self.min_action = -1.0
        self.max_action = 1.0
        self.action_space = Box(
            low=self.min_action,
            high=self.max_action,
            shape=(1,),
            dtype=np.float32
        )
        # Define observation space
        # Observered values are the position of the boat, its angle, the wind and when it comes
        self.spaces = {
            "x_pos": Box(low=0, high=150, dtype=np.float32),
            "y_pos": Box(low=-5, high=5, dtype=np.float32),
            "bangle": Box(low=math.radians(-90), high=math.radians(90), dtype=np.float32),
            "wforce": Box(low=0, high=2, dtype=np.float32),
            "wangle": Box(low=-1, high=1, dtype=np.float32)
        }
        self.observation_space = Dict(self.spaces)

    def step(self, action):
        # Apply action
        self.current_step += 1
        self.state

        # Reward calculation
        reward = self.reward_function.linear_reward(
            position=boat.position,
            a=1,
            x_pos_multiplier=0
        )

        # Check if goal is reached, or boat died
        pass

    def render(self):
        pass

    def reset(self):
        pass

class Boat:
    def __init__(self, config):
        # Env
        self.current_step = 0

        # Boat
        self.velocity = np.array([1, 0], dtype=np.float32)
        self.position = np.array([0, 0], dtype=np.float32)
        self.angle = 0 # In relation to x-axis, +y = +angle, -y = -angle, based on velocity
        self.mass = 0 # Not implemented for now

        # Wind
        self.wind_angle = random.choice([-1, 1]) # From where the wind comes, in relation to x-axis again
        self.wind_force = random.uniform(1, 2) # How hard the wind influences the boats angle, change happens each tick/step
        self.wind_delay = random.randint(0, 5) # Delay in steps from wind announcement to actual affect on boat
        self.wind_change_step = 0

    def get_angle(self):
        abs_velocity = math.sqrt(math.pow(self.velocity[0], 2) + math.pow(self.velocity[1], 2))
        self.angle = math.acos(self.velocity[0] / abs_velocity)

    def set_boat_position(self):
        self.position += self.velocity

    def set_velocities(self, delta_angle):
        self.get_angle()
        self.angle += delta_angle
        abs_velocity = math.sqrt(math.pow(self.velocity[0], 2) + math.pow(self.velocity[1], 2))
        self.velocity[0] = abs_velocity * math.cos(self.angle)
        self.velocity[1] = abs_velocity * math.sin(self.angle)

    def set_wind_attributes(self):
        self.wind_change_step = self.current_step
        self.wind_angle = random.choice([-1, 1])
        self.force = random.uniform(1, 2) # Set config
        self.wind_delay = random.randint(0, 5)


    def apply_wind(self):
        self.get_angle()
        # Wind only affects the boat angle for now, wind comes from 90° or -90°
        if self.current_step - self.wind_change_step >= self.wind_delay:
            self.set_velocities(math.radians(1) * self.wind_angle * self.wind_force)


class RewardFunction:
    def __init__(self, track_width):
        self.track_width = track_width

    def linear_reward(self, position, a, x_pos_multiplier=0):
        x = position[0] * x_pos_multiplier
        y = position[1]
        linear_function = y * a
        return self.track_width - linear_function + x

    def quadratic_reward(self, position, a, b, x_pos_multiplier=0):
        x = position[0] * x_pos_multiplier
        y = position[1]
        quadratic_function = math.pow(y, 2) * a + y * b
        return self.track_width - quadratic_function + x

    def exponential_reward(self, position, a, x_pos_multiplier=0):
        x = position[0] * x_pos_multiplier
        y = position[1]
        exponential_function = a * math.exp(y)
        return self.track_width - exponential_function + x

if __name__ == '__main__':
    boat = Boat("bla")
    print(boat.velocity, boat.angle)
    boat.set_velocities(math.radians(90))
    print(boat.velocity, boat.angle)