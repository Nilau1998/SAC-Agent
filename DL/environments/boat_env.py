from gym import Env
from gym.spaces import Box, Dict
import numpy as np
import random
import math

class BoatEnv(Env):
    def __init__(self, config):
        self.config = config
        self.boat = Boat(self.config)
        self.reward_function = RewardFunction(
            track_width=self.config.boat_env.track_width
        )

        # Define action space
        self.min_action = -1.0
        self.max_action = 1.0
        self.action_space = Box(
            low=self.min_action,
            high=self.max_action,
            shape=(1,),
            dtype=np.float32
        )

        # Define observation space
        self.spaces = {
            "x_pos": Box(low=np.array([0]), high=np.array([150]), dtype=np.float32),
            "y_pos": Box(low=np.array([-5]), high=np.array([5]), dtype=np.float32),
            "boat_angle": Box(low=np.array([math.radians(-90)]), high=np.array([math.radians(90)]), dtype=np.float32),
            "current_wind_force": Box(low=np.array([0]), high=np.array([2]), dtype=np.float32),
            "current_wind_angle": Box(low=np.array([-1]), high=np.array([1]), dtype=np.float32),
            "next_wind_angle": Box(low=np.array([-1]), high=np.array([1]), dtype=np.float32)
        }
        self.observation_space = Dict(self.spaces)

        # bounds = np.ones((6), dtype=np.float32)
        # self.observation_space = Box(low=-bounds, high=bounds)

    # Define the step an agent can take. This means that within a step the agent can change the boats angle
    # and additionally all other calulcations happen like the wind impact.
    def step(self, action):
        # Recalc boat angle just in case (Probs not needed?), also set attributes
        self.boat.recalc_angle()
        self.boat.next_wind_change()
        self.boat.current_step += 1
        self.boat.fuel -= 1

        # Apply action
        self.boat.set_velocities(action)
        # Apply wind
        self.boat.apply_wind()

        # Finish action with boat reposition after direction change and wind application
        self.boat.set_boat_position()


        self.state = {
            "x_pos": self.boat.position[0],
            "y_pos": self.boat.position[1],
            "boat_angle": self.boat.angle,
            "current_wind_force": self.boat.current_wind_force,
            "current_wind_angle": self.boat.current_wind_angle,
            "next_wind_angle": self.boat.next_wind_angle
        }

        # Reward calculation
        reward = self.reward_function.linear_reward(
            position=self.boat.position,
            a=1,
            x_pos_multiplier=0
        )

        # Check if goal is reached, or if the boat ran out of fuel
        # Timeout
        done = False
        if self.boat.fuel <= 0:
            done = True
        elif self.boat.position[0] >= 100:
            done = True

        info = {}

        return self.state, reward, done, info

    def render(self):
        pass

    def reset(self):
        self.boat = Boat(self.config)

        self.state = {
            "x_pos": self.boat.position[0],
            "y_pos": self.boat.position[1],
            "boat_angle": self.boat.angle,
            "current_wind_force": self.boat.current_wind_force,
            "current_wind_angle": self.boat.current_wind_angle,
            "next_wind_angle": self.boat.next_wind_angle
        }

        info = {}

        return self.state, info

class Boat:
    def __init__(self, config):
        self.config = config

        # Env
        self.current_step = 0
        self.wind_forecast = generate_randint(self.config)

        # Boat
        self.velocity = np.array([self.config.boat_env.boat_velocity, 0], dtype=np.float32)
        self.position = np.array([0, 0], dtype=np.float32)
        self.angle = 0 # In relation to x-axis, +y = +angle, -y = -angle, based on velocity
        self.fuel = self.config.boat_env.boat_fuel
        self.mass = 0 # Not implemented for now

        # Wind
        self.current_wind_angle = 0 # From where the wind comes, in relation to x-axis again
        self.current_wind_force = 0 # How hard the wind influences the boats angle, change happens each tick/step
        self.next_wind_angle = 0
        self.next_wind_force = 0

    def recalc_angle(self):
        """
        Recalculates the current angle of the ship based on the velocity vector.
        """
        abs_velocity = math.sqrt(math.pow(self.velocity[0], 2) + math.pow(self.velocity[1], 2))
        self.angle = math.acos(self.velocity[0] / abs_velocity)

    def set_boat_position(self):
        """
        Moves the ship based on the current velocity if a step passes.
        """
        self.position += self.velocity

    def set_velocities(self, delta_angle):
        """
        Rotates the velocity vector by delta_angle. The absolute value of the velocity vector stays constant.
        delta_angle is passed in degree
        """
        self.angle += math.radians(delta_angle * self.config.boat_env.boat_angle_scale)
        abs_velocity = math.sqrt(math.pow(self.velocity[0], 2) + math.pow(self.velocity[1], 2))
        self.velocity[0] = abs_velocity * math.cos(self.angle)
        self.velocity[1] = abs_velocity * math.sin(self.angle)

    def next_wind_change(self) -> int:
        """
        Returns the amount of timesteps until the next weather event happens
        This is used to simulate the latency of the dynamic system
        This method also sets the new wind attributes so they can be applied on the ship
        """
        steps_until_wind_change = self.wind_forecast[0] - self.current_step
        # Create initial next attributes for the first forecast
        if self.current_step == 0:
            self.set_wind_attributes()
            self.wind_forecast = np.delete(self.wind_forecast, 0, 0)

        # On weather forecast step, next -> current, generate new next, pop current forecast
        if self.wind_forecast[0] == self.current_step:
            self.current_wind_angle = self.next_wind_angle
            self.current_wind_force = self.next_wind_angle
            self.set_wind_attributes()
            self.wind_forecast = np.delete(self.wind_forecast, 0, 0)
        return steps_until_wind_change

    def set_wind_attributes(self):
        """
        Creates a new set of wind attributes for the upcoming forecast.
        """
        self.next_wind_angle = random.choice([-1, 1])
        self.next_wind_force = random.uniform(0, self.config.boat_env.wind_force)

    def apply_wind(self):
        """
        Applies the wind on the boat.
        Wind can blow from either 90° or -90° in relation to the x-axis. (90° towads -y, -90° is towards y).
        Wind changes the rotation of the boat and therefore gives it a little challenge the agent has to solve,
        the challenge is increased by the wind changes that happen after every couple steps.
        """
        # Wind only affects the boat angle for now, wind comes from 90° or -90°
        self.set_velocities(self.current_wind_angle * self.current_wind_force)
        self.recalc_angle()


class RewardFunction:
    """
    Class that implements a bunch of reward functions that can be used to calculate the rewards
    for the agent.
    The reward functions need a track_width which describes the area the boat should stay in.
    Additionally the reward functions have a x_pos_multiplier that gives extra reward the further the boat gets
    this parameter however can be turned off by just not setting it and leaving it in its default being 0.
    """
    def __init__(self, track_width):
        self.track_width = track_width

    def linear_reward(self, position, a, x_pos_multiplier=0) -> float:
        x = position[0] * x_pos_multiplier
        y = position[1]
        linear_function = y * a
        return self.track_width - linear_function + x

    def quadratic_reward(self, position, a, b, x_pos_multiplier=0) -> float:
        x = position[0] * x_pos_multiplier
        y = position[1]
        quadratic_function = math.pow(y, 2) * a + y * b
        return self.track_width - quadratic_function + x

    def exponential_reward(self, position, a, x_pos_multiplier=0) -> float:
        x = position[0] * x_pos_multiplier
        y = position[1]
        exponential_function = a * math.exp(y)
        return self.track_width - exponential_function + x


def generate_randint(config):
    """
    This method creates a basic np randint but makes sure there are no duplicates.
    The point here is that having two wind changes on the same day is a bit lame.
    """
    while True:
        wind_forecast = np.random.randint( # at which step the wind should be changed/set
                low=10,
                high=config.boat_env.boat_fuel - 15, # Little offset to make it more interesting
                size=config.boat_env.wind_events)
        # Check if every value is unique in the generated randint array.
        if np.sum(np.unique(wind_forecast, return_counts=True)[1]) == config.boat_env.wind_events:
            wind_forecast = np.insert(wind_forecast, 0, 0)
            break
    return wind_forecast