from gym import Env
from gym.spaces import Box
import numpy as np
import random
import math

class BoatEnv(Env):
    def __init__(self, config, experiment=None):
        self.config = config
        self.experiment_dir = experiment.experiment_dir
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


        # Define obersvation space
        # Following states are observed:
        # x_pos, y_pos, boat_angle, current_wind_force, current_wind_angle, next_wind_angle, steps_until_wind_change
        self.low_state = np.array(
            [0, -5, math.radians(-90), 0, -1, -1, 0], dtype=np.float32
        )
        self.high_state = np.array(
            [150, 5, math.radians(90), 2, 1, 1, 120], dtype=np.float32
        )
        self.observation_space = Box(
            low=self.low_state,
            high=self.high_state,
            dtype=np.float32
        )

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

        self.boat.recalc_angle()

        self.state = self.boat.return_state()

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
            print(f"Boat ran out of fuel!")
            done = True
        elif self.boat.position[0] >= self.config.boat_env.goal_line:
            print(f"Boat reached the destination!")
            done = True
        elif abs(self.boat.position[1]) > 8:
            print(f"Boat out of bounds!")
            done = True

        info = {}

        return self.state, reward, done, info

    def render(self):
        pass

    def reset(self):
        self.boat = Boat(self.config)

        self.state = self.boat.return_state()

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
        self.steps_until_wind_change = 0
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
        if len(self.wind_forecast) is 0:
            pass
        else:
            self.steps_until_wind_change = self.wind_forecast[0] - self.current_step
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

    def return_state(self):
        state = np.array([
            self.position[0],
            self.position[1],
            self.angle,
            self.current_wind_force,
            self.current_wind_angle,
            self.next_wind_angle,
            self.steps_until_wind_change
        ])
        return state


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