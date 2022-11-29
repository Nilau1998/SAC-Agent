from gym import Env
from gym.spaces import Box
from environment.reward_functions import RewardFunction
from environment.wind import Wind
import numpy as np
import math


class BoatEnv(Env):
    def __init__(self, config, experiment=None):
        self.config = config
        self.experiment_dir = experiment.experiment_dir
        self.action = [0]
        self.reward = 0
        self.boat = Boat(self.config)
        self.reward_function = RewardFunction(
            track_width=self.config.boat_env.track_width,
            a=1
        )

        self.info = {
            "termination": "",
            "reached_goal": 0,
            "out_of_bounds": 0,
            "episode_reward": 0
        }

        # Define action space
        self.min_action = -1.0 * self.config.boat_env.boat_angle_scale
        self.max_action = 1.0 * self.config.boat_env.boat_angle_scale
        self.action_space = Box(
            low=self.min_action,
            high=self.max_action,
            shape=(1,),
            dtype=np.float32
        )

        # Define obersvation space
        # Following states are observed:
        # x_pos, y_pos, boat_angle, current_wind_force, current_wind_angle
        self.low_state = np.array(
            [0, -5, math.radians(-90), 0, -1], dtype=np.float32
        )
        self.high_state = np.array(
            [150, 5, math.radians(90), 1, 1], dtype=np.float32
        )
        self.observation_space = Box(
            low=self.low_state,
            high=self.high_state,
            dtype=np.float32
        )

    # Define the step an agent can take. This means that within a step the agent can change the boats angle
    # and additionally all other calulcations happen like the wind impact.
    def step(self, action):
        self.action = action
        self.boat.dt += 1

        # Apply action
        self.boat.set_velocities(action[0])

        # Apply wind
        self.boat.apply_wind()

        # Finish action with boat reposition after direction change and wind application
        self.boat.set_boat_position()

        self.state = self.boat.return_state()

        # Reward calculation
        self.reward = self.reward_function.linear_reward(
            position=self.boat.position)
        self.info["episode_reward"] += self.reward

        # Check if goal is reached, or if the boat ran out of fuel
        # Timeout
        done = False
        if self.boat.position[0] >= self.config.boat_env.goal_line:
            done = True
            self.info["termination"] = "reached_goal"
            self.info["reached_goal"] += 1
        elif abs(self.boat.position[1]) > self.boat.out_of_bounds:
            done = True
            self.info["termination"] = "out_of_bounds"
            self.info["out_of_bounds"] += 1

        return self.state, self.reward, done, self.info

    def render(self):
        pass

    def reset(self):
        self.boat = Boat(self.config)

        self.state = self.boat.return_state()

        self.info["episode_reward"] = 0

        info = {}

        return self.state, info

    def return_all_data(self):
        data = {
            # Boat
            "boat_position_x": self.boat.position[0],
            "boat_position_y": self.boat.position[1],
            "boat_velocity_x": self.boat.velocity[0],
            "boat_velocity_y": self.boat.velocity[1],
            "boat_angle": self.boat.angle,
            "boat_mass": self.boat.mass,
            "wind_angle": self.boat.wind.get_wind(self.boat.dt)[1],
            "wind_force": self.boat.wind.get_wind(self.boat.dt)[0],
            # Agent
            "action": self.action[0],
            "reward": self.reward
        }
        return data


class Boat:
    def __init__(self, config):
        self.config = config

        # Env
        self.dt = 0
        self.wind = Wind(config)

        # Boat
        self.position = np.array([0, 0], dtype=np.float32)
        self.velocity = np.array(
            [self.config.boat_env.boat_velocity, 0], dtype=np.float32)
        self.angle = 0.0  # In relation to x-axis, +y = +angle, -y = -angle, based on velocity
        self.out_of_bounds = self.config.boat_env.track_width + \
            self.config.boat_env.boat_out_of_bounds_offset
        self.mass = 0  # Not implemented for now

    def set_boat_position(self):
        """
        Moves the ship based on the current velocity if a step passes.
        """
        self.position += self.velocity

    def set_velocities(self, d_angle):
        """
        Rotates the velocity vector by delta_angle. The absolute value of the velocity vector stays constant.
        delta_angle is passed in degree
        """
        self.angle += math.radians(d_angle)
        abs_velocity = math.sqrt(
            math.pow(self.velocity[0], 2) + math.pow(self.velocity[1], 2))
        self.velocity[0] = abs_velocity * math.cos(self.angle)
        self.velocity[1] = abs_velocity * math.sin(self.angle)

    def apply_wind(self):
        """
        Applies the wind on the boat.
        Wind can blow from either 90° or -90° in relation to the x-axis. (90° towads -y, -90° is towards y).
        Wind changes the rotation of the boat and therefore gives it a little challenge the agent has to solve,
        the challenge is increased by the wind changes that happen after every couple steps.
        """
        # Wind only affects the boat angle for now, wind comes from 90° or -90°
        wind_attr = self.wind.get_wind(self.dt)
        self.set_velocities(wind_attr[0])

    def return_state(self):
        state = np.array([
            self.position[0],
            self.position[1],
            self.angle,
            self.wind.get_wind(self.dt)[0],
            self.wind.get_wind(self.dt)[1]
        ])
        return state
