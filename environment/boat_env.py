from gym import Env
from gym.spaces import Box
from environment.reward_functions import RewardFunction
from environment.wind import Wind
from control_theory.control_blocks import Integrator, Scope
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
            config=config,
            a=1
        )

        self.info = {
            'termination': '',
            'reached_goal': 0,
            'out_of_bounds': 0,
            'out_of_fuel': 0,
            'timeout': 0,
            'episode_reward': 0
        }

        # Define action space
        # Following actions can be choosen:
        # rad rudder, n
        self.action_space = Box(
            low=np.array([-0.1, -1]),
            high=np.array([0.1, 1]),
            shape=(2,),
            dtype=np.float32
        )

        # Define obersvation space
        # Following states are observed:
        # x_pos, y_pos, boat_angle, rudder angle, n, fuel
        self.low_state = np.array(
            [0, -(self.config.boat_env.track_width / 2), 0, -np.pi/4, 0, 0], dtype=np.float32
        )
        self.high_state = np.array(
            [3900, (self.config.boat_env.track_width / 2), 2*np.pi, np.pi/4, self.config.boat.n_max, self.config.boat.fuel], dtype=np.float32
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
        self.boat.t += self.boat.dt

        self.boat.fuel -= 1

        self.boat.rudder_angle += action[0]
        if self.boat.rudder_angle > np.pi/4:
            self.boat.rudder_angle = np.pi/4
        elif self.boat.rudder_angle < -np.pi/4:
            self.boat.rudder_angle = -np.pi/4
        self.boat.n += action[1]
        if self.boat.n > self.config.boat.n_max:
            self.boat.n = self.config.boat.n_max
        elif self.boat.n < 0:
            self.boat.n = 0

        self.boat.run_model_step()

        self.state = self.boat.return_state()

        # Reward calculation
        self.reward = self.reward_function.linear_reward(
            position=[self.boat.kinematics[4], self.boat.kinematics[5]])
        self.info['episode_reward'] += self.reward

        # Check if goal is reached, or if the boat ran out of fuel
        # Timeout
        done = False
        if self.boat.kinematics[4] >= self.config.boat_env.goal_line:
            done = True
            self.info['termination'] = 'reached_goal'
            self.info['reached_goal'] += 1
        elif abs(self.boat.kinematics[5]) > self.boat.out_of_bounds or self.boat.kinematics[4] < 0:
            done = True
            self.info['termination'] = 'out_of_bounds'
            self.info['out_of_bounds'] += 1
        elif self.boat.fuel < 0:
            done = True
            self.info['termination'] = 'out_of_fuel'
            self.info['out_of_fuel'] += 1
        elif self.boat.t_max <= self.boat.t:
            done = True
            self.info['termination'] = 'timeout'
            self.info['timeout'] += 1

        return self.state, self.reward, done, self.info

    def render(self):
        pass

    def reset(self):
        self.boat = Boat(self.config)

        self.state = self.boat.return_state()

        self.info['episode_reward'] = 0

        info = {}

        return self.state, info

    def return_all_data(self):
        data = {
            # Boat
            'boat_position_x': self.boat.kinematics[4],
            'boat_position_y': self.boat.kinematics[5],
            'boat_velocity_x': self.boat.v_x,
            'boat_velocity_y': self.boat.v_y,
            'boat_angle': self.boat.kinematics[3],
            'boat_mass': self.config.boat.fuel,
            # Agent
            'action': self.action[0],
            'reward': self.reward,
            'rudder_angle': self.boat.rudder_angle,
            'n': self.boat.n
        }
        return data


class Boat:
    def __init__(self, config):
        self.config = config

        self.fuel = config.boat.fuel

        self.t = 0
        self.dt = config.base_settings.dt
        self.t_max = config.boat.fuel
        self.index = 0  # Used to access arrays since dt is a float not an int
        self.wind = Wind(config)

        self.a_x_integrator = Integrator(initial_value=1)
        self.a_x_integrator.dt = self.dt
        self.v_x_integrator = Integrator()
        self.v_x_integrator.dt = self.dt

        self.a_y_integrator = Integrator()
        self.a_y_integrator.dt = self.dt
        self.v_y_integrator = Integrator()
        self.v_y_integrator.dt = self.dt

        self.a_r_integrator = Integrator()
        self.a_r_integrator.dt = self.dt
        self.v_r_integrator = Integrator()
        self.v_r_integrator.dt = self.dt

        # Boat
        self.n = 0
        self.rudder_angle = 0

        self.a_x = 0
        self.v_x = 0

        self.a_y = 0
        self.v_y = 0

        self.a_r = 0
        self.v_r = 0

        self.kinematics = self.get_kinematics()

        self.out_of_bounds = self.config.boat_env.track_width + \
            self.config.boat_env.boat_out_of_bounds_offset

    def run_model_step(self):
        self.eom_longitudinal()
        self.v_x = self.a_x_integrator.integrate_signal(self.a_x)
        self.eom_transverse()
        self.v_y = self.a_y_integrator.integrate_signal(self.a_y)
        self.eom_yawning()
        self.v_r = self.a_r_integrator.integrate_signal(self.a_r)
        self.kinematics = self.get_kinematics()
        self.index += 1

    def eom_longitudinal(self):
        params = self.config.boat

        # Resistance F_R
        F_R = np.square(self.v_x) * params.c_r_front * 0.5 * \
            params.rho * params.boat_area_front

        # Thrust F_T
        v_x_w = self.v_x * (1 - params.wake_friction)
        J = 0
        if self.n != 0:
            J = v_x_w / (self.n * params.propeller_diameter)
        KT = np.sin(J)
        F_T = KT * np.square(self.n) * params.rho * np.power(
            params.propeller_diameter, 4) * (1 - params.thrust_deduction)

        # Centrifugal force F_C
        F_C = self.v_y * (params.boat_m + params.boat_m_y) * self.v_r

        self.a_x = (-F_R + F_T + F_C) / (params.boat_m + params.boat_m_x)

    def eom_transverse(self):
        params = self.config.boat
        v_y_sign = np.sign(self.v_y)

        # Resistance F_R
        F_R = np.square(self.v_y) * params.c_r_side * 0.5 * \
            params.rho * params.boat_area_side * v_y_sign

        # Rudder force F_RU
        F_RU = np.square(self.v_x) * params.c_r_front * \
            0.5 * params.rho * params.rudder_area
        F_RU = np.sin(self.rudder_angle) * F_RU

        # Centrifugal force F_C
        F_C = self.v_x * (params.boat_m + params.boat_m_x) * self.v_r

        self.a_y = (-F_R + F_RU + F_C) / (params.boat_m + params.boat_m_y)

    def eom_yawning(self):
        params = self.config.boat
        v_r_sign = np.sign(self.v_r)
        v_x_sign = np.sign(self.v_x)

        # Momentum from hull M_hull
        M_hull = np.square(self.v_r) * params.c_r_side * 0.5 * params.rho * \
            params.boat_area_side * params.boat_l * 5 * v_r_sign

        # Moment from rudder M_rudder
        M_rudder = np.square(self.v_x) * params.c_r_side * 0.5 * params.rho * \
            params.rudder_area * \
            np.sin(self.rudder_angle) * (params.boat_b / 2) * v_x_sign

        self.a_r = (-M_hull + M_rudder) / (params.boat_I + params.boat_Iz)

    def get_kinematics(self):
        params = self.config.boat
        # v
        v = np.sqrt(np.square(self.v_x) + np.square(self.v_y))

        # drift_angle
        drift_angle = np.arctan2(self.v_x, self.v_y)

        # turning rate
        turning_rate = 0
        if v != 0:
            turning_rate = (self.v_r * params.boat_l) / v

        # heading
        s_r = self.v_r_integrator.integrate_signal(self.v_r)
        heading = s_r

        # s_x in new coordinate system
        direction = drift_angle - heading
        v_x_new = np.sin(direction) * \
            (np.square(self.v_x) + np.square(self.v_y))
        s_x = self.v_x_integrator.integrate_signal(v_x_new)

        # s_y in new coordinate system
        v_y_new = np.cos(direction) * \
            (np.square(self.v_x) + np.square(self.v_y))
        s_y = self.v_y_integrator.integrate_signal(v_y_new)
        return [v, drift_angle, turning_rate, heading, s_x, s_y]

    def return_state(self):
        state = np.array([
            self.kinematics[4],
            self.kinematics[5],
            self.kinematics[3],
            self.rudder_angle,
            self.n,
            self.fuel
        ])
        return state
