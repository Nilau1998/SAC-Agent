from utils.config_reader import get_config
import random
import os
import math
import pytest
from environments.boat_env import Boat, RewardFunction
# from environments.boat_env import BoatEnv

# Ever tried testing with randomness? It sucks!
import numpy
numpy.random.seed(1)
random.seed(1)


class TestBoat:
    config = get_config("config.yaml")
    test_boat_1 = Boat(config)
    # wind_forecast = [0 22 47 82]
    # print(self.test_boat_1.wind_forecast)

    def test_simple_forward_move(self):
        assert self.test_boat_1.position[0] == 0
        assert self.test_boat_1.position[1] == 0

        self.test_boat_1.set_boat_position()

        assert self.test_boat_1.position[0] == self.test_boat_1.config.test.boat_velocity
        assert self.test_boat_1.position[1] == 0

    def test_angle_change(self):
        assert self.test_boat_1.angle == 0

        self.test_boat_1.set_velocities(45)

        assert math.degrees(self.test_boat_1.angle) == 45

    def test_angled_move_forward(self):
        assert self.test_boat_1.position[0] == self.test_boat_1.config.test.boat_velocity
        assert self.test_boat_1.position[1] == 0

        self.test_boat_1.set_boat_position()
        # 1 move to 1   0
        # 2 move to 1.7 0.7
        assert self.test_boat_1.position[0] == pytest.approx(1.7, 0.1)
        assert self.test_boat_1.position[1] == pytest.approx(0.7, 0.1)

    def test_next_wind_change(self):
        assert len(self.test_boat_1.wind_forecast) == 4
        assert self.test_boat_1.next_wind_change() == 0

        # Simulate 21 steps to get to current_step = 11 where wind change is 1 step away
        for _ in range(21):
            self.test_boat_1.dt += 1
            self.test_boat_1.next_wind_change()

        assert len(self.test_boat_1.wind_forecast) == 3
        assert self.test_boat_1.next_wind_change() == 1

        # Simulate another step to get the step where a wind change happens
        self.test_boat_1.dt += 1
        self.test_boat_1.next_wind_change()

        assert len(self.test_boat_1.wind_forecast) == 2
        assert self.test_boat_1.next_wind_change() == 25

    # Create new boat for wind attribute testing
    test_boat_2 = Boat(config)
    # wind_forecast = [0 15 19 85]
    # print(self.test_boat_2.wind_forecast)

    def test_set_wind_attributes(self):
        # Init with 0 0
        assert self.test_boat_2.current_wind_force == 0
        assert self.test_boat_2.current_wind_angle == 0

        # Simulate 10 steps for wind change
        for _ in range(17):
            self.test_boat_2.dt += 1
            self.test_boat_2.next_wind_change()

        assert self.test_boat_2.current_wind_force == pytest.approx(0.76, 0.1)
        assert self.test_boat_2.current_wind_angle == 1

    # Create new boat for wind application testing
    test_boat_3 = Boat(config)
    print(test_boat_3.wind_forecast)
    # wind_forecast = [16 64 79]
    # wf, wa
    # 0 , 0
    # 1 , 0.651
    #-1 , 0.093
    #-1 , 0.893

    def test_apply_wind(self):
        # Simulate 120 steps with the boat sitting still to have wind being applied
        # Ship will not turn, turn hard right, then turn hard left, then turn soft left
        for _ in range(15):
            self.test_boat_3.dt += 1
            self.test_boat_3.next_wind_change()
            self.test_boat_3.apply_wind()
        # Ships hasn't turned yet
        assert math.degrees(self.test_boat_3.angle) == 0

        # Ship turns hard right over 31°
        for _ in range(63-15):
            self.test_boat_3.dt += 1
            self.test_boat_3.next_wind_change()
            self.test_boat_3.apply_wind()
        assert math.degrees(self.test_boat_3.angle) > 31

        # Ship turns a bit left by <2°
        for _ in range(79-63):
            self.test_boat_3.dt += 1
            self.test_boat_3.next_wind_change()
            self.test_boat_3.apply_wind()
        assert math.degrees(self.test_boat_3.angle) < 29


# class TestRewards:
#     config = get_config("config.yaml")
#     reward_function = RewardFunction(config.test.track_width)

#     # Position cases
#     test_cases = [
#         [0, 0], # Origin
#         # Upper tracks half
#         [5, config.test.track_width - 1],
#         [5, config.test.track_width],
#         [5, config.test.track_width + 1],
#         # Lower tracks half
#         [5, -config.test.track_width + 1],
#         [5, -config.test.track_width],
#         [5, config.test.track_width - 1]
#     ]

#     def test_linear_reward(self):
#         # Track_width = 5
#         # Best score -> 5, worst score -> 0
#         print(self.reward_function.linear_reward(self.test_position_1, 1))
#         assert 2==3
