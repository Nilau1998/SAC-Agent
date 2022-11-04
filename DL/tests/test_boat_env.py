import os
import math
import pytest
from environments.boat_env import Boat
# from environments.boat_env import BoatEnv

# Ever tried testing with randomness? It sucks!
import numpy
numpy.random.seed(1)
import random
random.seed(1)

from utils.config_reader import get_config

class TestBoat:
    test_boat_1 = Boat(get_config("config.yaml"))
    # wind_forecast = [12 37 72]

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
        #1 move to 1   0
        #2 move to 1.7 0.7
        assert self.test_boat_1.position[0] == pytest.approx(1.7, 0.1)
        assert self.test_boat_1.position[1] == pytest.approx(0.7, 0.1)

    def test_next_wind_change(self):
        print(self.test_boat_1.wind_forecast)
        assert len(self.test_boat_1.wind_forecast) == 3
        assert self.test_boat_1.next_wind_change() == 12

        # Simulate 11 steps to get to current_step = 11 where wind change is 1 step away
        for _ in range(11):
            self.test_boat_1.current_step += 1
            self.test_boat_1.next_wind_change()

        assert len(self.test_boat_1.wind_forecast) == 3
        assert self.test_boat_1.next_wind_change() == 1

        # Simulate another step to get the step where a wind change happens
        self.test_boat_1.current_step += 1
        self.test_boat_1.next_wind_change()

        assert len(self.test_boat_1.wind_forecast) == 2
        assert self.test_boat_1.next_wind_change() == 25


    # Create new boat for wind attribute testing
    test_boat_2 = Boat(get_config("config.yaml"))
    # wind_forecast = [5 9 75]

    def test_set_wind_attributes(self):
        # Init with 0 0
        assert self.test_boat_2.wind_force == 0
        assert self.test_boat_2.wind_angle == 0

        # Simulate 10 steps for wind change
        for _ in range(10):
            self.test_boat_2.current_step += 1
            self.test_boat_2.next_wind_change()

        assert self.test_boat_2.wind_force == pytest.approx(0.76, 0.1)
        assert self.test_boat_2.wind_angle == 1


    # Create new boat for wind application testing
    test_boat_3 = Boat(get_config("config.yaml"))
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
            self.test_boat_3.current_step += 1
            self.test_boat_3.next_wind_change()
            self.test_boat_3.apply_wind()
        # Ships hasn't turned yet
        assert math.degrees(self.test_boat_3.angle) == 0

        # Ship turns hard right over 31°
        for _ in range(63-15):
            self.test_boat_3.current_step += 1
            self.test_boat_3.next_wind_change()
            self.test_boat_3.apply_wind()
        assert math.degrees(self.test_boat_3.angle) > 31

        # Ship turns a bit left by <2°
        for _ in range(79-63):
            self.test_boat_3.current_step += 1
            self.test_boat_3.next_wind_change()
            self.test_boat_3.apply_wind()
        assert math.degrees(self.test_boat_3.angle) < 29