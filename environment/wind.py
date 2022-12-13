import numpy as np
from scipy.interpolate import interp1d


class Wind:
    """
    This class defines the wind for the boat environment. Within this class a 2 functions will be defined together with helper methods that might come in handy.
    The first function will define a continous range which describes the winds force over a whole episode.
    The second function will define the continous range for the angle of the wind over a whole episode.
    """

    def __init__(self, config):
        self.config = config
        self.wind_range_length = int(
            config.base_settings.t_max / config.base_settings.dt)
        self.wind_force = self.generate_wind_force()
        self.wind_angle = self.generate_wind_angle()

    def get_wind(self, index):
        """
        Returns wind attributes for this timestep.
        """
        return np.array([self.wind_force[index], self.wind_angle[index]])

    def generate_wind_force(self):
        """
        Generates a range that defines the wind force that can be used to affect the boat.
        """
        if self.config.wind.fixed_points < 4:
            raise ValueError(
                "Please select at least 4 fixed_points in your config. The interpolation doesn't work otherwise!")
        fixed_points = np.linspace(
            0, self.wind_range_length, num=self.config.wind.fixed_points)
        fixed_point_values = np.random.sample(
            self.config.wind.fixed_points) * self.config.wind.max_force

        complete_range = np.linspace(0, self.wind_range_length,
                                     num=self.wind_range_length + 1, endpoint=True)
        interpolation = interp1d(
            fixed_points, fixed_point_values, kind='cubic', fill_value='extrapolate')
        interpolated_range = interpolation(complete_range)

        # Normalize incase interpolation exceeds 0-1 range.
        if np.any((interpolated_range < 0) | (interpolated_range > 1)):
            interpolated_range = (interpolated_range - np.min(interpolated_range)) / (
                np.max(interpolated_range) - np.min(interpolated_range))
        return interpolated_range

    def generate_wind_angle(self):
        """
        Generates a range that defines the wind angle that can be used with the wind force to affect the boat.
        Angle is defined in rad!
        """
        if self.config.wind.fixed_points < 4:
            raise ValueError(
                "Please select at least 4 fixed_points in your config. The interpolation doesn't work otherwise!")
        fixed_points = np.linspace(
            0, self.wind_range_length, num=self.config.wind.fixed_points)
        fixed_point_values = np.random.sample(
            self.config.wind.fixed_points)

        complete_range = np.linspace(0, self.wind_range_length,
                                     num=self.wind_range_length + 1, endpoint=True)
        interpolation = interp1d(
            fixed_points, fixed_point_values, kind='cubic', fill_value='extrapolate')
        interpolated_range = interpolation(complete_range)

        # Normalize incase interpolation exceeds 0-1 range.
        if np.any((interpolated_range < 0) | (interpolated_range > 1)):
            interpolated_range = (interpolated_range - np.min(interpolated_range)) / (
                np.max(interpolated_range) - np.min(interpolated_range))
        return interpolated_range
