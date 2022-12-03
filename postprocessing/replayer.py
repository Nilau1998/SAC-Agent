import os
import pandas as pd
from utils.config_reader import get_experiment_config


class Replayer:
    """
    The replayer is used to read the csv files of an episode so that for example the renderer can render gifs.
    """

    def __init__(self, experiment_dir):
        self.experiment_dir = experiment_dir
        self.info_data = self.read_info_csv()
        self.episode_data = None
        self.experiment_config = get_experiment_config(
            experiment_dir, os.path.join('config.yaml'))
        self.total_episodes = self.read_info_csv().shape[0]
        self.total_dt = 0
        self.episode_index_memory = -1

    def read_data_csv(self, episode_index):
        if episode_index != self.episode_index_memory:
            data_file = os.path.join(
                self.experiment_dir, 'episodes', f"episode_{episode_index}_data.csv")
            self.episode_data = pd.read_csv(data_file, delimiter=';')
            self.total_dt = self.episode_data.shape[0]

            wind_file = os.path.join(
                self.experiment_dir, 'episodes', 'wind.csv')
            wind_df = pd.read_csv(wind_file, delimiter=';')
            self.episode_data = pd.concat([self.episode_data, wind_df], axis=1)

    def read_info_csv(self):
        info_file = os.path.join(
            self.experiment_dir, 'episodes', 'info.csv')
        return pd.read_csv(info_file, delimiter=';')
