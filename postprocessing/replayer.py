import os
import pandas as pd
import numpy as np
from utils.config_reader import get_experiment_config


class Replayer:
    """
    The replayer is used to read the csv files of an episode so that for example the renderer can render gifs.
    """

    def __init__(self, experiment_dir):
        self.experiment_dir = experiment_dir
        self.experiment_config = get_experiment_config(
            experiment_dir, os.path.join('tuned_configs.yaml'))

        self.info_data = self.read_info_csv()
        self.wind_data = self.read_wind_csv()
        self.episode_data = None

        self.total_episodes = self.read_info_csv().shape[0]
        self.total_dt = 0
        self.episode_index_memory = -1

    def read_data_csv(self, episode_index):
        if episode_index != self.episode_index_memory:
            data_file = os.path.join(
                self.experiment_dir, 'episodes', f"episode_{episode_index}_data.csv")
            self.episode_data = pd.read_csv(data_file, delimiter=';')
            self.total_dt = self.episode_data.shape[0]

            self.episode_data = pd.concat(
                [self.episode_data, self.wind_data], axis=1)

    def read_wind_csv(self):
        wind_file = os.path.join(
            self.experiment_dir, 'episodes', 'wind.csv')
        return pd.read_csv(wind_file, delimiter=';')

    def read_info_csv(self):
        info_file = os.path.join(
            self.experiment_dir, 'episodes', 'info.csv')
        return pd.read_csv(info_file, delimiter=';')

    def analyse_experiment(self):
        relevant_episodes = []
        best_episodes = []
        best_score = float('-inf')
        for episode_index in range(self.total_episodes):
            score = self.info_data.iloc[episode_index]['episode_reward']
            avg_score = np.average(
                self.info_data.iloc[-self.experiment_config.base_settings.avg_lookback:]['episode_reward'])

            if episode_index == 0 or episode_index == self.total_episodes:
                relevant_episodes.append(episode_index)

            if score > avg_score and relevant_episodes[-1] != episode_index:
                relevant_episodes.append(episode_index)

            if score > best_score and episode_index != 0:
                best_score = score
                if episode_index not in relevant_episodes:
                    relevant_episodes.append(episode_index)
                best_episodes.append(episode_index)
        return relevant_episodes, best_episodes
