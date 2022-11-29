import os
import pandas as pd


class Replayer:
    """
    The replayer is used to read the csv files of an episode so that for example the renderer can render gifs.
    """

    def __init__(self, env):
        self.experiment_dir = env.experiment_dir
        self.info_data = self.read_info_csv()

    def read_data_csv(self, episode_index):
        data_file = os.path.join(
            self.experiment_dir, "episodes", f"episode_{episode_index}_data.csv")
        return pd.read_csv(data_file, delimiter=";")

    def read_info_csv(self):
        info_file = os.path.join(
            self.experiment_dir, "episodes", "info.csv")
        return pd.read_csv(info_file, delimiter=";")
