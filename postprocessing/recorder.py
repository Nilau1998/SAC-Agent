import csv
import os
import pandas as pd


class Recorder:
    """
    Records the data of an episode and stores it into a csv for later use. Can for example be used by the env renderer.
    """

    def __init__(self, env):
        self.env = env
        self.experiment_dir = env.experiment_dir
        self.data_file = None
        self.info_file = None

    def create_csvs(self, episode_index):
        self.data_file = os.path.join(
            self.experiment_dir, "episodes", f"episode_{episode_index}_data.csv")
        if not os.path.exists(self.data_file):
            with open(self.data_file, "x") as csv_file:
                writer = csv.writer(csv_file, delimiter=";")
                writer.writerow(self.env.return_all_data().keys())

        self.info_file = os.path.join(
            self.experiment_dir, "episodes", "info.csv")
        if not os.path.exists(self.info_file):
            with open(self.info_file, "x") as csv_file:
                writer = csv.writer(csv_file, delimiter=";")
                writer.writerow(self.env.info.keys())

    def write_data_to_csv(self):
        with open(self.data_file, "a") as csv_file:
            writer = csv.writer(csv_file, delimiter=";")
            writer.writerow(self.env.return_all_data().values())

    def write_info_to_csv(self):
        with open(self.info_file, "a") as csv_file:
            writer = csv.writer(csv_file, delimiter=";")
            writer.writerow(self.env.info.values())
