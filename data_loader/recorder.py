import csv
import os


class Recorder:
    """
    Records the data of an episode and stores it into a csv for later use. Can for example be used by the env renderer.
    """

    def __init__(self, config, env):
        self.config = config
        self.env = env
        self.experiment_dir = env.experiment_dir
        self.current_episode_file = None

    def create_csv(self, episode_index):
        self.current_episode_file = os.path.join(self.experiment_dir,
                                                 "episodes", f"episode_{episode_index}.csv")
        if not os.path.exists(self.current_episode_file):
            with open(self.current_episode_file, "x") as csv_file:
                writer = csv.writer(csv_file, delimiter=";")
                writer.writerow(self.env.return_all_data().keys())

    def write_to_csv(self):
        with open(self.current_episode_file, "a") as csv_file:
            writer = csv.writer(csv_file, delimiter=";")
            writer.writerow(self.env.return_all_data().values())
