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

    def write_data(self, current_step):
        csv_file_path = os.path.join(self.experiment_dir,
                                     "episodes", f"episode_{current_step}.csv")
        if not os.path.exists(csv_file_path):
            with open(csv_file_path, "x") as csv_file:
                pass
        with open(csv_file_path, "a") as csv_file:
            writer = csv.writer(csv_file, delimiter=";")
            writer.writerow(self.env.return_all_data())
