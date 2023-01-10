import os
import time
from distutils.dir_util import copy_tree


class Experiment:
    def __init__(self, experiment_name='experiment', subdir=None) -> None:
        self.timestamp = time.strftime('_%m-%d_%H-%M-%S%f')[:-2]
        self.experiments_dir = 'experiments'
        if subdir != None:
            self.experiments_dir = os.path.join('experiments', subdir)

        # Create parent directiory for all experiments
        if not os.path.exists(self.experiments_dir):
            os.makedirs(self.experiments_dir)

        self.experiment_name = experiment_name + self.timestamp
        self.experiment_dir = os.path.join(
            self.experiments_dir,
            self.experiment_name
        )

        # Create subfolders in current experiment
        os.makedirs(os.path.join(self.experiment_dir, 'plots'))
        os.makedirs(os.path.join(self.experiment_dir, 'checkpoints'))
        os.makedirs(os.path.join(self.experiment_dir, 'configs'))
        os.makedirs(os.path.join(self.experiment_dir, 'rendering'))
        os.makedirs(os.path.join(self.experiment_dir, 'episodes'))

        print(f"Created: {self.experiment_dir}")

    def save_configs(self):
        copy_tree('configs', os.path.join(self.experiment_dir, 'configs'))
