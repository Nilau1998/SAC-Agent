import os
import torch as T
import torch.nn as nn

class BaseNetwork(nn.Module):
    def __init__(self, name, experiment_dir):
        super(BaseNetwork, self).__init__()
        self.experiment_dir = experiment_dir
        self.checkpoints_dir = os.path.join(experiment_dir, "checkpoints")
        self.checkpoint_file = os.path.join(self.checkpoints_dir, name)

    def save_checkpoint(self):
        T.save(self.state_dict(),  self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(T.load(self.checkpoint_file))