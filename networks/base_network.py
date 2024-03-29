import os
import torch as T
import torch.nn as nn

class BaseNetwork(nn.Module):
    def __init__(self, name, experiment_dir):
        super(BaseNetwork, self).__init__()
        self.name = name
        self.experiment_dir = experiment_dir
        self.checkpoints_dir = os.path.join(experiment_dir, "checkpoints")
        self.checkpoint_file = os.path.join(self.checkpoints_dir, name)

    def save_checkpoint(self):
        T.save(self.state_dict(),  self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(T.load(self.checkpoint_file))

    def return_summed_weights(self):
        total_sum = 0
        for layer in self.children():
            if isinstance(layer, nn.Linear):
                total_sum += T.sum(layer.state_dict()["weight"])
        print(f"{self.name}: {total_sum:.2f}")