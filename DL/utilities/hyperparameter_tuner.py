from pathlib import Path
import yaml
from dotmap import DotMap
import numpy as np

class HPTuner:
    def __init__(self, base_config_file):
        self.base_config = yaml.safe_load(Path("configs", base_config_file).read_text())
        self.base_config_dotmap = DotMap(self.base_config)
        self.hpset = HPSet()

    def build_hpset(self):
        self.set_layer_count()
        self.set_layer_sizes()
        self.set_learn_rates()

    def set_layer_count(self):
        self.hpset.layer_count = np.random.choice(
            list(
                range(
                    self.base_config_dotmap.hp.layer_count_min,
                    self.base_config_dotmap.hp.layer_count_max + 1,
                )
            )
        )

    def set_layer_sizes(self):
        if self.hpset.layer_count is None:
            raise ValueError(f"Layer count not set yet!")
        layer_sizes = np.empty((self.hpset.layer_count))
        for i, element in enumerate(layer_sizes):
            layer_sizes[i] = np.random.choice(
                [self.base_config_dotmap.hp.layer_size_min,
                self.base_config_dotmap.hp.layer_size_max + 1]
            )
        self.hpset.layer_sizes = layer_sizes

    def set_learn_rates(self):
        self.hpset.lr_alpha = np.random.choice(
            [self.base_config_dotmap.hp.alpha_min,
            self.base_config_dotmap.hp.alpha_max],
            0.0001
        )




class HPSet:
    def __init__(self):
        self.layer_count = None
        self.layer_sizes = None
        self.lr_alpha = None
        self.lr_beta = None



if __name__ == '__main__':
    hptuner = HPTuner("base_config.yaml")
    hptuner.build_hpset()
    print(hptuner.hpset.lr_alpha)