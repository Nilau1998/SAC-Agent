from dataclasses import dataclass
from pathlib import Path
import yaml
from dotmap import DotMap
import random


class HPTuner:
    def __init__(self, base_config_file):
        self.base_config = yaml.safe_load(
            Path('configs', base_config_file).read_text())
        self.base_config_dotmap = DotMap(self.base_config)
        self.hpset = HPSet(
            alpha=self.alpha(),
            beta=self.beta(),
            gamma=self.gamma(),
            tau=self.tau()
        )

    def alpha(self):
        return round(random.uniform(
            self.base_config_dotmap.agent.alpha_min,
            self.base_config_dotmap.agent.alpha_max
        ), 4)

    def beta(self):
        return round(random.uniform(
            self.base_config_dotmap.agent.beta_min,
            self.base_config_dotmap.agent.beta_max
        ), 4)

    def gamma(self):
        self.base_config_dotmap.agent.test = 5
        return round(random.uniform(
            self.base_config_dotmap.agent.gamma_min,
            self.base_config_dotmap.agent.gamma_max
        ), 4)

    def tau(self):
        return round(random.uniform(
            self.base_config_dotmap.agent.tau_min,
            self.base_config_dotmap.agent.tau_max
        ), 4)

    def set_config_file(self, config):
        config.agent.learning_rate_alpha = self.hpset.alpha
        config.agent.learning_rate_beta = self.hpset.beta
        config.agent.gamma = self.hpset.gamma
        config.agent.tvn_parameter_modulation_tau = self.hpset.tau
        config_dict = config.toDict()
        with open(Path('configs', 'config.yaml'), 'w') as outfile:
            yaml.dump(config_dict, outfile, default_flow_style=False)


@dataclass
class HPSet:
    alpha: float
    beta: float
    gamma: float
    tau: float


if __name__ == '__main__':
    hptuner = HPTuner('hp_configs.yaml')
