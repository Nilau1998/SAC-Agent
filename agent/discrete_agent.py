import torch as T
import torch.nn.functional as F
from agent.buffer import ReplayBuffer
from agent.base_agent import Agent
from networks.networks import ActorNetwork, CriticNetwork, ValueNetwork

class DiscreteAgent(Agent):
    def __init__(self, config, experiment_dir, input_dims, env):
        super().__init__(env)
        self.config = config
        self.gamma = config.agent.gamma
        self.tau = config.agent.tau
        self.memory = ReplayBuffer(config.agent.max_size, input_dims, self.get_n_actions())
        self.batch_size = config.agent.batch_size