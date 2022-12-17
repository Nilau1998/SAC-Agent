import os
import torch as T
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from networks.base_network import BaseNetwork

"""
input_dims = NN input vector
n_actions = NN output vector
"""


class ActorNetwork(BaseNetwork):
    def __init__(self, experiment_dir, alpha, input_dims, max_action, fc1_dims=256, fc2_dims=256, n_actions=2, name="actor_network"):
        super().__init__(name, experiment_dir)
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions
        self.max_action = max_action
        self.reparam_noise = 1e-6

        # Define layers
        self.fc1 = nn.Linear(*self.input_dims, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        self.mean = nn.Linear(self.fc2_dims, self.n_actions)
        self.std = nn.Linear(self.fc2_dims, self.n_actions)

        # Define optimizer
        self.optimizer = optim.Adam(self.parameters(), lr=alpha)

        # Set device
        self.device = T.device("cuda:0" if T.cuda.is_available() else "cpu")
        self.to(self.device)

    def forward(self, state):
        prob = F.relu(self.fc1(state))
        prob = F.relu(self.fc2(prob))

        mean = self.mean(prob)
        std = self.std(prob)

        return mean, std

    def sample_normal(self, state, reparameterize=True):
        LOG_STD_MAX = 2
        LOG_STD_MIN = -5  # https://github.com/vwxyzjn/cleanrl/blob/401a4bedf974d0a80b3dcf330c65195fe29cf6cf/cleanrl/sac_continuous_action.py#L107

        mean, std = self.forward(state)

        log_std = T.tanh(std)
        log_std = LOG_STD_MIN + 0.5 * \
            (LOG_STD_MAX - LOG_STD_MIN) * (log_std + 1)
        std = log_std.exp()

        normal = T.distributions.Normal(mean, std)

        if reparameterize:
            actions = normal.rsample()
        else:
            actions = normal.sample()

        action = T.tanh(actions) * T.tensor(self.max_action).to(self.device)
        log_probs = normal.log_prob(actions)
        log_probs -= T.log(1 - action.pow(2) + self.reparam_noise)
        log_probs = log_probs.sum(1, keepdim=True)

        return action, log_probs


class CriticNetwork(BaseNetwork):
    def __init__(self, experiment_dir, beta, input_dims, n_actions, fc1_dims=256, fc2_dims=256, name="critic_network"):
        super(CriticNetwork, self).__init__(name, experiment_dir)
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions

        # Define layers
        self.fc1 = nn.Linear(self.input_dims[0] + n_actions, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        self.q = nn.Linear(self.fc2_dims, 1)

        # Define optimizer
        self.optimizer = optim.Adam(self.parameters(), lr=beta)

        # Set device
        self.device = T.device("cuda:0" if T.cuda.is_available() else "cpu")
        self.to(self.device)

    def forward(self, state, action):
        action_value = self.fc1(T.cat([state, action], dim=1))
        action_value = F.relu(action_value)
        action_value = self.fc2(action_value)
        action_value = F.relu(action_value)

        q = self.q(action_value)

        return q


class ValueNetwork(BaseNetwork):
    def __init__(self, experiment_dir, beta, input_dims, fc1_dims=256, fc2_dims=256, name="value_network"):
        super().__init__(name, experiment_dir)
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims

        # Define layers
        self.fc1 = nn.Linear(*self.input_dims, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        self.v = nn.Linear(self.fc2_dims, 1)

        # Define optimizer
        self.optimizer = optim.Adam(self.parameters(), lr=beta)

        # Set device
        self.device = T.device("cuda:0" if T.cuda.is_available() else "cpu")
        self.to(self.device)

    def forward(self, state):
        state_value = self.fc1(state)
        state_value = F.relu(state_value)
        state_value = self.fc2(state_value)
        state_value = F.relu(state_value)

        v = self.v(state_value)

        return v
