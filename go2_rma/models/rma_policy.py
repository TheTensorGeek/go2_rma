# go2_rma/models/rma_policy.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class RMAPolicy(nn.Module):
    """
    Simple RMA-style policy:
    - Input: proprio obs x (obs_dim) and extrinsics z (z_dim)
    - Output: Gaussian mean actions (act_dim) and value estimate V.
    """

    def __init__(self, obs_dim: int, act_dim: int, z_dim: int, hidden_sizes=(256, 256)):
        super().__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.z_dim = z_dim

        input_dim = obs_dim + z_dim

        # Actor MLP
        layers = []
        last = input_dim
        for hs in hidden_sizes:
            layers.append(nn.Linear(last, hs))
            layers.append(nn.LeakyReLU())
            last = hs
        layers.append(nn.Linear(last, act_dim))
        self.actor = nn.Sequential(*layers)

        # Critic MLP
        v_layers = []
        last = input_dim
        for hs in hidden_sizes:
            v_layers.append(nn.Linear(last, hs))
            v_layers.append(nn.LeakyReLU())
            last = hs
        v_layers.append(nn.Linear(last, 1))
        self.critic = nn.Sequential(*v_layers)

        # Log std as a parameter (shared across actions)
        self.log_std = nn.Parameter(torch.zeros(act_dim))

    def forward(self, obs: torch.Tensor, z: torch.Tensor):
        """
        Forward pass: returns mean actions and values.
        obs: [B, obs_dim]
        z:   [B, z_dim]
        """
        x = torch.cat([obs, z], dim=-1)
        mean = self.actor(x)
        value = self.critic(x).squeeze(-1)
        return mean, value

    def get_action_and_value(self, obs: torch.Tensor, z: torch.Tensor):
        """
        Sample action and compute log-prob + value.
        """
        mean, value = self.forward(obs, z)
        std = self.log_std.exp().expand_as(mean)
        dist = torch.distributions.Normal(mean, std)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(dim=-1)
        return action, log_prob, value

    def evaluate_actions(self, obs: torch.Tensor, z: torch.Tensor, actions: torch.Tensor):
        """
        Used by PPO to compute log_prob and entropy for given actions.
        """
        mean, value = self.forward(obs, z)
        std = self.log_std.exp().expand_as(mean)
        dist = torch.distributions.Normal(mean, std)
        log_prob = dist.log_prob(actions).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        return log_prob, entropy, value
