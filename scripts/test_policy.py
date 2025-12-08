# scripts/test_policy.py

import torch
from go2_rma.models.rma_policy import RMAPolicy

def main():
    obs_dim = 48
    act_dim = 12
    z_dim = 8

    policy = RMAPolicy(obs_dim, act_dim, z_dim)
    obs = torch.randn(2, obs_dim)
    z = torch.randn(2, z_dim)

    action, logp, value = policy.get_action_and_value(obs, z)
    print("Action shape:", action.shape)
    print("Log prob shape:", logp.shape)
    print("Value shape:", value.shape)

if __name__ == "__main__":
    main()
