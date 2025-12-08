# scripts/test_encoder.py

import torch
from go2_rma.models.extrinsics_encoder import ExtrinsicsEncoder

def main():
    obs_dim = 48
    act_dim = 12
    context_len = 40
    z_dim = 8

    encoder = ExtrinsicsEncoder(obs_dim, act_dim, context_len, z_dim)
    ctx_dim = obs_dim + act_dim + 1
    context = torch.randn(3, context_len, ctx_dim)  # batch size 3

    z = encoder(context)
    print("z shape:", z.shape)

if __name__ == "__main__":
    main()
