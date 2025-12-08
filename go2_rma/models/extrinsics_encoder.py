# go2_rma/models/extrinsics_encoder.py

import torch
import torch.nn as nn

class ExtrinsicsEncoder(nn.Module):
    """
    Encodes a short history ("context") of:
        [obs_t, action_t, reward_t]
    into a low-dimensional extrinsics vector z.

    context input shape: [B, T, ctx_dim]
        where ctx_dim = obs_dim + act_dim + 1 (reward)
    """

    def __init__(self, obs_dim: int, act_dim: int, context_len: int, z_dim: int = 8):
        super().__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.context_len = context_len
        self.z_dim = z_dim

        ctx_dim = obs_dim + act_dim + 1

        self.rnn = nn.GRU(input_size=ctx_dim, hidden_size=128, num_layers=1, batch_first=True)
        self.fc = nn.Sequential(
            nn.Linear(128, 64),
            nn.LeakyReLU(),
            nn.Linear(64, z_dim)
        )

    def forward(self, context: torch.Tensor) -> torch.Tensor:
        """
        context: [B, T, ctx_dim]
        returns: z [B, z_dim]
        """
        _, h = self.rnn(context)   # h: [1, B, 128]
        h = h.squeeze(0)
        z = self.fc(h)
        return z
