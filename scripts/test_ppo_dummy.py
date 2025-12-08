# scripts/test_ppo_dummy.py

import numpy as np
import torch

from go2_rma.envs.dummy_go2_env import DummyGo2Env
from go2_rma.utils.context_buffer import ContextBuffer
from go2_rma.algo.ppo_rma import PPOWithRMA, PPOConfig

def main():
    device = "cpu"

    # Create dummy env
    env = DummyGo2Env(episode_len=50)
    obs_dim = env.obs_dim
    act_dim = env.act_dim

    cfg = PPOConfig(
        obs_dim=obs_dim,
        act_dim=act_dim,
        device=device,
        batch_size=1024,
        mini_batch_size=256,
        context_len=20,
        z_dim=8,
    )

    algo = PPOWithRMA(cfg)

    ctx_buf = ContextBuffer(obs_dim, act_dim, cfg.context_len)

    total_steps = cfg.batch_size
    obs = env.reset()
    ctx_buf.reset(obs)

    for step in range(total_steps):
        # Build context array [T, obs+act+1]
        ctx_array = ctx_buf.as_array()[None, :, :]  # [1, T, ctx_dim]
        ctx_t = torch.tensor(ctx_array, dtype=torch.float32, device=device)
        with torch.no_grad():
            z = algo.encoder(ctx_t)          # [1, z_dim]
            obs_t = torch.tensor(obs[None, :], dtype=torch.float32, device=device)
            act_t, logp_t, val_t = algo.policy.get_action_and_value(obs_t, z)

        action = act_t.cpu().numpy()[0]
        logp = logp_t.cpu().numpy()[0]
        val = val_t.cpu().numpy()[0]

        next_obs, reward, done, info = env.step(action)

        # store transition
        algo.store_transition(obs, action, reward, done, logp, val, ctx_array[0])

        # update context
        ctx_buf.append(obs, action, reward)

        obs = next_obs
        if done:
            obs = env.reset()
            ctx_buf.reset(obs)

    # Now update PPO
    stats = algo.update()
    print("PPO update stats:", stats)

if __name__ == "__main__":
    main()
