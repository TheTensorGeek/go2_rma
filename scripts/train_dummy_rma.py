# scripts/train_dummy_rma.py

import torch

from go2_rma.envs.dummy_go2_env import DummyGo2Env
from go2_rma.utils.context_buffer import ContextBuffer
from go2_rma.algo.ppo_rma import PPOWithRMA, PPOConfig


def main():
    device = "cpu"

    env = DummyGo2Env(episode_len=200)
    obs_dim = env.obs_dim
    act_dim = env.act_dim

    cfg = PPOConfig(
        obs_dim=obs_dim,
        act_dim=act_dim,
        device=device,
        batch_size=2048,
        mini_batch_size=256,
        context_len=40,
        z_dim=8,
    )

    algo = PPOWithRMA(cfg)

    num_iters = 5  # increase later
    for it in range(num_iters):
        obs = env.reset()
        ctx_buf = ContextBuffer(obs_dim, act_dim, cfg.context_len)
        ctx_buf.reset(obs)

        steps_collected = 0
        ep_return = 0.0
        ep_len = 0
        ep_returns = []

        while steps_collected < cfg.batch_size:
            # build context
            ctx_array = ctx_buf.as_array()[None, :, :]
            ctx_t = torch.tensor(ctx_array, dtype=torch.float32, device=device)
            with torch.no_grad():
                z = algo.encoder(ctx_t)
                obs_t = torch.tensor(algo.normalize_obs(obs)[None, :], dtype=torch.float32, device=device)
                act_t, logp_t, val_t = algo.policy.get_action_and_value(obs_t, z)

            action = act_t.cpu().numpy()[0]
            logp = logp_t.cpu().numpy()[0]
            val = val_t.cpu().numpy()[0]

            next_obs, reward, done, info = env.step(action)

            algo.store_transition(obs, action, reward, done, logp, val, ctx_array[0])
            ctx_buf.append(obs, action, reward)

            obs = next_obs
            ep_return += reward
            ep_len += 1
            steps_collected += 1

            if done:
                ep_returns.append(ep_return)
                obs = env.reset()
                ctx_buf.reset(obs)
                ep_return = 0.0
                ep_len = 0

        stats = algo.update()
        avg_return = sum(ep_returns) / max(1, len(ep_returns))
        print(f"[Iter {it}] avg_return={avg_return:.2f} stats={stats}")

if __name__ == "__main__":
    main()
