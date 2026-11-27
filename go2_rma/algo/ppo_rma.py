# go2_rma/algo/ppo_rma.py

from dataclasses import dataclass
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from go2_rma.models.rma_policy import RMAPolicy
from go2_rma.models.extrinsics_encoder import ExtrinsicsEncoder
from go2_rma.utils.normalization import RunningMeanStd


@dataclass
class PPOConfig:
    obs_dim: int
    act_dim: int
    z_dim: int = 8
    context_len: int = 40
    gamma: float = 0.99
    lam: float = 0.95
    clip_ratio: float = 0.2
    lr: float = 3e-4
    train_epochs: int = 4
    batch_size: int = 2048
    mini_batch_size: int = 256
    device: str = "cpu"


class PPOWithRMA:
    """
    PPO training loop for:
      - RMAPolicy (actor+critic)
      - ExtrinsicsEncoder
    """

    def __init__(self, cfg: PPOConfig):
        self.cfg = cfg
        self.device = torch.device(cfg.device)

        self.policy = RMAPolicy(cfg.obs_dim, cfg.act_dim, cfg.z_dim).to(self.device)
        self.encoder = ExtrinsicsEncoder(cfg.obs_dim, cfg.act_dim, cfg.context_len, cfg.z_dim).to(self.device)

        self.optimizer = optim.Adam(
            list(self.policy.parameters()) + list(self.encoder.parameters()),
            lr=cfg.lr
        )

        self.obs_rms = RunningMeanStd(shape=(cfg.obs_dim,))

        # Storage for one batch of experience
        self.reset_storage()

    def reset_storage(self):
        self.obs_buf = []
        self.act_buf = []
        self.rew_buf = []
        self.done_buf = []
        self.logp_buf = []
        self.val_buf = []
        self.ctx_buf = []   # context arrays [T, ctx_dim] per step

    def store_transition(self, obs, action, reward, done, logp, value, ctx_array):
        self.obs_buf.append(obs.copy())
        self.act_buf.append(action.copy())
        self.rew_buf.append(float(reward))
        self.done_buf.append(float(done))
        self.logp_buf.append(float(logp))
        self.val_buf.append(float(value))
        self.ctx_buf.append(ctx_array.copy())

    def normalize_obs(self, obs: np.ndarray) -> np.ndarray:
        """Normalize with the current running stats (use this at rollout time)."""
        return self.obs_rms.normalize(obs).astype(np.float32)

    def compute_advantages(self, last_value=0.0):
        """
        GAE-Lambda advantage computation.
        """
        rews = np.array(self.rew_buf, dtype=np.float32)
        vals = np.array(self.val_buf + [last_value], dtype=np.float32)
        dones = np.array(self.done_buf, dtype=np.float32)

        adv = np.zeros_like(rews)
        last_gae = 0.0
        for t in reversed(range(len(rews))):
            nonterminal = 1.0 - dones[t]
            delta = rews[t] + self.cfg.gamma * vals[t+1] * nonterminal - vals[t]
            last_gae = delta + self.cfg.gamma * self.cfg.lam * nonterminal * last_gae
            adv[t] = last_gae

        returns = adv + vals[:-1]
        return adv, returns

    def update(self):
        """
        Perform a PPO update using the collected batch in buffers.
        """
        obs = np.array(self.obs_buf, dtype=np.float32)
        acts = np.array(self.act_buf, dtype=np.float32)
        old_logp = np.array(self.logp_buf, dtype=np.float32)
        ctx = np.array(self.ctx_buf, dtype=np.float32)  # [N, T, ctx_dim]

        # Normalize obs with the SAME stats the rollout used (so old_logp matches),
        # then fold this batch into the running stats for the next rollout.
        obs_norm = self.obs_rms.normalize(obs)
        self.obs_rms.update(obs)

        # Compute advantages
        # For dummy env we use last_value = 0
        adv, rets = self.compute_advantages(last_value=0.0)
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        # Convert to torch
        obs_t = torch.tensor(obs_norm, dtype=torch.float32, device=self.device)
        acts_t = torch.tensor(acts, dtype=torch.float32, device=self.device)
        old_logp_t = torch.tensor(old_logp, dtype=torch.float32, device=self.device)
        adv_t = torch.tensor(adv, dtype=torch.float32, device=self.device)
        rets_t = torch.tensor(rets, dtype=torch.float32, device=self.device)
        ctx_t = torch.tensor(ctx, dtype=torch.float32, device=self.device)

        N = obs_t.shape[0]
        idxs = np.arange(N)

        for epoch in range(self.cfg.train_epochs):
            np.random.shuffle(idxs)
            for start in range(0, N, self.cfg.mini_batch_size):
                end = start + self.cfg.mini_batch_size
                mb_idx = idxs[start:end]

                mb_obs = obs_t[mb_idx]
                mb_acts = acts_t[mb_idx]
                mb_old_logp = old_logp_t[mb_idx]
                mb_adv = adv_t[mb_idx]
                mb_rets = rets_t[mb_idx]
                mb_ctx = ctx_t[mb_idx]

                # Encode context -> z
                z = self.encoder(mb_ctx)

                # Policy evaluation
                new_logp, entropy, value = self.policy.evaluate_actions(mb_obs, z, mb_acts)

                ratio = torch.exp(new_logp - mb_old_logp)
                unclipped = ratio * mb_adv
                clipped = torch.clamp(ratio, 1.0 - self.cfg.clip_ratio, 1.0 + self.cfg.clip_ratio) * mb_adv
                policy_loss = -torch.mean(torch.min(unclipped, clipped))

                value_loss = torch.mean((mb_rets - value)**2)
                entropy_loss = -torch.mean(entropy)

                loss = policy_loss + 0.5 * value_loss + 0.01 * entropy_loss

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 1.0)
                torch.nn.utils.clip_grad_norm_(self.encoder.parameters(), 1.0)
                self.optimizer.step()

        # Clear buffers for next batch
        self.reset_storage()

        return {
            "policy_loss": float(policy_loss.detach().cpu().item()),
            "value_loss": float(value_loss.detach().cpu().item()),
            "entropy": float(entropy.mean().detach().cpu().item()),
        }
