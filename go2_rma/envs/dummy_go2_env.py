# go2_rma/envs/dummy_go2_env.py

import numpy as np

class DummyGo2Env:
    """
    Simple toy environment with:
      - obs_dim = 48 (e.g., joint pos/vel, base state, etc.)
      - act_dim = 12 (e.g., 12 joints)

    Dynamics are fake; it's only for testing the RMA+PPO code locally.
    """

    def __init__(self, episode_len: int = 200):
        self.obs_dim = 48
        self.act_dim = 12
        self.episode_len = episode_len
        self.step_count = 0
        self.state = None

    def reset(self):
        self.step_count = 0
        self.state = np.zeros(self.obs_dim, dtype=np.float32)
        return self._get_obs()

    def _get_obs(self):
        # Here we just return the state; could add some noise
        return self.state.copy()

    def step(self, action: np.ndarray):
        """
        Very simple fake dynamics:
         - next_state = state + 0.01 * action_padded
         - reward = - ||action||^2 (penalize large actions)
        """
        self.step_count += 1

        # pad/trim action
        a = np.zeros(self.obs_dim, dtype=np.float32)
        a[:min(self.act_dim, self.obs_dim)] = action[:min(self.act_dim, self.obs_dim)]
        self.state = self.state + 0.01 * a

        reward = -float(np.linalg.norm(action)**2)
        done = self.step_count >= self.episode_len
        info = {}

        return self._get_obs(), reward, done, info
