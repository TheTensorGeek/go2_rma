# go2_rma/utils/context_buffer.py

from collections import deque
import numpy as np

class ContextBuffer:
    """
    Maintains a rolling history of (obs, action, reward) for one environment.
    Used to feed the ExtrinsicsEncoder.
    """

    def __init__(self, obs_dim: int, act_dim: int, context_len: int):
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.context_len = context_len
        self.buffer = deque(maxlen=context_len)

    def reset(self, init_obs: np.ndarray):
        """
        Called at episode start. Fills buffer with dummy transitions using init_obs.
        """
        self.buffer.clear()
        dummy_action = np.zeros(self.act_dim, dtype=np.float32)
        dummy_reward = 0.0
        for _ in range(self.context_len):
            self.buffer.append((init_obs.copy(), dummy_action.copy(), dummy_reward))

    def append(self, obs: np.ndarray, action: np.ndarray, reward: float):
        self.buffer.append((obs.copy(), action.copy(), float(reward)))

    def as_array(self) -> np.ndarray:
        """
        Returns: [T, obs_dim + act_dim + 1]
        """
        if len(self.buffer) == 0:
            raise RuntimeError("ContextBuffer is empty. Call reset() first.")
        obs_list, act_list, rew_list = zip(*self.buffer)
        obs_arr = np.stack(obs_list, axis=0)          # [T, obs_dim]
        act_arr = np.stack(act_list, axis=0)          # [T, act_dim]
        rew_arr = np.array(rew_list, dtype=np.float32).reshape(-1, 1)  # [T, 1]
        return np.concatenate([obs_arr, act_arr, rew_arr], axis=-1)
