# go2_rma/utils/normalization.py

import numpy as np
import torch

class RunningMeanStd:
    """
    Tracks running mean and variance of observations for normalization.
    """

    def __init__(self, shape, epsilon=1e-4):
        self.mean = np.zeros(shape, dtype=np.float64)
        self.var = np.ones(shape, dtype=np.float64)
        self.count = epsilon

    def update(self, x: np.ndarray):
        x = np.array(x, dtype=np.float64)
        batch_mean = x.mean(axis=0)
        batch_var = x.var(axis=0)
        batch_count = x.shape[0]
        self._update_from_moments(batch_mean, batch_var, batch_count)

    def _update_from_moments(self, batch_mean, batch_var, batch_count):
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * (self.count)
        m_b = batch_var * (batch_count)
        M2 = m_a + m_b + delta**2 * self.count * batch_count / tot_count
        new_var = M2 / tot_count

        self.mean = new_mean
        self.var = new_var
        self.count = tot_count

    def normalize(self, x: np.ndarray, clip=10.0) -> np.ndarray:
        x = (x - self.mean) / (np.sqrt(self.var) + 1e-8)
        return np.clip(x, -clip, clip)

    def torch_normalize(self, x: torch.Tensor, clip=10.0) -> torch.Tensor:
        mean = torch.from_numpy(self.mean).to(x.device, dtype=x.dtype)
        std = torch.sqrt(torch.from_numpy(self.var).to(x.device, dtype=x.dtype) + 1e-8)
        x = (x - mean) / std
        return torch.clamp(x, -clip, clip)
