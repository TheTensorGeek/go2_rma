# go2_rma/envs/isaac_go2_env.py

"""
Skeleton for IsaacLab-based Go2 environment.

You will fill this later when you are on the GPU machine with IsaacLab installed.
For now, this file is just a placeholder so imports work.
"""

import numpy as np

class IsaacGo2Env:
    def __init__(self, urdf_path: str, episode_len: int = 200, device: str = "cuda:0"):
        """
        urdf_path: path to go2_description.urdf from the Unitree repo.
        """
        self.urdf_path = urdf_path
        self.episode_len = episode_len
        self.device = device

        # Once IsaacLab is installed, here you will:
        #  - launch Isaac
        #  - load the Go2 robot from URDF/USD
        #  - set up terrain, sensors, etc.

        # For now, just set fake dims so code can import:
        self.obs_dim = 48
        self.act_dim = 12

    def reset(self):
        """
        Later: reset Isaac simulation and robot pose, return initial observation.
        """
        raise NotImplementedError("Fill this when integrating with IsaacLab.")

    def step(self, action: np.ndarray):
        """
        Later: apply action, step Isaac simulation, compute reward and done.
        """
        raise NotImplementedError("Fill this when integrating with IsaacLab.")
