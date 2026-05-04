
from typing import Dict, Tuple

import numpy as np
import torch


class ReplayBuffer:

    def __init__(
        self,
        capacity: int,
        obs_shape: Tuple[int, int, int],
        device: torch.device = torch.device("cpu"),
    ) -> None:
        self.capacity = capacity
        self.device = device

        # Preallocate memory as uint8 (to save massive amounts of RAM for images)
        self.obs = np.empty((capacity, *obs_shape), dtype=np.uint8)
        self.next_obs = np.empty((capacity, *obs_shape), dtype=np.uint8)
        self.actions = np.empty((capacity, 1), dtype=np.int64)
        self.rewards = np.empty((capacity, 1), dtype=np.float32)
        self.dones = np.empty((capacity, 1), dtype=np.float32)

        self.index: int = 0
        self.size: int = 0

    def add(
        self,
        obs: np.ndarray,
        action: int,
        reward: float,
        next_obs: np.ndarray,
        done: bool,
    ) -> None:
        
        self.obs[self.index] = obs
        self.actions[self.index] = action
        self.rewards[self.index] = reward
        self.next_obs[self.index] = next_obs
        self.dones[self.index] = float(done)

        self.index = (self.index + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int) -> Dict[str, torch.Tensor]:

        assert self.size >= batch_size, "Not enough transitions to sample."

        # Randomly pick indices
        indices = np.random.randint(0, self.size, size=batch_size)

        # Slice numpy arrays
        b_obs = self.obs[indices]
        b_actions = self.actions[indices]
        b_rewards = self.rewards[indices]
        b_next_obs = self.next_obs[indices]
        b_dones = self.dones[indices]

        # Convert to PyTorch tensors & move to device
        # Normalise observations from [0, 255] uint8 -> [0.0, 1.0] float32
        t_obs = torch.as_tensor(b_obs, dtype=torch.float32, device=self.device) / 255.0
        t_next = torch.as_tensor(b_next_obs, dtype=torch.float32, device=self.device) / 255.0

        return {
            "obs": t_obs,
            "actions": torch.as_tensor(b_actions, dtype=torch.int64, device=self.device),
            "rewards": torch.as_tensor(b_rewards, dtype=torch.float32, device=self.device),
            "next_obs": t_next,
            "dones": torch.as_tensor(b_dones, dtype=torch.float32, device=self.device),
        }

    def __len__(self) -> int:
        """Return the current number of transitions in the buffer."""
        return self.size
