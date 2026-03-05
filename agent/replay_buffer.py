"""
replay_buffer.py
----------------
Experience replay memory for DQN to decorrelate training transitions.
Handles transitions: (state, action, reward, next_state, done)
"""

from typing import Dict, Tuple

import numpy as np
import torch


class ReplayBuffer:
    """Fixed-size cyclic buffer to store and sample transitions.

    Args:
        capacity: Maximum number of transitions to hold.
        obs_shape: Shape of the observation space (C, H, W).
        device: PyTorch device to move sampled batches to.
    """

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
        """Insert a single transition into the buffer, overwriting oldest if full.

        Args:
            obs: state (uint8 array).
            action: action index.
            reward: scalar reward.
            next_obs: next state (uint8 array).
            done: whether the episode terminated.
        """
        self.obs[self.index] = obs
        self.actions[self.index] = action
        self.rewards[self.index] = reward
        self.next_obs[self.index] = next_obs
        self.dones[self.index] = float(done)

        self.index = (self.index + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int) -> Dict[str, torch.Tensor]:
        """Sample a randomised batch of transitions.

        Args:
            batch_size: Number of transitions to return.

        Returns:
            Dictionary mapping keys to PyTorch tensors on the configured format/device.
            Note: Observations are upcast from uint8 to float32 and normalised to [0,1].
        """
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
