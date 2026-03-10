"""
neural_network.py
-----------------
PyTorch neural network architectures for Q-learning.

Provides:
  - Base CNN module for processing image observations
  - Standard DQN network head
  - Dueling DQN network head (optional improvement)
"""

from typing import Tuple

import torch
import torch.nn as nn


def _calc_conv_output_dim(input_shape: Tuple[int, int, int], net: nn.Sequential) -> int:
    """Helper to calculate the flattened size of a convolutional network's output.

    Args:
        input_shape: (C, H, W) of the input tensor.
        net: Convolutional nn.Sequential block.

    Returns:
        Flattened dimension size (C * H * W).
    """
    with torch.no_grad():
        dummy = torch.zeros(1, *input_shape)
        out = net(dummy)
        return int(torch.prod(torch.tensor(out.size())))


class NatureCNN(nn.Module):
    """Convolutional encoder matching the original DeepMind 'Nature' paper.

    Expects input shape matching the frame-stacked observation (e.g., 4x84x84).
    """

    def __init__(self, input_shape: Tuple[int, int, int], output_dim: int = 512):
        super().__init__()
        c_in, _, _ = input_shape

        self.features = nn.Sequential(
            nn.Conv2d(c_in, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
        )

        conv_out_size = _calc_conv_output_dim(input_shape, self.features)

        self.fc = nn.Sequential(
            nn.Linear(conv_out_size, output_dim),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Normalised observation tensor [B, C, H, W] in range [0, 1].

        Returns:
            Feature embeddings [B, output_dim].
        """
        x = self.features(x)
        x = x.view(x.size(0), -1)  # Flatten
        return self.fc(x)


class DQNNetwork(nn.Module):
    """Standard DQN architecture: CNN encoder + Linear action-value head."""

    def __init__(
        self,
        input_shape: Tuple[int, int, int],
        num_actions: int,
        fc_units: int = 512,
    ):
        super().__init__()
        self.encoder = NatureCNN(input_shape, fc_units)
        self.q_head = nn.Linear(fc_units, num_actions)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute Q-values for all actions.

        Args:
            x: Normalised image tensor [B, C, H, W] in range [0, 1].

        Returns:
            Action values [B, num_actions].
        """
        features = self.encoder(x)
        return self.q_head(features)


class DuelingDQNNetwork(nn.Module):
    """Dueling DQN architecture.

    Separates value estimation V(s) from advantage estimation A(s, a)
    to improve learning stability across many similar-value actions.
    Q(s, a) = V(s) + A(s, a) - mean(A(s, a'))
    """

    def __init__(
        self,
        input_shape: Tuple[int, int, int],
        num_actions: int,
        fc_units: int = 512,
    ):
        super().__init__()
        # Encoder output is smaller because we split into two streams
        self.encoder = NatureCNN(input_shape, fc_units)

        self.value_stream = nn.Sequential(
            nn.Linear(fc_units, fc_units // 2),
            nn.ReLU(),
            nn.Linear(fc_units // 2, 1),
        )

        self.advantage_stream = nn.Sequential(
            nn.Linear(fc_units, fc_units // 2),
            nn.ReLU(),
            nn.Linear(fc_units // 2, num_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute Q-values via dueling aggregation.

        Args:
            x: Normalised image tensor [B, C, H, W].

        Returns:
            Action values [B, num_actions].
        """
        features = self.encoder(x)

        value = self.value_stream(features)          # [B, 1]
        advantage = self.advantage_stream(features)  # [B, num_actions]

        # Combine streams
        q_values = value + (advantage - advantage.mean(dim=1, keepdim=True))
        return q_values
