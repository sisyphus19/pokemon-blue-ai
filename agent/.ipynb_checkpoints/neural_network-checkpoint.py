# neural_network.py
# Modular architecture: Encoder (CNN + latent projection) + Q-head (fully connected)
#
# Structure:
#   Encoder   : raw pixels → normalise (/255) → CNN → flatten → Linear → latent z
#   DQNNetwork: Encoder → Q-head (z → hidden → Q-values)
#   DuelingDQNNetwork: Encoder → value stream + advantage stream

from typing import Tuple
from agent.vit_encoder import ViTEncoder

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _calc_conv_output_dim(input_shape: Tuple[int, int, int], net: nn.Sequential) -> int:
    """Return the flattened output size of a convolutional stack."""
    with torch.no_grad():
        dummy = torch.zeros(1, *input_shape)
        out = net(dummy)
        return int(torch.prod(torch.tensor(out.size())))


# ---------------------------------------------------------------------------
# Encoder
# ---------------------------------------------------------------------------

class Encoder(nn.Module):
    """CNN feature extractor: raw pixel observations → latent vector z.

    Pipeline:
        x (uint8 or float) → /255.0 → CNN → flatten → Linear(latent_dim) → z

    Args:
        input_shape: (C, H, W) of the stacked-frame observation.
        latent_dim:  Size of the output latent vector (default 256).
    """

    def __init__(self, input_shape: Tuple[int, int, int], latent_dim: int = 256):
        super().__init__()
        c_in, _, _ = input_shape

        # Convolutional stack — attribute name kept as `features` for
        # backward-compatible checkpoint keys (encoder.features.*)
        self.features = nn.Sequential(
            nn.Conv2d(c_in, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
        )

        conv_out_size = _calc_conv_output_dim(input_shape, self.features)

        # Linear projection to fixed-size latent space
        self.fc = nn.Sequential(
            nn.Linear(conv_out_size, latent_dim),
            nn.ReLU(),
        )

        self.latent_dim = latent_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return latent vector z of shape [B, latent_dim]."""
        # Normalise pixel values to [0, 1] — handles both uint8 and pre-divided input
        x = x.float() / 255.0
        x = self.features(x)
        x = x.view(x.size(0), -1)   # flatten spatial dims
        return self.fc(x)            # [B, latent_dim]


# ---------------------------------------------------------------------------
# DQN Network
# ---------------------------------------------------------------------------

class DQNNetwork(nn.Module):
    """Standard DQN: Encoder → Q-head → Q-values.

    Q-head architecture: z → Linear(latent_dim) → ReLU → Linear(num_actions)

    Args:
        input_shape: (C, H, W) observation shape.
        num_actions: Number of discrete actions.
        latent_dim:  Latent vector size produced by the Encoder (default 256).
    """

    def __init__(
        self,
        input_shape: Tuple[int, int, int],
        num_actions: int,
        latent_dim: int = 256,
    ):
        super().__init__()

        # --- Encoder ---
        use_vit = True  # toggle
        
        if use_vit:
            self.encoder = ViTEncoder(input_shape, latent_dim)
        else:
            self.encoder = Encoder(input_shape, latent_dim)

        print(">>> USING ENCODER:", type(self.encoder).__name__)

        # --- Q-head: latent → hidden → Q-values ---
        self.q_head = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, num_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute Q-values for all actions.

        Args:
            x: Raw pixel tensor [B, C, H, W].
        Returns:
            Q-values [B, num_actions].
        """
        z = self.encoder(x)          # [B, latent_dim]
        return self.q_head(z)        # [B, num_actions]


# ---------------------------------------------------------------------------
# Dueling DQN Network
# ---------------------------------------------------------------------------

class DuelingDQNNetwork(nn.Module):
    """Dueling DQN: Encoder → separate value & advantage streams.

    Q(s,a) = V(s) + (A(s,a) − mean_a A(s,a))

    Args:
        input_shape: (C, H, W) observation shape.
        num_actions: Number of discrete actions.
        latent_dim:  Latent vector size produced by the Encoder (default 256).
    """

    def __init__(
        self,
        input_shape: Tuple[int, int, int],
        num_actions: int,
        latent_dim: int = 256,
    ):
        super().__init__()

        # --- Encoder ---
        self.encoder = Encoder(input_shape, latent_dim)

        hidden = latent_dim // 2

        # --- Value stream: z → scalar V(s) ---
        self.value_stream = nn.Sequential(
            nn.Linear(latent_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

        # --- Advantage stream: z → A(s, a) for each action ---
        self.advantage_stream = nn.Sequential(
            nn.Linear(latent_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, num_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute Q-values via dueling decomposition.

        Args:
            x: Raw pixel tensor [B, C, H, W].
        Returns:
            Q-values [B, num_actions].
        """
        z = self.encoder(x)                              # [B, latent_dim]

        value = self.value_stream(z)                     # [B, 1]
        advantage = self.advantage_stream(z)             # [B, num_actions]

        # Combine streams (mean-centred advantage for identifiability)
        q_values = value + (advantage - advantage.mean(dim=1, keepdim=True))
        return q_values
