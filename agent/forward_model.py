# forward_model.py
# Predicts next latent state given current latent state + action.
#
# Architecture:
#   (z_t, one_hot(a_t)) → Linear → ReLU → Linear → z_(t+1) predicted
#
# NOT connected to the training loop yet — standalone module only.

import torch
import torch.nn as nn
import torch.nn.functional as F


class ForwardModel(nn.Module):
    """Predicts the next latent state in encoder space.

    Given the current latent vector ``z_t`` produced by the Encoder and the
    action ``a_t`` taken at that step, the model learns to predict:

        z_(t+1) ≈ ForwardModel(z_t, a_t)

    The action is embedded as a one-hot vector and concatenated with ``z_t``
    before being passed through two fully-connected layers.

    Args:
        latent_dim:  Dimensionality of the latent vector (must match Encoder
                     output, e.g. 256).
        num_actions: Total number of discrete actions in the environment.
        hidden_dim:  Width of the hidden layer (default: 256).
    """

    def __init__(
        self,
        latent_dim: int,
        num_actions: int,
        hidden_dim: int = 256,
    ) -> None:
        super().__init__()

        self.latent_dim = latent_dim
        self.num_actions = num_actions

        # Input to the MLP: z_t (latent_dim) + one-hot action (num_actions)
        input_dim = latent_dim + num_actions

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),
        )

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def forward(self, z: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Predict next latent state.

        Args:
            z:      Current latent vector, shape ``[B, latent_dim]``.
            action: Integer action indices, shape ``[B]`` or ``[B, 1]``.

        Returns:
            Predicted next latent vector ``z_pred``, shape ``[B, latent_dim]``.
        """
        # --- Normalise action shape to [B] ---
        action = action.view(-1).long()

        # --- One-hot encode action → [B, num_actions] ---
        action_onehot = F.one_hot(action, num_classes=self.num_actions).float()

        # --- Concatenate latent + action → [B, latent_dim + num_actions] ---
        x = torch.cat([z, action_onehot], dim=-1)

        # --- Predict z_(t+1) ---
        return self.net(x)   # [B, latent_dim]
