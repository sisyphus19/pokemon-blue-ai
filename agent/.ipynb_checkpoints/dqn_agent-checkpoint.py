# Implements DQN logic
# Epislon greedy action sel, experience, training loop

import math
import logging
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from agent.neural_network import Encoder, DQNNetwork, DuelingDQNNetwork
from agent.forward_model import ForwardModel
from agent.replay_buffer import ReplayBuffer

logger = logging.getLogger(__name__)


class DQNAgent: # For handling action selection, gradient descent etc

    def __init__(self, config: Dict[str, Any], env_info: Dict[str, Any]):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info("DQNAgent initialising on device: %s", self.device)

        self.num_actions = env_info["num_actions"]
        self.obs_shape = env_info["obs_shape"]

        # Hyperparameters
        cfg = config["dqn"]
        self.batch_size = cfg["batch_size"]
        self.gamma = cfg["gamma"]
        self.epsilon_start = cfg["epsilon_start"]
        self.epsilon_end = cfg["epsilon_end"]
        self.epsilon_decay = cfg["epsilon_decay"]
        self.target_update_freq = cfg["target_update_freq"]
        self.min_replay_size = cfg["min_replay_size"]
        self.train_freq = cfg["train_freq"]
        self.gradient_clip = cfg["gradient_clip"]

        # Neural Networks
        # use_dueling = config["network"].get("dueling_dqn", False)
        # `latent_dim` is the size of the Encoder output vector.
        # Falls back to fc_units for backward config compatibility.
        latent_dim = config["network"].get("latent_dim", config["network"].get("fc_units", 256))
        NetworkClass = DQNNetwork

        self.policy_net = NetworkClass(self.obs_shape, self.num_actions, latent_dim).to(self.device)
        self.target_net = NetworkClass(self.obs_shape, self.num_actions, latent_dim).to(self.device)

        # Clone weights to target at first and freeze target gradients
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        for param in self.target_net.parameters():
            param.requires_grad = False

        # Forward Dynamics Model (optional auxiliary self-supervised module)
        # Set network.forward_model: false in config to disable.
        net_cfg = config["network"]
        use_forward_model = net_cfg.get("forward_model", True)
        self.lambda_ssl: float = net_cfg.get("lambda_ssl", 0.05)

        if use_forward_model and self.lambda_ssl > 0.0:
            self.forward_model: ForwardModel | None = ForwardModel(
                latent_dim=latent_dim,
                num_actions=self.num_actions,
            ).to(self.device)
            logger.info(
                "ForwardModel enabled  (latent_dim=%d, lambda_ssl=%.3f)",
                latent_dim, self.lambda_ssl,
            )
        else:
            self.forward_model = None
            logger.info("ForwardModel disabled.")

        # Optimizer — includes ForwardModel params when active
        optim_params = list(self.policy_net.parameters())
        if self.forward_model is not None:
            optim_params += list(self.forward_model.parameters())
        self.optimizer = optim.Adam(optim_params, lr=cfg["learning_rate"])

        # Loss function (Huber loss for RL)
        self.loss_fn = F.smooth_l1_loss

        # Replay Buffer
        self.replay_buffer = ReplayBuffer(
            capacity=cfg["replay_buffer_size"],
            obs_shape=self.obs_shape,
            device=self.device,
        )

        self.global_step: int = 0
        self.epsilon: float = self.epsilon_start

    def select_action(self, obs: np.ndarray, deterministic: bool = False) -> int: 
        #Select action using epsilon-greedy policy

        # Linear epsilon decay
        if not deterministic:
            fraction = min(1.0, float(self.global_step) / self.epsilon_decay)
            self.epsilon = self.epsilon_start - fraction * (self.epsilon_start - self.epsilon_end)
        else:
            self.epsilon = 0.0

        # Exploration
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.num_actions)

        # Exploitation
        with torch.no_grad():
            self.policy_net.eval()  # Avoid BN/Dropout effects if any
            # Normalisation is handled inside Encoder.forward(); pass raw obs.
            obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
            q_values = self.policy_net(obs_t)
            self.policy_net.train()
            return int(q_values.argmax(dim=1).item())

    def store_transition(
        self,
        obs: np.ndarray,
        action: int,
        reward: float,
        next_obs: np.ndarray,
        done: bool,
    ) -> None:
        self.replay_buffer.add(obs, action, reward, next_obs, done)
        self.global_step += 1

    def train_step(self) -> dict | None:  # Sample one batch, then gradient descent (one iter)
        """One step of combined RL + SSL (forward dynamics) training.

        Returns:
            dict with keys ``rl_loss``, ``ssl_loss``, ``total_loss``
            or ``None`` if the buffer is not ready / not a training step.
        """
        if len(self.replay_buffer) < self.min_replay_size:
            return None

        if self.global_step < 5:
            print("Latent shape:", z.shape)

        if self.global_step % self.train_freq != 0:
            return None

        # ── Sample batch ────────────────────────────────────────────────
        batch = self.replay_buffer.sample(self.batch_size)
        b_obs     = batch["obs"]       # [B, C, H, W]  raw float32
        b_actions = batch["actions"]   # [B, 1]  int64
        b_rewards = batch["rewards"]   # [B, 1]
        b_next_obs = batch["next_obs"] # [B, C, H, W]
        b_dones   = batch["dones"]     # [B, 1]

        # ── Encode current observations (shared between RL + SSL) ────────
        # Running through the encoder once and reusing z avoids a second
        # forward pass through the CNN for the same batch.
        z = self.policy_net.encoder(b_obs)                         # [B, latent_dim]

        # ── RL Loss (DQN Bellman target) ────────────────────────────────
        current_q = self.policy_net.q_head(z).gather(1, b_actions) # [B, 1]

        with torch.no_grad():
            max_next_q = self.target_net(b_next_obs).max(dim=1, keepdim=True)[0]
            # If done, there is no future reward
            target_q = b_rewards + (1.0 - b_dones) * self.gamma * max_next_q

        rl_loss = self.loss_fn(current_q, target_q)

        # ── SSL Loss (Forward Dynamics) ──────────────────────────────────
        ssl_loss_val = 0.0
        total_loss   = rl_loss

        if self.forward_model is not None:
            # Encode next state; detach so the SSL target doesn't back-prop
            # into the encoder through the target path.
            z_next = self.policy_net.encoder(b_next_obs).detach()  # [B, latent_dim]

            # Predict next latent from (z_t, a_t)
            z_pred = self.forward_model(z, b_actions.squeeze(1))   # [B, latent_dim]

            ssl_loss = F.mse_loss(z_pred, z_next)
            total_loss = rl_loss + self.lambda_ssl * ssl_loss
            ssl_loss_val = ssl_loss.item()

        # ── Gradient step ────────────────────────────────────────────────
        self.optimizer.zero_grad()
        total_loss.backward()

        # Clip all optimised parameters (policy_net + forward_model)
        all_params = list(self.policy_net.parameters())
        if self.forward_model is not None:
            all_params += list(self.forward_model.parameters())
        nn.utils.clip_grad_norm_(all_params, self.gradient_clip)

        self.optimizer.step()

        # ── Sync target network ──────────────────────────────────────────
        if self.global_step % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())


        return {
            "rl_loss":    rl_loss.item(),
            "ssl_loss":   ssl_loss_val,
            "total_loss": total_loss.item(),
        }

    def save(self, filepath: str) -> None:
        """Serialise model state dict to disk."""
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "policy_net": self.policy_net.state_dict(),
            "target_net": self.target_net.state_dict(),
            "optimizer":  self.optimizer.state_dict(),
            "global_step": self.global_step,
            "epsilon":     self.epsilon,
        }
        if self.forward_model is not None:
            payload["forward_model"] = self.forward_model.state_dict()
        torch.save(payload, filepath)
        logger.info("Agent saved: %s", filepath)

    def load(self, filepath: str) -> None:
        """Load model state dict from disk."""
        if not Path(filepath).exists():
            raise FileNotFoundError(f"Checkpoint '{filepath}' not found.")

        checkpoint = torch.load(filepath, map_location=self.device)
        self.policy_net.load_state_dict(checkpoint["policy_net"])
        self.target_net.load_state_dict(checkpoint["target_net"])

        if "optimizer" in checkpoint and self.optimizer is not None:
            try:
                self.optimizer.load_state_dict(checkpoint["optimizer"])
            except ValueError:
                print("Optimizer state incompatible, reinitializing optimizer.")

        if self.forward_model is not None and "forward_model" in checkpoint:
            self.forward_model.load_state_dict(checkpoint["forward_model"])

        self.global_step = checkpoint.get("global_step", 0)
        self.epsilon = checkpoint.get("epsilon", self.epsilon_end)

        logger.info("Agent loaded: %s  (Step: %d)", filepath, self.global_step)
