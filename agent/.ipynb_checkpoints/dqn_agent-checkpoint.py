"""
dqn_agent.py
------------
Implementation of the Deep Q-Network logic.

Capabilities:
  - Epsilon-greedy action selection
  - Experience accumulation
  - PyTorch training loop across mini-batches
  - Hard target network updates
  - Model checkpointing
"""

import math
import logging
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from agent.neural_network import DQNNetwork, DuelingDQNNetwork
from agent.replay_buffer import ReplayBuffer

logger = logging.getLogger(__name__)


class DQNAgent:
    """The Deep Q-Network Agent.

    Handles action selection, buffer management, gradient descent,
    and target network synchronisation.
    """

    def __init__(self, config: Dict[str, Any], env_info: Dict[str, Any]):
        """Initialisation.

        Args:
            config: Agent configuration (typically loaded from YAML).
            env_info: Contains at least "obs_shape" and "num_actions".
        """
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
        use_dueling = config["network"].get("dueling_dqn", False)
        fc_units = config["network"].get("fc_units", 512)
        NetworkClass = DuelingDQNNetwork if use_dueling else DQNNetwork

        self.policy_net = NetworkClass(self.obs_shape, self.num_actions, fc_units).to(self.device)
        self.target_net = NetworkClass(self.obs_shape, self.num_actions, fc_units).to(self.device)

        # Clone weights to target initially and freeze target gradients
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        for param in self.target_net.parameters():
            param.requires_grad = False

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=cfg["learning_rate"])
        
        # Loss function (Huber loss is standard for DQN to combat outliers)
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
        """Select an action using an epsilon-greedy policy.

        Args:
            obs: Observation numpy array (uint8).
            deterministic: If True, bypass randomness (epsilon=0).

        Returns:
            Chosen action index.
        """
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
            # Prepare tensor: [H,W,C] (or stacked C,H,W) -> [1, C, H, W], normalise -> 0-1
            obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0) / 255.0
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
        """Deposit a transition into the replay memory and bump global step."""
        self.replay_buffer.add(obs, action, reward, next_obs, done)
        self.global_step += 1

    def train_step(self) -> float | None:
        """Sample a batch and perform one gradient descent iteration.

        Returns:
            Scalar loss value, or None if the buffer is too small or it isn't time.
        """
        if len(self.replay_buffer) < self.min_replay_size:
            return None

        if self.global_step % self.train_freq != 0:
            return None

        # Sample batch
        batch = self.replay_buffer.sample(self.batch_size)
        
        b_obs = batch["obs"]
        b_actions = batch["actions"]
        b_rewards = batch["rewards"]
        b_next_obs = batch["next_obs"]
        b_dones = batch["dones"]

        # Compute current Q_policy(s, a)
        # gather() extracts the Q-value for the action actually taken
        current_q = self.policy_net(b_obs).gather(1, b_actions)

        # Compute max Q_target(s', a)
        with torch.no_grad():
            max_next_q = self.target_net(b_next_obs).max(dim=1, keepdim=True)[0]
            # If done, there is no future reward
            target_q = b_rewards + (1.0 - b_dones) * self.gamma * max_next_q

        # Calculate loss
        loss = self.loss_fn(current_q, target_q)

        # Gradient step
        self.optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping prevents exploding gradients during reward spikes
        nn.utils.clip_grad_norm_(self.policy_net.parameters(), self.gradient_clip)
        self.optimizer.step()

        # Update target network periodically
        if self.global_step % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

        return loss.item()

    def save(self, filepath: str) -> None:
        """Serialise model state dict to disk."""
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "policy_net": self.policy_net.state_dict(),
            "target_net": self.target_net.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "global_step": self.global_step,
            "epsilon": self.epsilon,
        }, filepath)
        logger.info("Agent saved: %s", filepath)

    def load(self, filepath: str) -> None:
        """Load model state dict from disk."""
        if not Path(filepath).exists():
            raise FileNotFoundError(f"Checkpoint '{filepath}' not found.")
            
        checkpoint = torch.load(filepath, map_location=self.device)
        self.policy_net.load_state_dict(checkpoint["policy_net"])
        self.target_net.load_state_dict(checkpoint["target_net"])
        
        if "optimizer" in checkpoint and self.optimizer is not None:
            self.optimizer.load_state_dict(checkpoint["optimizer"])
        
        self.global_step = checkpoint.get("global_step", 0)
        self.epsilon = checkpoint.get("epsilon", self.epsilon_end)
        
        logger.info("Agent loaded: %s  (Step: %d)", filepath, self.global_step)
