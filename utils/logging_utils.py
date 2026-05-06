"""
logging_utils.py
----------------
Structured logging utilities for the Pokémon Blue RL training pipeline.

Provides:
  - CSV-based episode log writer
  - TensorBoard SummaryWriter wrapper
  - Convenience helpers for console output
"""

import csv
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


def setup_logging(log_dir: str, name: str = "train", level: int = logging.INFO) -> None:
    """Configure root logger to write to both console and a file.

    Args:
        log_dir: Directory where the log file will be saved.
        name: Base name for the log file (e.g. "train" → "train.log").
        level: Python logging level.
    """
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    log_path = Path(log_dir) / f"{name}.log"

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # File handler
    file_handler = logging.FileHandler(log_path, mode="a", encoding="utf-8")
    file_handler.setFormatter(formatter)

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    root = logging.getLogger()
    root.setLevel(level)
    root.addHandler(file_handler)
    root.addHandler(console_handler)

    logger.info("Logging initialised → %s", log_path)


class EpisodeLogger:
    """Writes per-episode metrics to a CSV file for later analysis.

    Args:
        log_dir: Directory to store the CSV file.
        filename: Name of the CSV file.
    """

    FIELDNAMES = [
        "episode",
        "total_reward",
        "episode_length",
        "epsilon",
        "rl_loss",
        "ssl_loss",
        "total_loss",
        "loss",  # kept for backward compatibility (same as total_loss)
        "tiles_visited",
        "maps_visited",
        "timestamp",
    ]

    def __init__(self, log_dir: str, filename: str = "episode_log.csv") -> None:
        Path(log_dir).mkdir(parents=True, exist_ok=True)
        self._path = Path(log_dir) / filename
        self._file = open(self._path, "w", newline="", encoding="utf-8")
        self._writer = csv.DictWriter(self._file, fieldnames=self.FIELDNAMES)
        self._writer.writeheader()
        self._file.flush()

    def log(
        self,
        episode: int,
        total_reward: float,
        episode_length: int,
        epsilon: float,
        rl_loss: Optional[float] = None,
        ssl_loss: Optional[float] = None,
        total_loss: Optional[float] = None,
        loss: Optional[float] = None,
        tiles_visited: int = 0,
        maps_visited: int = 0,
    ) -> None:
        """Write one row of episode metrics.

        Args:
            episode: Episode number.
            total_reward: Cumulative reward for the episode.
            episode_length: Number of steps in the episode.
            epsilon: Current exploration rate.
            rl_loss: Average reinforcement learning loss.
            ssl_loss: Average self-supervised forward model loss.
            total_loss: Average total combined loss.
            loss: Average training loss (legacy field, mapped to total_loss).
            tiles_visited: Cumulative unique tiles visited.
            maps_visited: Cumulative unique maps visited.
        """
        # Ensure 'loss' field matches 'total_loss' if not explicitly provided
        final_total = total_loss if total_loss is not None else loss
        final_legacy = loss if loss is not None else total_loss

        row = {
            "episode": episode,
            "total_reward": round(total_reward, 4),
            "episode_length": episode_length,
            "epsilon": round(epsilon, 6),
            "rl_loss": round(rl_loss, 6) if rl_loss is not None else "",
            "ssl_loss": round(ssl_loss, 6) if ssl_loss is not None else "",
            "total_loss": round(final_total, 6) if final_total is not None else "",
            "loss": round(final_legacy, 6) if final_legacy is not None else "",
            "tiles_visited": tiles_visited,
            "maps_visited": maps_visited,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
        self._writer.writerow(row)
        self._file.flush()

    def close(self) -> None:
        """Close the underlying CSV file."""
        self._file.close()
        logger.info("Episode log saved to: %s", self._path)


class TensorBoardLogger:
    """Thin wrapper around TensorBoard SummaryWriter.

    Gracefully degrades if TensorBoard is unavailable.

    Args:
        log_dir: TensorBoard log directory.
        comment: Optional suffix appended to the run directory.
    """

    def __init__(self, log_dir: str, comment: str = "") -> None:
        self._writer = None
        try:
            from torch.utils.tensorboard import SummaryWriter
            self._writer = SummaryWriter(log_dir=log_dir, comment=comment)
            logger.info("TensorBoard writer initialised at: %s", log_dir)
        except ImportError:
            logger.warning(
                "TensorBoard not available. Install tensorboard for live dashboards."
            )

    def scalar(self, tag: str, value: float, step: int) -> None:
        """Log a scalar value.

        Args:
            tag: Metric name.
            value: Scalar value.
            step: Global step counter.
        """
        if self._writer:
            self._writer.add_scalar(tag, value, step)

    def scalars(self, metrics: Dict[str, float], step: int) -> None:
        """Log multiple scalars at once.

        Args:
            metrics: Dictionary of tag → value.
            step: Global step counter.
        """
        for tag, value in metrics.items():
            self.scalar(tag, value, step)

    def close(self) -> None:
        """Flush and close the TensorBoard writer."""
        if self._writer:
            self._writer.close()
