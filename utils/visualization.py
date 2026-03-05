"""
visualization.py
----------------
Matplotlib-based visualisation utilities for analysing training progress.

Generates:
  - Reward curves (episode reward over time)
  - Rolling-average smoothed reward curves
  - Exploration maps (tiles visited per map)
  - Training metrics dashboard (reward, loss, epsilon, episode length)
"""

import json
import csv
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend (safe for headless servers)
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np


def _smooth(values: List[float], window: int = 20) -> np.ndarray:
    """Apply a simple moving-average smoothing.

    Args:
        values: Raw metric values.
        window: Smoothing window size.

    Returns:
        Smoothed numpy array (same length as input).
    """
    if len(values) < window:
        return np.array(values, dtype=float)
    kernel = np.ones(window) / window
    padded = np.pad(values, (window // 2, window - 1 - window // 2), mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def plot_reward_curve(
    rewards: List[float],
    save_path: str,
    title: str = "Episode Reward",
    smooth_window: int = 20,
) -> None:
    """Plot raw and smoothed episode rewards.

    Args:
        rewards: List of total rewards per episode.
        save_path: File path to save the figure.
        title: Plot title.
        smooth_window: Window size for smoothing.
    """
    fig, ax = plt.subplots(figsize=(12, 5))
    episodes = np.arange(1, len(rewards) + 1)

    ax.plot(episodes, rewards, alpha=0.3, color="#5b9bd5", label="Raw Reward")
    smoothed = _smooth(rewards, smooth_window)
    ax.plot(episodes, smoothed, color="#2e75b6", linewidth=2, label=f"Smoothed (w={smooth_window})")

    ax.set_xlabel("Episode", fontsize=12)
    ax.set_ylabel("Total Reward", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_training_dashboard(
    log_csv_path: str,
    save_path: str,
    smooth_window: int = 20,
) -> None:
    """Generate a 4-panel training metrics dashboard from a CSV log.

    Panels: Episode Reward | Episode Length | Epsilon | Loss

    Args:
        log_csv_path: Path to the episode_log.csv file.
        save_path: Path to save the output figure.
        smooth_window: Smoothing window size.
    """
    # Load CSV
    rows = []
    with open(log_csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    if not rows:
        return

    episodes       = [int(r["episode"]) for r in rows]
    rewards        = [float(r["total_reward"]) for r in rows]
    lengths        = [int(r["episode_length"]) for r in rows]
    epsilons       = [float(r["epsilon"]) for r in rows]
    losses         = [float(r["loss"]) if r["loss"] else None for r in rows]
    tiles          = [int(r["tiles_visited"]) for r in rows]

    loss_vals      = [l for l in losses if l is not None]
    loss_episodes  = [episodes[i] for i, l in enumerate(losses) if l is not None]

    fig = plt.figure(figsize=(16, 10))
    fig.suptitle("Pokémon Blue RL Training Dashboard", fontsize=16, fontweight="bold")
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.4, wspace=0.35)

    # Panel 1 – Episode Reward
    ax1 = fig.add_subplot(gs[0, :2])
    ax1.plot(episodes, rewards, alpha=0.25, color="#5b9bd5")
    ax1.plot(episodes, _smooth(rewards, smooth_window), color="#2e75b6", linewidth=2)
    ax1.set_title("Episode Reward")
    ax1.set_xlabel("Episode")
    ax1.set_ylabel("Total Reward")
    ax1.grid(True, alpha=0.3)

    # Panel 2 – Tiles Visited
    ax2 = fig.add_subplot(gs[0, 2])
    ax2.plot(episodes, tiles, color="#70ad47", linewidth=2)
    ax2.set_title("Unique Tiles Visited")
    ax2.set_xlabel("Episode")
    ax2.set_ylabel("Tiles")
    ax2.grid(True, alpha=0.3)

    # Panel 3 – Episode Length
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.plot(episodes, lengths, alpha=0.3, color="#ed7d31")
    ax3.plot(episodes, _smooth(lengths, smooth_window), color="#c55a11", linewidth=2)
    ax3.set_title("Episode Length")
    ax3.set_xlabel("Episode")
    ax3.set_ylabel("Steps")
    ax3.grid(True, alpha=0.3)

    # Panel 4 – Epsilon
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.plot(episodes, epsilons, color="#a020f0", linewidth=2)
    ax4.set_title("Exploration Rate (ε)")
    ax4.set_xlabel("Episode")
    ax4.set_ylabel("Epsilon")
    ax4.set_ylim(0, 1.05)
    ax4.grid(True, alpha=0.3)

    # Panel 5 – Loss
    ax5 = fig.add_subplot(gs[1, 2])
    if loss_vals:
        ax5.plot(loss_episodes, loss_vals, alpha=0.3, color="#ff2052")
        ax5.plot(
            loss_episodes,
            _smooth(loss_vals, min(smooth_window, len(loss_vals))),
            color="#c8102e",
            linewidth=2,
        )
    ax5.set_title("Training Loss")
    ax5.set_xlabel("Episode")
    ax5.set_ylabel("Loss")
    ax5.grid(True, alpha=0.3)

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_exploration_map(
    visited_tiles: List[Tuple[int, int, int]],
    map_id: int,
    save_path: str,
    title: Optional[str] = None,
) -> None:
    """Render a bird's-eye exploration heatmap for a single map.

    Args:
        visited_tiles: List of (map_id, x, y) tuples for all visited tiles.
        map_id: Filter to this particular map ID.
        save_path: Output file path.
        title: Optional figure title.
    """
    tiles = [(x, y) for (m, x, y) in visited_tiles if m == map_id]
    if not tiles:
        return

    xs, ys = zip(*tiles)
    max_x = max(xs) + 1
    max_y = max(ys) + 1

    grid = np.zeros((max_y, max_x), dtype=np.int32)
    for x, y in tiles:
        grid[y, x] += 1

    fig, ax = plt.subplots(figsize=(max(6, max_x // 2), max(6, max_y // 2)))
    im = ax.imshow(grid, cmap="YlOrRd", interpolation="nearest", origin="upper")
    plt.colorbar(im, ax=ax, label="Visit Count")
    ax.set_title(title or f"Exploration Map — Map ID {map_id}", fontweight="bold")
    ax.set_xlabel("Tile X")
    ax.set_ylabel("Tile Y")

    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
