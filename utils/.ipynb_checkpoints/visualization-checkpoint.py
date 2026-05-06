"""
visualization.py
----------------
Matplotlib-based visualisation utilities for analysing training progress.

Generates:
  - Reward curves (episode reward over time)
  - Rolling-average smoothed reward curves
  - Per-map exploration heatmaps (tile visit frequency)
  - Combined exploration heatmap dashboard (all maps in one figure)
  - Training metrics dashboard (reward, loss, epsilon, episode length, tiles)
"""

import csv
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend (safe for headless servers)
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LogNorm
import numpy as np


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _smooth(values: List[float], window: int = 20) -> np.ndarray:
    """Apply a simple moving-average smoothing."""
    if len(values) < window:
        return np.array(values, dtype=float)
    kernel = np.ones(window) / window
    padded = np.pad(values, (window // 2, window - 1 - window // 2), mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def _build_grid(
    tiles_xy: List[Tuple[int, int]],
    counts: List[int],
) -> np.ndarray:
    """Convert (x, y, count) data into a 2-D visit-count grid."""
    if not tiles_xy:
        return np.zeros((1, 1), dtype=np.int32)
    xs = [t[0] for t in tiles_xy]
    ys = [t[1] for t in tiles_xy]
    max_x = max(xs) + 1
    max_y = max(ys) + 1
    grid = np.zeros((max_y, max_x), dtype=np.int32)
    for (x, y), c in zip(tiles_xy, counts):
        grid[y, x] += c
    return grid


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def plot_reward_curve(
    rewards: List[float],
    save_path: str,
    title: str = "Episode Reward",
    smooth_window: int = 20,
) -> None:
    """Plot raw and smoothed episode rewards."""
    fig, ax = plt.subplots(figsize=(12, 5))
    episodes = np.arange(1, len(rewards) + 1)

    ax.plot(episodes, rewards, alpha=0.3, color="#5b9bd5", label="Raw Reward")
    smoothed = _smooth(rewards, smooth_window)
    ax.plot(episodes, smoothed, color="#2e75b6", linewidth=2,
            label=f"Smoothed (w={smooth_window})")

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
    """Generate a 6-panel training metrics dashboard from a CSV log.

    Panels: Reward | Unique Tiles | Episode Length | Epsilon | RL vs SSL Loss | Total Loss
    """
    print(">>> DASHBOARD START <<<")
    rows = []
    with open(log_csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    if not rows:
        return

    print(f"Rows loaded: {len(rows)}")

    episodes      = [int(r["episode"])           for r in rows]
    rewards       = [float(r["total_reward"])     for r in rows]
    lengths       = [int(r["episode_length"])     for r in rows]
    epsilons      = [float(r["epsilon"])          for r in rows]
    tiles         = [int(r["tiles_visited"])      for r in rows]

    # Handle potentially missing loss columns gracefully
    def _get_float(r, key):
        val = r.get(key, "")
        return float(val) if val and val != "" else None

    rl_losses     = [_get_float(r, "rl_loss")    for r in rows]
    ssl_losses    = [_get_float(r, "ssl_loss")   for r in rows]
    total_losses  = [_get_float(r, "total_loss") for r in rows]
    # Fallback to legacy 'loss' if total_loss is missing
    if all(l is None for l in total_losses):
        total_losses = [_get_float(r, "loss") for r in rows]

    fig = plt.figure(figsize=(16, 14))
    fig.patch.set_facecolor("#111827")
    fig.suptitle("Pokémon Blue RL — Training Dashboard",
                 fontsize=18, fontweight="bold", color="white", y=0.98)
    
    # 3x2 Grid
    gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.4, wspace=0.25)

    _DARK = "#1f2937"
    _GRID = "#374151"

    def _style(ax, title):
        ax.set_facecolor(_DARK)
        ax.set_title(title, color="white", fontsize=12, fontweight="bold", pad=10)
        ax.tick_params(colors="white", labelsize=9)
        ax.xaxis.label.set_color("white")
        ax.yaxis.label.set_color("white")
        ax.grid(True, color=_GRID, linewidth=0.5, alpha=0.5)
        for spine in ax.spines.values():
            spine.set_edgecolor(_GRID)

    def _plot_with_smooth(ax, x, y, color, label=None, alpha=0.2):
        valid_indices = [i for i, v in enumerate(y) if v is not None]
        if not valid_indices:
            return
        vx = np.array([x[i] for i in valid_indices])
        vy = np.array([y[i] for i in valid_indices])
        
        ax.plot(vx, vy, alpha=alpha, color=color)
        smoothed = _smooth(vy, min(smooth_window, len(vy)))
        ax.plot(vx, smoothed, color=color, linewidth=2, label=label)

    # Panel 1 – Episode Reward
    ax1 = fig.add_subplot(gs[0, 0])
    _plot_with_smooth(ax1, episodes, rewards, color="#3b82f6", label="Reward")
    ax1.set_xlabel("Episode")
    ax1.set_ylabel("Total Reward")
    _style(ax1, "Episode Reward")

    # Panel 2 – Unique Tiles Visited
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(episodes, tiles, color="#10b981", linewidth=2)
    ax2.fill_between(episodes, tiles, alpha=0.1, color="#10b981")
    ax2.set_xlabel("Episode")
    ax2.set_ylabel("Tiles")
    _style(ax2, "Exploration (Unique Tiles)")

    # Panel 3 – Episode Length
    ax3 = fig.add_subplot(gs[1, 0])
    _plot_with_smooth(ax3, episodes, lengths, color="#f59e0b", label="Steps")
    ax3.set_xlabel("Episode")
    ax3.set_ylabel("Steps")
    _style(ax3, "Episode Length")

    # Panel 4 – Epsilon
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.plot(episodes, epsilons, color="#8b5cf6", linewidth=2)
    ax4.set_xlabel("Episode")
    ax4.set_ylabel("ε")
    ax4.set_ylim(-0.02, 1.05)
    _style(ax4, "Exploration Rate (ε)")

    # Panel 5 – RL vs SSL Loss
    ax5 = fig.add_subplot(gs[2, 0])
    _plot_with_smooth(ax5, episodes, rl_losses, color="#f87171", label="RL (DQN)", alpha=0.15)
    _plot_with_smooth(ax5, episodes, ssl_losses, color="#fbbf24", label="SSL (Forward)", alpha=0.15)
    ax5.set_xlabel("Episode")
    ax5.set_ylabel("Loss")
    ax5.set_yscale("log") # Loss often varies orders of magnitude
    ax5.legend(loc="upper right", fontsize=8, facecolor=_DARK, edgecolor=_GRID, labelcolor="white")
    _style(ax5, "RL vs SSL Loss (Log Scale)")

    # Panel 6 – Total Combined Loss
    ax6 = fig.add_subplot(gs[2, 1])
    _plot_with_smooth(ax6, episodes, total_losses, color="#ec4899", label="Total")
    ax6.set_xlabel("Episode")
    ax6.set_ylabel("Loss")
    ax6.set_yscale("log")
    _style(ax6, "Total Loss (Log Scale)")

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)

    print(">>> SAVING DASHBOARD <<<")
    
    fig.savefig(save_path, dpi=150, facecolor=fig.get_facecolor())
    plt.close(fig)


def plot_exploration_map(
    visited_tiles: List[Tuple[int, int, int]],
    map_id: int,
    save_path: str,
    title: Optional[str] = None,
) -> None:
    """Render a bird's-eye exploration heatmap for a single map (legacy API)."""
    counts: Counter = Counter()
    for m, x, y in visited_tiles:
        if m == map_id:
            counts[(x, y)] += 1

    if not counts:
        return

    tiles_xy = list(counts.keys())
    cnt_list = [counts[k] for k in tiles_xy]
    grid = _build_grid(tiles_xy, cnt_list)

    fig, ax = plt.subplots(
        figsize=(max(6, grid.shape[1] // 2), max(6, grid.shape[0] // 2))
    )
    im = ax.imshow(grid, cmap="YlOrRd", interpolation="nearest", origin="upper")
    plt.colorbar(im, ax=ax, label="Visit Count")
    ax.set_title(title or f"Exploration Heatmap — Map ID {map_id}",
                 fontweight="bold")
    ax.set_xlabel("Tile X")
    ax.set_ylabel("Tile Y")

    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_exploration_heatmaps(
    tile_visit_counts: Counter,
    save_dir: str,
    top_n_maps: int = 9,
) -> None:
    """Generate per-map exploration heatmaps and a combined overview figure.

    Args:
        tile_visit_counts: Counter mapping (map_id, x, y) -> visit count.
        save_dir: Directory to save all output files.
        top_n_maps: How many of the most-visited maps to include in the
                    combined overview grid.
    """
    out = Path(save_dir)
    out.mkdir(parents=True, exist_ok=True)

    if not tile_visit_counts:
        return

    # Group by map_id
    per_map: Dict[int, Dict[Tuple[int, int], int]] = {}
    for (map_id, x, y), count in tile_visit_counts.items():
        per_map.setdefault(map_id, {})[(x, y)] = count

    # Sort maps by total visits (descending) for the overview
    sorted_maps = sorted(
        per_map.items(), key=lambda kv: sum(kv[1].values()), reverse=True
    )

    # ---- Individual per-map files ----------------------------------------
    for map_id, tile_counts in sorted_maps:
        tiles_xy = list(tile_counts.keys())
        cnt_list = [tile_counts[k] for k in tiles_xy]
        grid = _build_grid(tiles_xy, cnt_list)

        h, w = grid.shape
        fig, ax = plt.subplots(figsize=(max(5, w // 2), max(5, h // 2)))
        fig.patch.set_facecolor("#111827")
        ax.set_facecolor("#1f2937")

        vmax = grid.max() if grid.max() > 0 else 1
        im = ax.imshow(
            grid,
            cmap="plasma",
            interpolation="nearest",
            origin="upper",
            norm=LogNorm(vmin=1, vmax=vmax) if vmax > 1 else None,
        )
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label("Visit Count (log scale)", color="white")
        cbar.ax.yaxis.set_tick_params(color="white")
        plt.setp(cbar.ax.yaxis.get_ticklabels(), color="white")

        total = sum(tile_counts.values())
        ax.set_title(
            f"Map ID {map_id}  —  {len(tiles_xy)} unique tiles  |  {total} total visits",
            color="white", fontweight="bold", fontsize=11,
        )
        ax.set_xlabel("Tile X", color="white")
        ax.set_ylabel("Tile Y", color="white")
        ax.tick_params(colors="white")
        for spine in ax.spines.values():
            spine.set_edgecolor("#374151")

        plt.tight_layout()
        fig.savefig(out / f"heatmap_map_{map_id:03d}.png", dpi=150,
                    facecolor=fig.get_facecolor())
        plt.close(fig)

    # ---- Combined overview figure -----------------------------------------
    maps_to_show = sorted_maps[:top_n_maps]
    n = len(maps_to_show)
    if n == 0:
        return

    ncols = min(3, n)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(ncols * 5, nrows * 5),
        squeeze=False,
    )
    fig.patch.set_facecolor("#111827")
    fig.suptitle(
        f"Exploration Heatmaps — Top {n} Maps by Visit Count",
        fontsize=14, fontweight="bold", color="white", y=1.01,
    )

    for idx, (map_id, tile_counts) in enumerate(maps_to_show):
        row, col = divmod(idx, ncols)
        ax = axes[row][col]
        ax.set_facecolor("#1f2937")

        tiles_xy = list(tile_counts.keys())
        cnt_list = [tile_counts[k] for k in tiles_xy]
        grid = _build_grid(tiles_xy, cnt_list)

        vmax = grid.max() if grid.max() > 0 else 1
        im = ax.imshow(
            grid,
            cmap="plasma",
            interpolation="nearest",
            origin="upper",
            norm=LogNorm(vmin=1, vmax=vmax) if vmax > 1 else None,
        )
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        total = sum(tile_counts.values())
        ax.set_title(
            f"Map {map_id}  ({len(tiles_xy)} tiles, {total} visits)",
            color="white", fontsize=9, fontweight="bold",
        )
        ax.tick_params(colors="white", labelsize=7)
        for spine in ax.spines.values():
            spine.set_edgecolor("#374151")

    # Hide unused axes
    for idx in range(n, nrows * ncols):
        row, col = divmod(idx, ncols)
        axes[row][col].set_visible(False)

    plt.tight_layout()
    fig.savefig(out / "heatmap_overview.png", dpi=150,
                facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)

    # ---- Tile visit frequency histogram ----------------------------------
    all_counts = list(tile_visit_counts.values())
    fig, ax = plt.subplots(figsize=(10, 4))
    fig.patch.set_facecolor("#111827")
    ax.set_facecolor("#1f2937")

    ax.hist(all_counts, bins=50, color="#818cf8", edgecolor="#4f46e5", linewidth=0.5)
    ax.set_xlabel("Visit Count per Tile", color="white")
    ax.set_ylabel("Number of Tiles", color="white")
    ax.set_title("Tile Visit Frequency Distribution", color="white",
                 fontweight="bold", fontsize=12)
    ax.tick_params(colors="white")
    ax.grid(True, color="#374151", linewidth=0.6)
    for spine in ax.spines.values():
        spine.set_edgecolor("#374151")

    plt.tight_layout()
    fig.savefig(out / "tile_visit_histogram.png", dpi=150,
                facecolor=fig.get_facecolor())
    plt.close(fig)
