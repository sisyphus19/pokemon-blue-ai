
# evaluate agent and save performance

import argparse
import logging
from pathlib import Path

import numpy as np
import yaml
from colorama import Fore, Style, init

from agent import DQNAgent
from env import PokemonBlueEnv
from utils.visualization import plot_exploration_map

init(autoreset=True)
logger = logging.getLogger(__name__)


def load_config(config_path: str) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def evaluate(config: dict, model_path: str, num_episodes: int) -> None:
    env = PokemonBlueEnv(
        rom_path=config["env"]["rom_path"],
        render_mode="headless",
        frame_skip=config["env"]["frame_skip"],
        frame_stack=config["env"]["frame_stack"],
        screen_height=config["env"]["screen_height"],
        screen_width=config["env"]["screen_width"],
        grayscale=config["env"]["grayscale"],
        max_steps=config["env"]["max_steps_per_episode"],
        reward_cfg=config.get("rewards", {}),
    )

    env_info = {
        "obs_shape": env.observation_space.shape,
        "num_actions": env.action_space.n,
    }

    agent = DQNAgent(config, env_info)
    agent.load(model_path)

    # Force fully deterministic evaluation
    agent.epsilon = 0.0

    print(f"\n{Fore.CYAN}--- Starting Evaluation ---{Style.RESET_ALL}")
    print(f"Model: {model_path}")
    print(f"Episodes: {num_episodes}\n")

    rewards = []
    lengths = []
    tiles   = []
    
    # Store all visited tiles for the final heatmap
    all_visited_tiles = set()

    for episode in range(1, num_episodes + 1):
        obs, info = env.reset()
        done = False
        episode_reward = 0.0
        
        # Override the env's tracking because it accumulates across episodes
        # and we want clean per-episode metrics for evaluation.
        env._reward_system._visited_tiles.clear()
        env._reward_system._visited_maps.clear()

        while not done:
            action = agent.select_action(obs, deterministic=True)
            obs, reward, terminated, truncated, step_info = env.step(action)
            done = terminated or truncated
            episode_reward += reward

            # Hack to read internal state to get the exact (map, x, y)
            st = env._prev_state
            if st:
                all_visited_tiles.add((st.map_id, st.player_x, st.player_y))

        rewards.append(episode_reward)
        lengths.append(step_info["step"])
        
        # Get exploration for *this* episode
        exp_stats = env.exploration_stats
        tiles.append(exp_stats["tiles_visited"])

        print(
            f"Eval Ep {episode:2d} | "
            f"Reward: {episode_reward:7.2f} | "
            f"Steps: {step_info['step']:4d} | "
            f"Tiles: {exp_stats['tiles_visited']:4d}"
        )

    env.close()

    print(f"\n{Fore.GREEN}--- Evaluation Results ---{Style.RESET_ALL}")
    print(f"Average Reward: {np.mean(rewards):.2f} ± {np.std(rewards):.2f}")
    print(f"Average Length: {np.mean(lengths):.1f} steps")
    print(f"Average Tiles:  {np.mean(tiles):.1f} unique tiles")
    print(f"Max Reward:     {np.max(rewards):.2f}")
    
    # Render an exploration map for Map ID 0 (usually Pallet Town/Route 1)
    map_path = "experiments/eval_map_0.png"
    plot_exploration_map(list(all_visited_tiles), map_id=0, save_path=map_path)
    print(f"Exploration map saved to {map_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Trained Pokémon RL Agent")
    parser.add_argument("--config", type=str, default="config/training_config.yaml")
    parser.add_argument("--model", type=str, required=True, help="Path to checkpoint .pt")
    parser.add_argument("--episodes", type=int, default=10, help="Number of eval episodes")
    args = parser.parse_args()

    cfg = load_config(args.config)
    evaluate(cfg, args.model, args.episodes)
