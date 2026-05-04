
# initialize env and loag config, then run exp and gradient descent storage
import argparse
import logging
import time
from pathlib import Path

import numpy as np
import yaml
from colorama import Fore, Style, init

from agent import DQNAgent
from env import PokemonBlueEnv
from utils import EpisodeLogger, TensorBoardLogger, setup_logging

init(autoreset=True)
logger = logging.getLogger(__name__)


def load_config(config_path: str) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def train(config: dict) -> None:
    train_cfg = config["training"]
    log_dir = Path(train_cfg["log_dir"]) / train_cfg["experiment_name"]
    model_dir = Path(train_cfg["checkpoint_dir"]) / train_cfg["experiment_name"]
    
    log_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    setup_logging(str(log_dir))
    csv_logger = EpisodeLogger(str(log_dir))
    tb_logger = TensorBoardLogger(str(log_dir))

    logger.info("Starting training run: %s", train_cfg["experiment_name"])

    # Init Environment
    env = PokemonBlueEnv(
        rom_path=config["env"]["rom_path"],
        render_mode=config["env"]["render_mode"],
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

    # Init Agent
    agent = DQNAgent(config, env_info)
    
    # Check for existing checkpoint
    latest_checkpoint = model_dir / "latest.pt"
    if latest_checkpoint.exists():
        logger.info("Resuming from %s", latest_checkpoint)
        agent.load(str(latest_checkpoint))

    # Training Loop 
    start_time = time.time()
    total_steps = agent.global_step
    
    for episode in range(1, train_cfg["num_episodes"] + 1):
        obs, info = env.reset()
        done = False
        episode_reward = 0.0
        episode_losses = []

        while not done:
            # 1. Select action
            action = agent.select_action(obs)

            # 2. Step environment
            next_obs, reward, terminated, truncated, step_info = env.step(action)
            done = terminated or truncated

            # 3. Store transition
            agent.store_transition(obs, action, reward, next_obs, done)
            obs = next_obs
            
            episode_reward += reward
            total_steps += 1

            # 4. Train network
            loss = agent.train_step()
            if loss is not None:
                episode_losses.append(loss)

        avg_loss = float(np.mean(episode_losses)) if episode_losses else 0.0
        exp_stats = env.exploration_stats
        
        # Determine console colour based on outcome
        if 'battle_result' in step_info and step_info.get('total_trainers_defeated', 0) > 0:
            colour = Fore.GREEN
        elif terminated:
            colour = Fore.RED
        else:
            colour = Fore.CYAN

        msg = (
            f"{colour}Ep {episode:4d}{Style.RESET_ALL} | "
            f"Steps: {step_info['step']:4d} | "
            f"Reward: {episode_reward:7.2f} | "
            f"Tiles: {exp_stats['tiles_visited']:4d} | "
            f"Eps: {agent.epsilon:.3f} | "
            f"Loss: {avg_loss:.4f}"
        )
        print(msg)

        if episode % train_cfg["log_freq"] == 0:
            csv_logger.log(
                episode=episode,
                total_reward=episode_reward,
                episode_length=step_info["step"],
                epsilon=agent.epsilon,
                loss=avg_loss if episode_losses else None,
                tiles_visited=exp_stats["tiles_visited"],
                maps_visited=exp_stats["maps_visited"],
            )
            

            tb_logger.scalars(
                {
                    "Reward/Episode": episode_reward,
                    "Length/Episode": step_info["step"],
                    "Exploration/Epsilon": agent.epsilon,
                    "Exploration/UniqueTiles": exp_stats["tiles_visited"],
                    "Exploration/UniqueMaps": exp_stats["maps_visited"],
                    "Loss/Average": avg_loss,
                },
                step=total_steps,
            )

        if episode % train_cfg["save_freq"] == 0:
            agent.save(str(model_dir / f"checkpoint_{episode:05d}.pt"))
            agent.save(str(latest_checkpoint))
            
            # Generate visualisations
            from utils.visualization import plot_training_dashboard
            plot_training_dashboard(
                log_csv_path=str(log_dir / "episode_log.csv"),
                save_path=str(log_dir / "dashboard.png")
            )

    # Cleanup
    env.close()
    csv_logger.close()
    tb_logger.close()
    agent.save(str(model_dir / "final.pt"))
    
    duration = (time.time() - start_time) / 3600.0
    logger.info("Training complete in %.2f hours.", duration)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pokémon Blue RL Training")
    parser.add_argument(
        "--config", type=str, default="config/training_config.yaml", help="Path to YAML config"
    )
    args = parser.parse_args()

    try:
        cfg = load_config(args.config)
        train(cfg)
    except KeyboardInterrupt:
        logger.info("Training interrupted by user. Exiting cleanly.")
