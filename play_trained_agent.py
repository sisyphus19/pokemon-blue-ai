import argparse
import logging
import time

import yaml
from colorama import Fore, Style, init

from agent import DQNAgent
from env import PokemonBlueEnv

init(autoreset=True)
logger = logging.getLogger(__name__)


def load_config(config_path: str) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def play(config: dict, model_path: str, fps: int = 60) -> None:
    """Run a trained agent with live rendering."""
    # Force human rendering and no frame skip so we can watch smoothly
    env = PokemonBlueEnv(
        rom_path=config["env"]["rom_path"],
        render_mode="human",          # Must be human to launch SDL2 window
        frame_skip=config["env"]["frame_skip"],                 # 1 frame per action for smooth visual playback
        frame_stack=config["env"]["frame_stack"],
        screen_height=config["env"]["screen_height"],
        screen_width=config["env"]["screen_width"],
        grayscale=config["env"]["grayscale"],
        max_steps=200000,             # Run basically forever until user quits
        reward_cfg=config.get("rewards", {}),
    )

    env_info = {
        "obs_shape": env.observation_space.shape,
        "num_actions": env.action_space.n,
    }

    agent = DQNAgent(config, env_info)
    try:
        agent.load(model_path)
    except Exception as e:
        logger.error(f"Failed to load model from {model_path}: {e}")
        env.close()
        return

    # Fully deterministic, greedy policy
    agent.epsilon = 0.05

    print(f"\n{Fore.GREEN}▶ Starting Live Playback{Style.RESET_ALL}")
    print(f"Model ID : {model_path}")
    print(f"Target   : {fps} FPS")
    print("Press Ctrl+C in this terminal to stop.\n")

    frame_time = 1.0 / fps

    try:
        obs, _ = env.reset()
        done = False
        step = 0

        while not done:
            start_t = time.time()

            # Select deterministic action
            action = agent.select_action(obs)
            
            # Step emulator
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            step += 1

            # Throttle playback speed to match requested FPS
            # PyBoy in human mode automatically manages its own clock to an extent,
            # but this ensures we don't wildly over-accelerate if it's uncapped.
            elapsed = time.time() - start_t
            if elapsed < frame_time:
                time.sleep(frame_time - elapsed)
                
            if step % 600 == 0:  # Log roughly every 10 seconds at 60fps
                print(f"Step {step:5d} | Tiles Explored: {info.get('tiles_visited', 0)}")

    except KeyboardInterrupt:
        print(f"\n{Fore.YELLOW}Playback stopped by user.{Style.RESET_ALL}")
    finally:
        env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Watch a Trained Agent Play Pokémon Blue")
    parser.add_argument("--config", type=str, default="config/training_config.yaml")
    parser.add_argument("--model", type=str, required=True, help="Path to checkpoint .pt")
    parser.add_argument("--fps", type=int, default=60, help="Target playback frames per second")
    args = parser.parse_args()

    cfg = load_config(args.config)
    play(cfg, args.model, args.fps)
