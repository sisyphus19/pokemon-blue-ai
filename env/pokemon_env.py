#Uses a pre-generated savestate so every episode starts directly


import logging
from typing import Any, Dict, Optional, Tuple

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from env.action_space import (
    ACTION_TO_PYBOY_EVENTS,
    NUM_ACTIONS,
    get_action_name,
)

from env.observation_space import (
    FrameProcessor,
    FrameStack,
    build_observation_space,
)

from env.reward_system import (
    GameState,
    RewardConfig,
    RewardSystem,
)

from utils.emulator_utils import (
    get_screen_array,
    load_emulator,
    read_game_state,
    send_action,
)

logger = logging.getLogger(__name__)


class PokemonBlueEnv(gym.Env):

    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(
        self,
        rom_path: str = "roms/pokemon_blue.gb",
        savestate_path: str = "assets/start.state",
        render_mode: Optional[str] = None,
        frame_skip: int = 4,
        frame_stack: int = 4,
        screen_height: int = 84,
        screen_width: int = 84,
        grayscale: bool = True,
        max_steps: int = 20480,
        reward_cfg: Optional[Dict[str, float]] = None,
    ) -> None:

        super().__init__()

        self.rom_path = rom_path
        self.savestate_path = savestate_path
        self.render_mode = render_mode
        self.frame_skip = frame_skip
        self.max_steps = max_steps

        self._headless = render_mode not in ("human",)

        self.action_space = spaces.Discrete(NUM_ACTIONS)

        self.observation_space = build_observation_space(
            num_frames=frame_stack,
            height=screen_height,
            width=screen_width,
            grayscale=grayscale,
        )

        self._processor = FrameProcessor(screen_height, screen_width, grayscale)

        self._frame_stack = FrameStack(
            frame_stack,
            screen_height,
            screen_width,
            grayscale,
        )

        reward_config = RewardConfig.from_dict(reward_cfg) if reward_cfg else RewardConfig()
        self._reward_system = RewardSystem(reward_config)

        self._pyboy: Optional[Any] = None

        self._step_count = 0
        self._episode = 0
        self._prev_state: Optional[GameState] = None
        self._total_reward = 0.0


    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:

        super().reset(seed=seed)

        # Close emulator if running
        if self._pyboy is not None:
            self._pyboy.stop()
            self._pyboy = None

        # Launch emulator
        self._pyboy = load_emulator(self.rom_path, headless=self._headless)

        with open(self.savestate_path, "rb") as f:
            self._pyboy.load_state(f)

        from pyboy.utils import WindowEvent
        
        # Clear any pending dialogue or event flag
        for _ in range(15):
            self._pyboy.send_input(WindowEvent.PRESS_BUTTON_A)
            self._pyboy.tick()
            self._pyboy.send_input(WindowEvent.RELEASE_BUTTON_A)
            self._pyboy.tick()
        
        # Advance frames so overworld logic resumes
        for _ in range(60):
            self._pyboy.tick()

        # advance a few frames so emulator stabilizes
        for _ in range(10):
            self._pyboy.tick()

        from pyboy.utils import WindowEvent
        
        # Clear any pending dialogue that might block movement
        for _ in range(20):
            self._pyboy.send_input(WindowEvent.PRESS_BUTTON_A)
            self._pyboy.tick()
            self._pyboy.send_input(WindowEvent.RELEASE_BUTTON_A)
            self._pyboy.tick()
        
        # Let the overworld engine stabilise
        for _ in range(60):
            self._pyboy.tick()

        self._step_count = 0
        self._episode += 1
        self._total_reward = 0.0

        self._reward_system.reset()

        raw_frame = get_screen_array(self._pyboy)
        processed = self._processor.process(raw_frame)

        obs = self._frame_stack.reset(processed)

        state_dict = read_game_state(self._pyboy)
        self._prev_state = GameState(**state_dict)

        return obs, self._build_info()

    def step(
        self,
        action: int,
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:

        assert self._pyboy is not None

        press_event, release_event = ACTION_TO_PYBOY_EVENTS[action]

        send_action(
            self._pyboy,
            press_event,
            release_event,
            hold_frames=self.frame_skip,
        )
        for _ in range(8):
            self._pyboy.tick()

        raw_frame = get_screen_array(self._pyboy)
        processed = self._processor.process(raw_frame)

        obs = self._frame_stack.push(processed)

        state_dict = read_game_state(self._pyboy)
        curr_state = GameState(**state_dict)

        reward = self._reward_system.compute(self._prev_state, curr_state)

        self._prev_state = curr_state

        self._total_reward += reward
        self._step_count += 1

        terminated = self._is_terminated(curr_state)
        truncated = self._step_count >= self.max_steps

        info = self._build_info()
        info["action_name"] = get_action_name(action)

        return obs, reward, terminated, truncated, info

    def render(self):

        if self._pyboy is None:
            return None

        if self.render_mode == "rgb_array":
            return get_screen_array(self._pyboy)

        return None

    def close(self):

        if self._pyboy is not None:
            self._pyboy.stop()
            self._pyboy = None

            logger.info(
                "Emulator closed after episode %d.",
                self._episode,
            )

 
    def _is_terminated(self, state: GameState):

        if state.party_hp and all(hp == 0 for hp in state.party_hp):
            return True

        return False

    def _build_info(self):

        exploration = self._reward_system.get_exploration_stats()

        return {
            "step": self._step_count,
            "episode": self._episode,
            "total_reward": self._total_reward,
            **exploration,
        }

    @property
    def exploration_stats(self):

        return self._reward_system.get_exploration_stats()