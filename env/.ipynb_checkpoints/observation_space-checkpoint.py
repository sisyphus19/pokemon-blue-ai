
# extract raw screen from game, downsample to grayscale (84x84), frame stacking
from collections import deque
from typing import Deque, Tuple

import cv2
import numpy as np
from gymnasium import spaces


# Default observation dimensions (DeepMind / Atari convention)
OBS_HEIGHT: int = 84
OBS_WIDTH: int = 84
OBS_CHANNELS: int = 1  # Grayscale


class FrameProcessor:

    def __init__(
        self,
        height: int = OBS_HEIGHT,
        width: int = OBS_WIDTH,
        grayscale: bool = True,
    ) -> None:
        self.height = height
        self.width = width
        self.grayscale = grayscale

    def process(self, frame: np.ndarray) -> np.ndarray:

        if frame is None:
            # Return black frame if emulator hasn't produced output yet
            if self.grayscale:
                return np.zeros((1, self.height, self.width), dtype=np.uint8)
            return np.zeros((3, self.height, self.width), dtype=np.uint8)

        # Drop alpha channel if present (RGBA → RGB)
        if frame.ndim == 3 and frame.shape[2] == 4:
            frame = frame[:, :, :3]

        if self.grayscale:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            frame = cv2.resize(frame, (self.width, self.height), interpolation=cv2.INTER_AREA)
            frame = frame[np.newaxis, :, :]  # (1, H, W)
        else:
            frame = cv2.resize(frame, (self.width, self.height), interpolation=cv2.INTER_AREA)
            frame = np.transpose(frame, (2, 0, 1))  # (C, H, W)

        return frame.astype(np.uint8)


class FrameStack:

    def __init__(
        self,
        num_frames: int = 4,
        height: int = OBS_HEIGHT,
        width: int = OBS_WIDTH,
        grayscale: bool = True,
    ) -> None:
        self.num_frames = num_frames
        self.height = height
        self.width = width
        self.channels = 1 if grayscale else 3
        self._frames: Deque[np.ndarray] = deque(maxlen=num_frames)

    def reset(self, initial_frame: np.ndarray) -> np.ndarray:

        for _ in range(self.num_frames):
            self._frames.append(initial_frame)
        return self._get_observation()

    def push(self, frame: np.ndarray) -> np.ndarray:

        self._frames.append(frame)
        return self._get_observation()

    def _get_observation(self) -> np.ndarray:
        """Concatenate stacked frames along the channel axis."""
        return np.concatenate(list(self._frames), axis=0)  # (C*N, H, W)

    @property
    def observation_shape(self) -> Tuple[int, int, int]:
        """Shape of the stacked observation tensor: (C*num_frames, H, W)."""
        return (self.channels * self.num_frames, self.height, self.width)


def build_observation_space(
    num_frames: int = 4,
    height: int = OBS_HEIGHT,
    width: int = OBS_WIDTH,
    grayscale: bool = True,
) -> spaces.Box:

    channels = 1 if grayscale else 3
    shape = (channels * num_frames, height, width)
    return spaces.Box(low=0, high=255, shape=shape, dtype=np.uint8)
