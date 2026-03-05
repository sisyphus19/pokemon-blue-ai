"""
action_space.py
---------------
Defines the discrete action space for the Pokémon Blue RL agent.

Actions map directly to Game Boy button presses understood by PyBoy.
"""

from enum import IntEnum
from typing import Dict, List

from pyboy.utils import WindowEvent


class GameBoyAction(IntEnum):
    """Enumeration of all supported Game Boy button actions."""
    A = 0
    B = 1
    UP = 2
    DOWN = 3
    LEFT = 4
    RIGHT = 5
    START = 6
    SELECT = 7
    NOOP = 8


# Total number of discrete actions
NUM_ACTIONS: int = len(GameBoyAction)

# Human-readable action labels (useful for logging/visualization)
ACTION_NAMES: List[str] = [action.name for action in GameBoyAction]

# Mapping from action index → (press_event, release_event)
# Each press is followed by a release to simulate a tap.
ACTION_TO_PYBOY_EVENTS: Dict[int, tuple] = {
    GameBoyAction.A:      (WindowEvent.PRESS_BUTTON_A,      WindowEvent.RELEASE_BUTTON_A),
    GameBoyAction.B:      (WindowEvent.PRESS_BUTTON_B,      WindowEvent.RELEASE_BUTTON_B),
    GameBoyAction.UP:     (WindowEvent.PRESS_ARROW_UP,      WindowEvent.RELEASE_ARROW_UP),
    GameBoyAction.DOWN:   (WindowEvent.PRESS_ARROW_DOWN,    WindowEvent.RELEASE_ARROW_DOWN),
    GameBoyAction.LEFT:   (WindowEvent.PRESS_ARROW_LEFT,    WindowEvent.RELEASE_ARROW_LEFT),
    GameBoyAction.RIGHT:  (WindowEvent.PRESS_ARROW_RIGHT,   WindowEvent.RELEASE_ARROW_RIGHT),
    GameBoyAction.START:  (WindowEvent.PRESS_BUTTON_START,  WindowEvent.RELEASE_BUTTON_START),
    GameBoyAction.SELECT: (WindowEvent.PRESS_BUTTON_SELECT, WindowEvent.RELEASE_BUTTON_SELECT),
    GameBoyAction.NOOP:   (None, None),  # No-op: no button pressed
}


def get_action_name(action_idx: int) -> str:
    """Return the human-readable name for a given action index.

    Args:
        action_idx: Integer index of the action.

    Returns:
        String name of the action.
    """
    return ACTION_NAMES[action_idx]
