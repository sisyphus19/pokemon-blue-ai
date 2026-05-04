# Actions map directly to Game Boy button presses understood by PyBoy.


from enum import IntEnum
from typing import Dict, List

from pyboy.utils import WindowEvent


class GameBoyAction(IntEnum):
    A = 0
    B = 1
    UP = 2
    DOWN = 3
    LEFT = 4
    RIGHT = 5



# Total number of discrete actions
NUM_ACTIONS: int = len(GameBoyAction)

# Human-readable action labels 
ACTION_NAMES: List[str] = [action.name for action in GameBoyAction]

ACTION_TO_PYBOY_EVENTS: Dict[int, tuple] = {
    GameBoyAction.A:      (WindowEvent.PRESS_BUTTON_A,      WindowEvent.RELEASE_BUTTON_A),
    GameBoyAction.B:      (WindowEvent.PRESS_BUTTON_B,      WindowEvent.RELEASE_BUTTON_B),
    GameBoyAction.UP:     (WindowEvent.PRESS_ARROW_UP,      WindowEvent.RELEASE_ARROW_UP),
    GameBoyAction.DOWN:   (WindowEvent.PRESS_ARROW_DOWN,    WindowEvent.RELEASE_ARROW_DOWN),
    GameBoyAction.LEFT:   (WindowEvent.PRESS_ARROW_LEFT,    WindowEvent.RELEASE_ARROW_LEFT),
    GameBoyAction.RIGHT:  (WindowEvent.PRESS_ARROW_RIGHT,   WindowEvent.RELEASE_ARROW_RIGHT),
}


def get_action_name(action_idx: int) -> str:
    #Return the name for a given action index.

    return ACTION_NAMES[action_idx]
