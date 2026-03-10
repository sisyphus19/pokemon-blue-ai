"""
emulator_utils.py
-----------------
Low-level utilities for interfacing with PyBoy to run Pokémon Blue.

Covers:
  - Emulator initialisation / teardown
  - Screen capture
  - Memory address definitions for relevant game variables
  - Reading game state from memory
"""

import logging
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Pokémon Blue Game Boy Memory Map (selected addresses)
# Reference: https://datacrystal.romhacking.net/wiki/Pok%C3%A9mon_Red/Blue
# ---------------------------------------------------------------------------

class MemoryMap:
    """Static namespace for Pokémon Blue memory addresses."""

    # Player position
    PLAYER_MAP_ID   = 0xD35E   # Current map/area ID
    PLAYER_X        = 0xD362   # Player tile X coordinate
    PLAYER_Y        = 0xD361   # Player tile Y coordinate

    # Battle state
    IN_BATTLE       = 0xD057   # 0 = not in battle, 1 = wild, 2 = trainer
    BATTLE_RESULT   = 0xD16B   # Last battle outcome (game internal)

    # Party data base addresses
    PARTY_COUNT     = 0xD163   # Number of Pokémon in party (0-6)
    PARTY_SPECIES   = 0xD164   # Species IDs for party slots (6 bytes)

    # Level for each party slot (6 offsets)
    PARTY_LEVEL     = [0xD18C, 0xD1B8, 0xD1E4, 0xD210, 0xD23C, 0xD268]

    # Current HP (2 bytes big-endian per slot)
    PARTY_HP        = [0xD16C, 0xD198, 0xD1C4, 0xD1F0, 0xD21C, 0xD248]

    # Max HP (2 bytes big-endian per slot)
    PARTY_MAX_HP    = [0xD18D, 0xD1B9, 0xD1E5, 0xD211, 0xD23D, 0xD269]

    # Gym badges bitmask  (bit 0 = Boulder, bit 1 = Cascade, …, bit 7 = Earth)
    BADGES          = 0xD356

    # Trainer defeat counter (may vary; use for experiment tracking)
    TRAINERS_BEATEN = 0xD5AB


def load_emulator(rom_path: str, headless: bool = True):
    """Load PyBoy with the specified ROM.

    Args:
        rom_path: Path to the Pokémon Blue .gb ROM file.
        headless: If True, use the 'null' window (no display). If False,
                  use SDL2 for live rendering.

    Returns:
        Initialised PyBoy instance.

    Raises:
        FileNotFoundError: If the ROM file does not exist.
        ImportError: If PyBoy is not installed.
    """
    try:
        from pyboy import PyBoy
    except ImportError as exc:
        raise ImportError(
            "PyBoy is required. Install it with: pip install pyboy"
        ) from exc

    rom_file = Path(rom_path)
    if not rom_file.exists():
        raise FileNotFoundError(
            f"ROM not found at '{rom_path}'. "
            "Please provide the Pokémon Blue ROM as 'roms/pokemon_blue.gb'."
        )

    window_type = "null" if headless else "SDL2"
    logger.info("Loading emulator: rom=%s  window=%s", rom_path, window_type)
    pyboy = PyBoy(str(rom_file), window=window_type)
    pyboy.set_emulation_speed(0)  # Run as fast as possible
    return pyboy


def get_screen_array(pyboy) -> np.ndarray:
    """Capture the current Game Boy screen as a numpy array.

    Args:
        pyboy: Running PyBoy instance.

    Returns:
        RGB numpy array of shape (144, 160, 3) with uint8 dtype.
    """
    screen = pyboy.screen.image  # PIL Image
    return np.array(screen, dtype=np.uint8)


def read_byte(pyboy, address: int) -> int:
    """Read a single byte from Game Boy memory.

    Args:
        pyboy: Running PyBoy instance.
        address: 16-bit memory address.

    Returns:
        Byte value (0–255).
    """
    return pyboy.memory[address]


def read_word_be(pyboy, address: int) -> int:
    """Read a big-endian 16-bit word from Game Boy memory.

    Args:
        pyboy: Running PyBoy instance.
        address: Starting 16-bit memory address (high byte first).

    Returns:
        16-bit integer value.
    """
    high = pyboy.memory[address]
    low  = pyboy.memory[address + 1]
    return (high << 8) | low


def read_game_state(pyboy) -> dict:
    """Extract all relevant game variables from memory into a dictionary.

    Args:
        pyboy: Running PyBoy instance.

    Returns:
        Dictionary with keys matching GameState field names.
    """
    mem = MemoryMap
    party_count = min(read_byte(pyboy, mem.PARTY_COUNT), 6)

    party_levels = tuple(
        read_byte(pyboy, mem.PARTY_LEVEL[i]) for i in range(party_count)
    )
    party_hp = tuple(
        read_word_be(pyboy, mem.PARTY_HP[i]) for i in range(party_count)
    )

    in_battle_val = read_byte(pyboy, mem.IN_BATTLE)

    return {
        "map_id":               read_byte(pyboy, mem.PLAYER_MAP_ID),
        "player_x":             read_byte(pyboy, mem.PLAYER_X),
        "player_y":             read_byte(pyboy, mem.PLAYER_Y),
        "badges":               read_byte(pyboy, mem.BADGES),
        "party_levels":         party_levels,
        "party_hp":             party_hp,
        "in_battle":            in_battle_val > 0,
        "battle_result":        0,  # Computed externally from transitions
        "num_trainers_defeated": read_byte(pyboy, mem.TRAINERS_BEATEN),
    }


def send_action(pyboy, press_event, release_event, hold_frames: int = 8) -> None:
    """Press and release a button, ticking the emulator for hold_frames.

    Args:
        pyboy: Running PyBoy instance.
        press_event: PyBoy WindowEvent for the button press.
        release_event: PyBoy WindowEvent for the button release.
        hold_frames: Number of emulator frames to hold the button before
                     releasing. Higher values simulate longer presses.
    """
    if press_event is None:
        # NOOP – just advance the emulator by hold_frames
        for _ in range(hold_frames):
            pyboy.tick()
        return

    pyboy.send_input(press_event)
    for _ in range(hold_frames):
        pyboy.tick()
    pyboy.send_input(release_event)
    pyboy.tick()  # One extra tick to register the release
