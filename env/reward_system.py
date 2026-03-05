"""
reward_system.py
----------------
Reward engineering for the Pokémon Blue RL environment.

Rewards are computed by comparing game-state snapshots before and after
each action. The system encourages:
  - Exploration (visiting new map tiles / areas)
  - Progression (levelling up, winning battles, earning badges)
  - Survival (penalising Pokémon fainting and lost battles)
  - Efficiency (small per-step time penalty)
"""

from dataclasses import dataclass, field
from typing import Dict, Set, Tuple


@dataclass
class RewardConfig:
    """Configurable reward magnitudes.

    Attributes can be overridden via a config dict loaded from YAML.
    """
    explore_new_tile: float = 1.0
    explore_new_map: float = 5.0
    win_battle: float = 10.0
    level_up: float = 20.0
    earn_badge: float = 50.0
    defeat_trainer: float = 15.0
    lose_battle: float = -10.0
    pokemon_faint: float = -5.0
    step_penalty: float = -0.01
    heal_pokemon: float = 2.0

    @classmethod
    def from_dict(cls, cfg: dict) -> "RewardConfig":
        """Instantiate from a dictionary (e.g. loaded from YAML).

        Args:
            cfg: Dictionary with reward magnitude keys.

        Returns:
            RewardConfig instance.
        """
        return cls(**{k: v for k, v in cfg.items() if hasattr(cls, k)})


@dataclass
class GameState:
    """Snapshot of relevant game variables read from memory.

    Attributes:
        map_id: Current map/area identifier.
        player_x: Player X tile coordinate.
        player_y: Player Y tile coordinate.
        badges: Bitmask of earned Gym badges.
        party_levels: List of Pokémon levels in the player's party.
        party_hp: List of current HP values for each party Pokémon.
        in_battle: Whether the player is currently in a battle.
        battle_result: 0 = ongoing, 1 = won, -1 = lost.
        num_trainers_defeated: Total trainers defeated (running tally).
    """
    map_id: int = 0
    player_x: int = 0
    player_y: int = 0
    badges: int = 0
    party_levels: Tuple[int, ...] = field(default_factory=tuple)
    party_hp: Tuple[int, ...] = field(default_factory=tuple)
    in_battle: bool = False
    battle_result: int = 0
    num_trainers_defeated: int = 0


class RewardSystem:
    """Computes shaped rewards by diffing consecutive game states.

    Args:
        config: RewardConfig instance with reward magnitudes.
    """

    def __init__(self, config: RewardConfig | None = None) -> None:
        self.config = config or RewardConfig()
        self._visited_tiles: Set[Tuple[int, int, int]] = set()  # (map_id, x, y)
        self._visited_maps: Set[int] = set()
        self._prev_badges: int = 0
        self._prev_levels: Tuple[int, ...] = ()
        self._prev_hp: Tuple[int, ...] = ()
        self._prev_trainers: int = 0

    def reset(self) -> None:
        """Clear exploration history and per-episode counters."""
        # NOTE: visited tiles are intentionally NOT cleared on reset so that
        # cumulative exploration is tracked across episodes. Override here
        # if you want per-episode exploration rewards.
        self._prev_badges = 0
        self._prev_levels = ()
        self._prev_hp = ()
        self._prev_trainers = 0

    def compute(self, prev: GameState, curr: GameState) -> float:
        """Compute the shaped reward for a single environment transition.

        Args:
            prev: Game state snapshot before the action.
            curr: Game state snapshot after the action.

        Returns:
            Total scalar reward for the transition.
        """
        reward: float = 0.0
        cfg = self.config

        # --- Step penalty (encourages efficiency) ---
        reward += cfg.step_penalty

        # --- Exploration: new tile ---
        tile_key = (curr.map_id, curr.player_x, curr.player_y)
        if tile_key not in self._visited_tiles:
            self._visited_tiles.add(tile_key)
            reward += cfg.explore_new_tile

        # --- Exploration: new map/area ---
        if curr.map_id not in self._visited_maps:
            self._visited_maps.add(curr.map_id)
            reward += cfg.explore_new_map

        # --- Progression: badges ---
        new_badges = bin(curr.badges).count("1") - bin(prev.badges).count("1")
        if new_badges > 0:
            reward += new_badges * cfg.earn_badge

        # --- Progression: Pokémon level ups ---
        if curr.party_levels and prev.party_levels:
            level_gain = sum(
                max(0, c - p)
                for c, p in zip(curr.party_levels, prev.party_levels)
            )
            reward += level_gain * cfg.level_up

        # --- Progression: trainer defeats ---
        trainer_gain = curr.num_trainers_defeated - prev.num_trainers_defeated
        if trainer_gain > 0:
            reward += trainer_gain * cfg.defeat_trainer

        # --- Battle outcome ---
        if not prev.in_battle and not curr.in_battle:
            if curr.battle_result == 1:
                reward += cfg.win_battle
            elif curr.battle_result == -1:
                reward += cfg.lose_battle

        # --- Survival: Pokémon fainted ---
        if curr.party_hp and prev.party_hp:
            fainted = sum(
                1 for c, p in zip(curr.party_hp, prev.party_hp) if p > 0 and c == 0
            )
            reward += fainted * cfg.pokemon_faint

            # Healing reward (HP recovered while not in battle)
            if not curr.in_battle:
                healed = sum(
                    max(0, c - p)
                    for c, p in zip(curr.party_hp, prev.party_hp)
                )
                if healed > 0:
                    reward += cfg.heal_pokemon

        return reward

    @property
    def total_tiles_visited(self) -> int:
        """Total unique tiles visited across all episodes."""
        return len(self._visited_tiles)

    @property
    def total_maps_visited(self) -> int:
        """Total unique maps visited across all episodes."""
        return len(self._visited_maps)

    def get_exploration_stats(self) -> Dict[str, int]:
        """Return exploration statistics dictionary."""
        return {
            "tiles_visited": self.total_tiles_visited,
            "maps_visited": self.total_maps_visited,
        }
