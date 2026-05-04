from dataclasses import dataclass, field
from typing import Dict, Set, Tuple


@dataclass
class RewardConfig:

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
        return cls(**{k: v for k, v in cfg.items() if hasattr(cls, k)})


@dataclass
class GameState:

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

    def __init__(self, config: RewardConfig | None = None) -> None:

        self.config = config or RewardConfig()

        self._visited_tiles: Set[Tuple[int, int, int]] = set()
        self._visited_maps: Set[int] = set()

    def reset(self) -> None:
        pass

    def compute(self, prev: GameState, curr: GameState) -> float:

        reward = 0.0
        cfg = self.config

        # Step penalty
        reward += cfg.step_penalty

        # --- Exploration: new tile ---
        tile_key = (curr.map_id, curr.player_x, curr.player_y)

        if tile_key not in self._visited_tiles:
            self._visited_tiles.add(tile_key)
            reward += cfg.explore_new_tile

        # --- Exploration: new map ---
        if curr.map_id not in self._visited_maps:
            self._visited_maps.add(curr.map_id)
            reward += cfg.explore_new_map

        # --- Gym badges ---
        new_badges = bin(curr.badges).count("1") - bin(prev.badges).count("1")

        if new_badges > 0:
            reward += new_badges * cfg.earn_badge

        # --- Level ups ---
        if curr.party_levels and prev.party_levels:

            level_gain = sum(
                max(0, c - p)
                for c, p in zip(curr.party_levels, prev.party_levels)
            )

            reward += level_gain * cfg.level_up

        # Trainer defeats 
        trainer_gain = curr.num_trainers_defeated - prev.num_trainers_defeated

        if trainer_gain > 0:
            reward += trainer_gain * cfg.defeat_trainer

        # Battle result 
        if not prev.in_battle and not curr.in_battle:

            if curr.battle_result == 1:
                reward += cfg.win_battle

            elif curr.battle_result == -1:
                reward += cfg.lose_battle

        # Pokémon fainted 
        if curr.party_hp and prev.party_hp:

            fainted = sum(
                1 for c, p in zip(curr.party_hp, prev.party_hp)
                if p > 0 and c == 0
            )

            reward += fainted * cfg.pokemon_faint

            # Healing
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
        return len(self._visited_tiles)

    @property
    def total_maps_visited(self) -> int:
        return len(self._visited_maps)

    def get_exploration_stats(self) -> Dict[str, int]:

        return {
            "tiles_visited": self.total_tiles_visited,
            "maps_visited": self.total_maps_visited,
        }