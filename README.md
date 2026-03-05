# Pokémon Blue Reinforcement Learning Agent

## Project Overview

This repository contains a full reinforcement learning (RL) pipeline designed to train an autonomous agent to play **Pokémon Blue**. Built with PyTorch and Gymnasium, the system directly interfaces with the Game Boy emulator [PyBoy](https://github.com/Baekalfen/PyBoy), extracting visual observations and internal game memory state to shape rewards.

The project is structured like a modern deep learning research codebase, focusing heavily on environment engineering, reward shaping, and training stability over raw algorithmic novelty.

### Status Highlights
- **Architecture**: Deep Q-Learning (Dueling DQN variant)
- **Environment**: Custom `gymnasium.Env` wrapper over PyBoy
- **Observations**: 84x84 stacked grayscale frames
- **Actions**: 9 discrete Game Boy buttons (`A`, `B`, `D-pad`, `START`, `SELECT`, `NOOP`)

---

## Reinforcement Learning Approach

### Deep Q-Network (DQN)
The agent relies on a classical Deep Q-Network design (from the original DeepMind Atari paper), bolstered by modern improvements:
- **Dueling Architecture**: Separates State-Value `V(s)` from Advantage `A(s, a)` to better handle states where multiple actions (like bumping into walls) have similar low values.
- **Experience Replay**: Decorrelates sequential Game Boy frames by sampling randomized mini-batches from a cyclic memory buffer (memory-optimized using uint8 numpy arrays).
- **Target Network**: Uses a delayed target network `Q_target` updated periodically to stabilize the temporal difference (TD) bootstrapping.
- **Epsilon-Greedy Exploration**: A linearly decaying exploration schedule forces the agent to try random inputs early before exploiting the learned policy.

### Neural Network Architecture
1. **Conv2D Encoder**: Three convolutional layers (`8x8 stride 4`, `4x4 stride 2`, `3x3 stride 1`) extract spatial features from the stacked frames.
2. **Flatten & FC**: Flattens the volume and feeds into a 512-unit fully connected layer.
3. **Dueling Head**: Splits into a 1-unit value stream and a 9-unit advantage stream.

---

## Environment Design

The `PokemonBlueEnv` implements the standard OpenAI `gymnasium` interface (`reset`, `step`, `render`).

### Observation Preprocessing
Raw `144x160` RGB output from the emulator is processed into neural-net ready tensors:
1. Converted to grayscale.
2. Downsampled to `84x84`.
3. Stacked temporally `(4 frames)` to give the CNN a sense of velocity and direction (e.g., is the player currently walking left or right?).

### Memory Hooking
We utilize explicit memory map locations (hooking into `$D35E`, `$D362`, etc.) to extract raw semantic information from the emulator silently. This avoids the necessity of writing a fragile OCR system to read battle text.

---

## Reward Engineering

Designing a reward function for a massive open-world RPG like Pokémon is notoriously difficult. A naive reward for "winning the game" is far too sparse. We use a shaped reward system (`env/reward_system.py`) encouraging:

1. **Exploration (Curiosity)**: 
   - `+1.0` for visiting a new `(X, Y)` tile coordinate on a specific map.
   - `+5.0` for entering an entirely new map area (e.g., a new route, a house).
2. **Progression**:
   - `+20.0` for leveling up a Pokémon in the party.
   - `+10.0` for winning a battle.
   - `+50.0` for earning a Gym Badge.
3. **Survival**:
   - `-5.0` when a Pokémon in the party faints.
   - `-10.0` for losing a battle (white out).
   - `-0.01` negative step penalty to encourage efficient, purposeful movement rather than spinning in circles.

---

## Get Started

### 1. Installation

Requires Python 3.10+.

```bash
# Clone the repository
git clone https://github.com/yourusername/pokemon-blue-ai.git
cd pokemon-blue-ai

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # Or `venv\Scripts\activate` on Windows

# Install dependencies
pip install -r requirements.txt
```

### 2. Add the ROM
Place a legitimate dump of **Pokémon Blue** inside the `roms/` folder and name it `pokemon_blue.gb`.

```text
pokemon-blue-ai/
└── roms/
    └── pokemon_blue.gb
```

---

## Running the Code

### Training the Agent
Modify configuration in `config/training_config.yaml` as needed, then start training:

```bash
python train.py --config config/training_config.yaml
```

Training saves metrics to `experiments/training_logs/` and periodic network weights to `models/saved_models/`.
You can monitor progress via TensorBoard:

```bash
tensorboard --logdir experiments/training_logs
```

### Evaluating performance
Runs the agent in headless mode using a deterministic policy (`epsilon=0.0`) to measure true performance metrics:

```bash
python evaluate.py --model models/saved_models/latest.pt --episodes 10
```

### Watch the Agent Play
Renders the PyBoy SDL2 window so you can watch your trained agent live:

```bash
python play_trained_agent.py --model models/saved_models/latest.pt --fps 120
```

---

## Results and Visualizations

*Visualizations are generated automatically during training inside the `experiments/` folder.*

### Example Metrics Tracked:
- **Episode Reward** smoothed over time.
- **Unique Tiles Visited** indicating the expansion of the agent's territorial knowledge.
- **Exploration Map Heatmaps** overlaying hot zones in Route 1 / Pallet Town.
- **Loss curves** to diagnose catastrophic forgetting or exploding gradients.

---

## Future Improvements & Stretch Goals

1. **PPO / Actor-Critic Implementation**: Replacing the DQN agent with Proximal Policy Optimization (PPO) via Stable-Baselines3, allowing for more stable convergence in continuous-style environments.
2. **Memory-Based Observations (RNN/LSTM)**: Adding a recurrent layer before the Q-value output to handle the extreme partial observability of Pokémon (e.g., remembering where the wall was a few steps ago).
3. **Save-State Warping**: Training uniquely on difficult segments (e.g., Mt. Moon, Gym Battles) by forcing the environment to `reset()` at specific internal save states rather than the beginning of the game.
