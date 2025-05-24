# SCRMBL

**SCRMBL** (StarCraft Reinforcement Model-Based Learning) is an extensible framework designed for reinforcement learning research in StarCraft II, leveraging Blizzard's StarCraft II API. The project aims to facilitate experimentation with various RL algorithms and environments within the complex dynamics of StarCraft II.

## 🌟 Objectives

* Implement and evaluate diverse RL algorithms (e.g., DQN, PPO) in StarCraft II scenarios.
* Provide a modular architecture for easy integration of new models and environments.
* Enable reproducible research through standardized training scripts and configurations.

## 📂 Project Structure

```
SCRMBL/
├── algos/                 # Implementation of RL algorithms
├── envs/                  # Environment wrappers for StarCraft II
├── managers/              # Training and evaluation pipelines
├── networks/              # Neural network architectures
├── scenarios/             # Specific StarCraft II scenarios and maps
├── spaces/                # Action and observation space definitions
├── utils/                 # Utility functions and helpers
├── DQNExample.py          # Example script using DQN
├── StableBaselines3_example.py  # Example using Stable Baselines3
├── dqn_v2.py              # Updated DQN implementation
├── dzb.dqn.h5             # Pretrained DQN model for 'dzb' scenario
├── mtb.dqn.h5             # Pretrained DQN model for 'mtb' scenario
├── requirements.txt       # Python dependencies
└── README.md              # Project overview
```

## 🚀 Getting Started

### Prerequisites

* Python 3.7 or higher
* StarCraft II installed
* [pysc2](https://github.com/deepmind/pysc2) for environment interfacing

### Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/c-daly/SCRMBL.git
   cd SCRMBL
   ```

2. Create and activate a virtual environment:

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install the required packages:

   ```bash
   pip install -r requirements.txt
   ```

## 🧪 Usage

### Training an Agent

To train an agent using the DQN algorithm:

```bash
python DQNExample.py
```

For training with Stable Baselines3:

```bash
python StableBaselines3_example.py
```

### Evaluation

Evaluate a pretrained model:

```bash
python evaluate.py --model_path models/dqn_model.h5 --scenario mtb
```

## 📚 Resources

* [Blizzard StarCraft II API](https://github.com/Blizzard/s2client-api)
* [PySC2 by DeepMind](https://github.com/deepmind/pysc2)
* [Stable Baselines3](https://github.com/DLR-RM/stable-baselines3)

## 📄 License

This project is licensed under the MIT License.
