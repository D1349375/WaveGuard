# WaveGuard: 5G-Driven Multi-Agent Reinforcement Learning Traffic Signal Optimization Platform

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://python.org)
[![SUMO](https://img.shields.io/badge/Eclipse%20SUMO-1.20%2B-brightgreen.svg)](https://eclipse.dev/sumo)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange.svg)](https://pytorch.org)
[![License](https://img.shields.io/badge/License-Apache%202.0-lightgrey.svg)](LICENSE)

WaveGuard is an intelligent traffic signal coordination system designed for Taiwan's complex urban environments featuring high-density motorcycle-car mixed traffic. Powered by **Multi-Agent Reinforcement Learning (MARL)** with **Graph Attention Networks (GAT)**, WaveGuard coordinates arterial corridors to anticipate, mitigate, and dissipate "phantom traffic jams" (stop-and-go shockwaves).

---

## 🌟 Core Features & Technical Innovations

1. **Dual-Channel Grid-based BEV Occupancy Matrix**:
   - Replaces 1D queue length with a $12 \times 12$ spatial occupancy tensor separating passenger cars and motorcycles.
   - Captures non-lane motorcycle filtering (*Lane Filtering*) enabled by SUMO's sublane model (`--lateral-resolution 0.8`).
   - Scalar features track motorcycle stop-box saturation, hook-turn waiting count, and pedestrian countdown.

2. **Taiwan-Specific Multi-Phase Action Structure**:
   - Explicitly models Two-stage Left Turn (*Hook-turn*) clearance, Motorcycle Stop Box surge green, and Leading Pedestrian Intervals (*LPI*).
   - Enforces statutory hard safety constraints: Yellow (3s) and All-Red (2s) clearance intervals are managed by a state machine and cannot be bypassed by RL agents.

3. **Multi-Agent Graph Attention Coordination (CoLight)**:
   - Centralized Training with Decentralized Execution (**CTDE**).
   - GAT layers aggregate traffic pressure vectors from neighboring intersections with dynamic attention weights.

4. **Multi-Objective Safety & Equity Reward**:
   $$R = \alpha R_{flow} + \beta R_{safety} + \gamma R_{equity}$$
   - $R_{flow}$: Negative average intersection vehicle delay.
   - $R_{safety}$: Penalizes Time-to-Collision conflicts ($TTC < 3.0s$).
   - $R_{equity}$: Minimizes waiting time disparity between motorcycles and passenger cars.

5. **Fail-Safe Fallback Mechanism**:
   - Communication watchdog monitors edge sensor feeds; upon sensor timeout or network degradation, control degrades smoothly to a decentralized **Max-Pressure** rule engine.

---

## 📁 Repository Structure

```
WaveGuard/
├── .gitignore
├── requirements.txt
├── pytest.ini
├── README.md
├── configs/
│   ├── default_config.yaml      # Simulation, BEV grid, reward weights, model hyperparameters
│   └── scenario_config.yaml     # Ghost traffic jam incident injection settings
├── sumo_nets/
│   ├── benchmark/               # 4-intersection 2x2 grid benchmark network
│   └── taiwan_xinyi/            # 6-intersection Taiwan Xinyi Road arterial corridor
├── envs/
│   ├── traci_wrapper.py         # SUMO TraCI lifecycle, BEV grid extraction, TTC tracking
│   └── waveguard_env.py         # Gymnasium MARL environment with multi-objective reward
├── agents/
│   ├── gat_net.py               # BEV Convolutional Encoder & Graph Attention Network (GAT)
│   ├── colight_agent.py         # CoLight CTDE MARL Agent with Experience Replay
│   └── rule_controllers.py      # Fixed-Time baseline, Max-Pressure, FailSafeCoordinator
├── scripts/
│   ├── build_networks.py        # Automated SUMO network and route builder
│   ├── train.py                 # MARL training pipeline with loss & reward logging
│   ├── evaluate.py              # Quantitative evaluation comparing Fixed-Time, Max-Pressure, AI
│   └── run_scenario.py          # Ghost Traffic Jam incident shockwave dissipation demo
├── tests/
│   ├── test_env.py              # Environment lifecycle and BEV extraction unit tests
│   ├── test_agent.py            # Neural network and controller unit tests
│   └── test_simulation.py       # Corridor simulation and incident injection tests
└── results/                     # Checkpoints, logs, comparison plots, and interactive map
```

---

## 🚀 Quick Start

### 1. Installation

```bash
# Clone the repository
git clone https://github.com/D1349375/WaveGuard.git
cd WaveGuard

# Install dependencies
pip install -r requirements.txt
```

Ensure Eclipse SUMO ($\ge 1.20$) is installed and `SUMO_HOME` is set.

### 2. Generate Road Networks

```bash
python scripts/build_networks.py
```

### 3. Run Unit Tests

```bash
pytest
```

---

## 🔬 Training & Evaluation

### Train CoLight MARL Agent

```bash
python scripts/train.py --episodes 15 --steps-per-episode 300 --net taiwan_xinyi
```
Outputs model checkpoints to `results/checkpoints/` and training curves to `results/training_curves.png`.

### Evaluate vs Classical Baselines

```bash
python scripts/evaluate.py --steps 300 --net taiwan_xinyi
```
Generates comparison report `results/evaluation_summary.md` comparing:
- Fixed-Time Control (Baseline)
- Max-Pressure Control
- WaveGuard MARL (AI)

---

## 🎬 Shockwave Dissipation Demo (Ghost Traffic Jam)

Run the incident injection experiment on the Taiwan Xinyi Road corridor:

```bash
python scripts/run_scenario.py
```

- Injects an artificial minor collision at $T=250s$ (45s duration) on edge `E_J1_J2`.
- Tracks backward-propagating shockwave across upstream ($J_1$), incident ($J_2$), and downstream ($J_3$) junctions.
- Outputs comparative time-series data and high-resolution chart `results/incident_shockwave_dissipation.png`.
- Generates interactive corridor map `results/xinyi_corridor_heatmap.html`.

---

## 📄 License

Apache License 2.0. Developed for the 2026 Chunghwa Telecom Smart Innovation Application Competition.
