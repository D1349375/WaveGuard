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

2. **Taiwan-Aware Signal Control**:
   - Grid BEV scalar features track motorcycle stop-box saturation and hook-turn waiting counts as heuristics computed from vehicle position/speed near the junction.
   - Enforces statutory hard safety constraints: Yellow and All-Red clearance intervals are actively held every simulation step by a state machine (re-asserting the signal state so SUMO's own program cannot silently auto-advance past them) and cannot be bypassed by RL agents.
   - **Current limitation**: the auto-generated traffic light programs are a plain 2-phase (arterial/cross-street) + yellow/all-red cycle; there is no dedicated two-stage left-turn (hook-turn) waiting lane, motorcycle stop-box lane, or pedestrian-exclusive/LPI phase defined in the network geometry yet. Tracked as future work rather than claimed as done.

3. **Multi-Agent Graph Attention Coordination (CoLight)**:
   - Centralized Training with Decentralized Execution (**CTDE**).
   - GAT layers aggregate learned per-intersection state embeddings (from the BEV/scalar encoder) from neighboring intersections with dynamic attention weights, verified to actually change a node's output when its neighbor's input changes (not merely produce the right output shape).

4. **Multi-Objective Safety & Equity Reward**:
   $$R = \alpha R_{flow} + \beta R_{safety} + \gamma R_{equity}$$
   - $R_{flow}$: Negative average intersection vehicle delay.
   - $R_{safety}$: Penalizes Time-to-Collision conflicts ($TTC < 3.0s$).
   - $R_{equity}$: Minimizes waiting time disparity between motorcycles and passenger cars.

5. **Fail-Safe Fallback Mechanism**:
   - Communication watchdog (`FailSafeCoordinator`) monitors the primary MARL agent's health; upon sustained failure it degrades to a decentralized **Max-Pressure** rule engine (computed from real per-lane queue differentials, not a BEV-grid approximation) and recovers automatically once healthy again.
   - Exercised end-to-end (not just unit-tested in isolation) by `scripts/evaluate.py`, which injects a simulated sensor/communication outage mid-run as a fourth "WaveGuard-AI+FailSafe" comparison arm.

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
- WaveGuard MARL + FailSafe watchdog (with a simulated sensor/comms outage injected mid-run)

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
