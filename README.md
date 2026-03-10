# DDT-SCN Simulation Code

## Overview

This repository contains the complete simulation code for the paper:

> **A Dual-Scale Digital Twin-Enabled System Architecture for Dynamic Resource Orchestration in Satellite Computing Networks**
>
> Published in *IEEE Systems Journal*

The simulation implements an event-driven LEO satellite network simulator with seven scheduling algorithms and four experiment scripts to reproduce all figures in the paper.

## Repository Structure

```
DDT-SCN-Simulation-Code/
├── simulation_core.py          # Core simulation engine
├── satellite_env.py            # Gymnasium RL environment
├── train_fdt_ppo.py            # PPO training pipeline
├── baseline_experiment.py      # Fig. 3: Baseline comparison
├── ablation_experiment.py      # Fig. 4: Ablation study
├── convergence_experiment.py   # Fig. 5: Convergence analysis
├── scalability_experiment.py   # Fig. 6: Scalability analysis
├── ppo_fdt.zip                 # Pre-trained PPO model
└── README.md                   # This file
```

## Dependencies

- Python ≥ 3.8
- NumPy
- Matplotlib
- Gymnasium
- Stable-Baselines3

Install all dependencies:

```bash
pip install numpy matplotlib gymnasium stable-baselines3
```

## Simulation Architecture

### Core Engine (`simulation_core.py`)

The simulation models a Walker-Delta LEO constellation with the following features:

- **M/G/1 queuing model** for per-satellite task processing
- **Heterogeneous tasks**: Category-0 (delay-sensitive, 30%) and Category-1 (delay-tolerant, 70%)
- **Priority scheduling**: Category-0 tasks are prioritized over Category-1
- **Graduated hotspot traffic**: Non-uniform task distribution modeling real-world traffic patterns
- **Physical layer modeling**: Propagation delay, transmission time, ISL bandwidth constraints

### Scheduling Algorithms

| Algorithm | Description |
|-----------|-------------|
| **Local** | Process all tasks locally without offloading |
| **Random** | Randomly offload to a reachable satellite |
| **Greedy** | Offload to the least-loaded neighbor |
| **Optimal** | Oracle-based global minimum delay (upper bound) |
| **Online-DRL** | Ground-based PPO with periodic full-state updates |
| **Offline-DRL** | On-board inference with cached global state |
| **DDT-SCN** | Proposed dual-scale architecture with event-triggered sync |

### Reinforcement Learning

- **Algorithm**: Proximal Policy Optimization (PPO)
- **Framework**: Stable-Baselines3
- **Network**: MLP [128, 128, 64]
- **Learning rate**: 3×10⁻⁴
- **Discount factor**: γ = 0.99

## Reproducing Results

### Step 1: Train the PPO Model (Optional)

A pre-trained model (`ppo_fdt.zip`) is included. To retrain:

```bash
python train_fdt_ppo.py
```

Training takes approximately 5-10 minutes on a modern CPU.

### Step 2: Generate Figures

Each experiment script generates both PDF and PNG output:

```bash
# Fig. 3: Baseline comparison (7 algorithms × 6 arrival rates)
python baseline_experiment.py

# Fig. 4: Ablation study (5 DDT-SCN variants × 6 arrival rates)
python ablation_experiment.py

# Fig. 5: PPO convergence (reward and latency over episodes)
python convergence_experiment.py

# Fig. 6: Scalability analysis (3 algorithms × 3 scales)
python scalability_experiment.py
```

### Expected Runtime

| Experiment | Approximate Runtime |
|------------|-------------------|
| Baseline comparison | 3-5 minutes |
| Ablation study | 2-3 minutes |
| Convergence analysis | 5-10 minutes |
| Scalability analysis | 3-5 minutes |

## Key Parameters

| Parameter | Value |
|-----------|-------|
| Number of satellites | 25 (5×5 Walker-Delta) |
| Orbital altitude | 1000 km |
| Processing rate | 100 Mcycles/s |
| ISL bandwidth | 100 Mbps |
| Task CPU (Cat-0) | [150M, 300M] cycles |
| Task CPU (Cat-1) | [300M, 500M] cycles |
| Deadline (Cat-0) | [8, 20] s |
| Deadline (Cat-1) | [25, 55] s |
| Sync threshold δ | 0.2 |

## Citation

If you use this code in your research, please cite:

```bibtex
@article{xu2026ddt,
  title={A Dual-Scale Digital Twin-Enabled System Architecture for Dynamic Resource Orchestration in Satellite Computing Networks},
  author={Xu, Zhenghuan and Yang, Jian and Zheng, Quan and Tan, Xiaobin},
  journal={IEEE Systems Journal},
  year={2026}
}
```

## License

This code is released for academic research purposes.
