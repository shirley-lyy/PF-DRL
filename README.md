# PF-DRL: Personalized Federated Deep Reinforcement Learning for Multi-UAV Trajectory Optimization in Internet of Vehicles

## Overview

This repository provides the source code for **PF-DRL**, a Personalized Federated Deep Reinforcement Learning framework for autonomous trajectory optimization of UAV swarms in UAV-assisted Internet of Vehicles (IoV).

### Key Features

- **PF-MADDPG**: A MADDPG-based instantiation of the PF-DRL framework with personalized federated aggregation.
- **Lightweight Design**: Selectively aggregates only Critic networks, reducing communication overhead by **~50%** compared to full aggregation (Section IV-C).
- **Adaptive Weight Adjustment**: A perturb-and-observe algorithm (Algorithm 2) that dynamically tunes the personalization weight α based on EMA-smoothed episode returns, without requiring distribution divergence estimation.
- **Probabilistic A2G Channel Model**: Air-to-Ground communication model with LoS/NLoS probability (Eqs. 1–3).
- **UAV Energy Model**: Rotary-wing propulsion energy (Eq. 4) + offloading computing energy (Eq. 5).
- **Safety Constraints**: Soft-contact collision avoidance between UAVs, static obstacles, and boundary enforcement.

---

## System Requirements

- **Python**: 3.8 or higher
- **GPU**: NVIDIA GPU with CUDA 11.x+ (optional, CPU training is supported)
- **RAM**: ≥ 8 GB recommended

---

## Installation

### 1. Create a Virtual Environment (Recommended)

```bash
# Using conda
conda create -n pfdrl python=3.9
conda activate pfdrl
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Quick Start

### Training (Mode 0 — Standard MADDPG baseline)

```bash
python main.py --training-mode 0
```

### Training (Mode 1 — Lightweight PF-MADDPG, fixed α)

```bash
python main.py \
    --training-mode 1 \
    --lightweight \
    --alpha 0.5 \
    --agg-interval 5
```

### Training (Mode 2 — Lightweight PF-MADDPG, adaptive α)

```bash
python main.py \
    --training-mode 2 \
    --lightweight \
    --alpha 0.5 \
    --alpha-step 0.003 \
    --ema-beta 0.9 \
    --agg-interval 5
```

### Evaluation Only

```bash
python main.py \
    --evaluate \
    --model-dir ./model/complex_service \
    --training-mode 2
```
