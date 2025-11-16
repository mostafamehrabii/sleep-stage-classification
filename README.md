# Mamba Meets Sleep: Do State Space Models Outperform CNNs for EEG Classification?

[![IEEE](https://img.shields.io/badge/IEEE-Conference-blue.svg)](https://ieeexplore.ieee.org)
[![Python](https://img.shields.io/badge/Python-3.8+-green.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Official implementation of **"Mamba Meets Sleep: Do State Space Models Outperform CNNs for EEG Classification?"**

**Authors:** Mostafa Mehrabi, Hamed Malek  
**Affiliation:** Faculty of Computer Science and Computer Engineering, Shahid Beheshti University, Tehran, Iran
---
Automatic sleep stage classification from EEG signals is critical for diagnosing sleep disorders, yet most studies evaluate architectures in isolation. We present a systematic evaluation of 15 models across five families: traditional machine learning, residual CNNs, hybrid ResNet-BiLSTM, efficient CNNs, and state space models. Using Sleep-EDF with leave-one-subject-out cross-validation on 20 subjects, we assessed subject-independent generalization. **Mamba Base achieved the highest accuracy (93.32% ± 3.01%)** and recall (87.11%), while **ResNet8 provided comparable accuracy (93.20%) with lowest variance (2.47%)**. Our comprehensive comparison reveals clear trade-offs between accuracy, stability, efficiency, and deployment constraints.

# Key Results

| Model | Accuracy (%) | Std Dev (%) | Parameters | Cohen's κ |
|-------|--------------|-------------|------------|-----------|
| **Mamba Base** | **93.32** | 3.01 | 703K | 0.8489 |
| **ResNet8** | **93.20** | **2.47** | 2.18M | 0.8485 |
| ResNet8-BiLSTM | 92.83 | 3.14 | 2.82M | 0.8428 |
| ResNet12 | 92.63 | 3.87 | 3.40M | 0.8401 |
| Mamba Small | 92.52 | 3.04 | 410K | 0.8389 |

*Full results for all 15 models available in the paper.*

---

## Repository Structure

```
├── models/                    # Model implementations
│   ├── resnet.py             # ResNet variants (8/12/16 layers)
│   ├── resnet_bilstm.py      # Hybrid ResNet-BiLSTM architectures
│   ├── mamba.py              # Mamba Small & Base (State Space Models)
│   ├── traditional_ml.py     # Random Forest, XGBoost, LightGBM
│   └── efficient_nets.py     # EfficientNet B1, MobileNetV4, ConvNeXt
├── data/                      # Data preprocessing scripts & instructions
├── results/                   # Output directory for trained models
├── docs/                      # Documentation & paper
├── requirements.txt           # Python dependencies
└── README.md                  # This file
```
