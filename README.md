# Mamba Meets Sleep: Do State Space Models Outperform CNNs for EEG Classification?

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
├── results/                   # Output directory for trained models
└── README.md                  # This file
```


## Model Architectures

### State Space Models (Mamba)
- **Mamba Small:** 410K parameters
- **Mamba Base:** 703K parameters
- Linear complexity O(n) for sequence modeling
- Selective state space mechanism

### Residual CNNs
- **ResNet8:** 2.18M parameters (best stability)
- **ResNet12:** 3.40M parameters
- **ResNet16:** 4.62M parameters
- Skip connections + batch normalization

### Hybrid Models
- ResNet8/12/16 + BiLSTM layers
- Combines spatial and temporal feature extraction

### Efficient CNNs
- **EfficientNet B1:** 7.00M parameters
- **MobileNetV4 Small:** 2.07M parameters  
- **ConvNeXt Femto:** 4.54M parameters

### Traditional ML
- Random Forest, Extra Trees, XGBoost, LightGBM
- 154 engineered time/frequency domain features

---

## Evaluation Protocol

**Cross-Validation:** Leave-One-Subject-Out (LOSO)  
**Subjects:** 20  
**Metrics:** Accuracy, Precision, Recall, F1-Score, Cohen's Kappa  
**Significance:** Subject-independent generalization reflects real clinical deployment

---

## Key Findings

1. **Simpler architectures often outperform complex ones:** ResNet8 beats ResNet12/16
2. **Adding recurrence doesn't always help:** BiLSTM layers increased cost without gains
3. **State space models are competitive:** Mamba achieves highest accuracy with linear complexity
4. **Traditional ML remains viable:** 90-91% accuracy with minimal compute
5. **Trade-offs matter:** Choose based on deployment constraints (GPU, edge, CPU-only)

---

## Citation

If you use this code or findings in your research, please cite:

```bibtex
@software{mehrabi2025mamba,
  title={Mamba Meets Sleep: Do State Space Models Outperform CNNs for EEG Classification?},
  author={Mostafa Mehrabi},
  year={2025},
  url={https://github.com/mostafamehrabii/sleep-stage-classification}
}
```
