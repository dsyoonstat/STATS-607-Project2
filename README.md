# Simulation Studies for the Adaptive Reference-Guided Estimator

Umich STATS 607 Unit 2 Project, Fall 2025

This project reproduces and extends simulation results of [**Yoon and Jung (2025)**](https://onlinelibrary.wiley.com/doi/full/10.1002/sta4.70081).
It implements three simulation modules:
1. **Single-spike, single-reference case** (Reproducing tables 1(normal) and 3($t$))  
2. **Multi-spike, multi-reference** (Reproducing tables 2(normal) and 4($t$))  
3. **Convergence-rate analysis** for Theorem 4 (Normal distribution only, not in the original paper)

For details, see the paper and `ADEMP.md` (requires MathJax for equation rendering).

All simulations output summary CSV tables under `results/tables/`, and plots summarizing the results are generated under `results/figures/`.

---

## Setup Instructions

1. **Clone or open the repository**, then create a Python environment:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

2. **Install required packages**:
   ```bash
   pip install -r requirements.txt
   ```

   The following packages are needed:
   ```
   numpy>=1.24
   pandas>=2.0
   tqdm>=4.65
   matplotlib>=3.7
   ```
   (Python ≥ 3.9 recommended)

---

## How to Run the Complete Analysis

This project is fully automated via the included **Makefile**.

Run the full pipeline:
```bash
make all
```

This will:
1. Run all simulations (`src/simulation.py`) — saves CSV tables under `results/tables/`
2. Generate all figures (`src/plot.py`) — saves plots under `results/figures/`

You can also run individual steps:
```bash
make simulate     # Run simulations only
make figures      # Generate plots only
make clean        # Remove all generated files
make test         # Run test suite
```

---

## Estimated Runtime

- **Single-spike + Multi-spike + Convergence-rate simulations** (`make simulate`):  
  < 10 minutes total on a high-end laptop (Apple M4 pro).  
- Plot generation (`make figures`): < 10 seconds.

You can reduce runtime by lowering `n_trials` or `p_list` values inside `src/simulation.py`.

---

## Summary of Key Findings

 - The ARG estimator significantly outperformed the naive PCA-based estimator under multivariate normal data, yielding smaller principal angles to the true subspace. The performance gap widened as the reference vector became more aligned with the true principal component subspace. Under the multivariate $t$ distribution, where the assumptions of Yoon and Jung (2025) are not satisfied, the performance gain was smaller but still consistent — the ARG estimator continued to outperform the naive estimator.
 - However, the estimated convergence rates of $|u_1^{\top}d_1|$ varied substantially across different signal-to-noise ratios, and no consistent trend (e.g., increasing $\alpha$ with SNR) was observed. This irregularity appears to stem from the complex behavior of eigenvalues and eigenvectors.


---
