# simulation.py
# - Single-spike, single-reference (Normal + t)
# - Multi-spike, multi-reference (Normal + t)
# - Convergence-rate study for Theorem 4 (Normal)
#
# Outputs:
#   results/tables/*.csv (summary tables)

from typing import List, Tuple
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm

# === import from files ===
from dgps import (
    generate_basis,
    generate_reference_vectors,
    sigma_single_spike,
    sigma_multi_spike,
    sample_normal,
    sample_t,
)
from methods import (
    compute_PC_subspace,
    compute_ARG_PC_subspace,
    compute_negative_ridge_discriminants,
)
from metrics import compute_principal_angles


# ------------------------------------------------------------
# Single-spike, single-reference simulation (Normal + t)
# ------------------------------------------------------------
def run_simulation_single(
    p_list: List[int],
    a_list: List[float],
    n: int,
    nu: int,
    n_trials: int,
    sigma_coef: Tuple[float, float],
    master_seed: int = 725,
) -> None:
    """
    Run the single-spike / single-reference simulation under Normal and t sampling
    and write summary tables (means, stds of principal angles) to disk.

    Parameters
    ----------
    p_list : List[int]
        List of ambient dimensions p to sweep over (e.g., [100, 200, 500, ...]).
    a_list : List[float]
        Reference-vector alignment coefficients a ∈ [0,1]. For each a, the reference
        is constructed as v = a·e1 + sqrt(1-a^2)·e2 using the basis from
        `generate_basis`. Columns in the output are labeled as "a^2={value}".
    n : int
        Sample size per trial.
    nu : int
        Degrees of freedom for Student-t sampling (used only for the t runs).
    n_trials : int
        Number of Monte Carlo trials per (p, distribution) setting.
    sigma_coef : Tuple[float, float]
        Covariance parameters (c1, c2) for the single-spike model:
        Σ = c1·p·e1e1ᵀ + c2·I_p. The true spike direction is u1 = e1.
    master_seed : int, default=725
        Master seed for reproducibility. A NumPy RNG is initialized with this seed,
        and per-trial seeds are drawn from it for Normal and t runs.

    Procedure
    ---------
    For each p in `p_list`:
      1) Draw `n_trials` datasets from N(0, Σ) and t_ν(0, Σ).
      2) For each dataset:
         - Baseline: compute top-1 PCA subspace and its principal angle to u1.
         - ARG: for each a in `a_list`, construct one reference vector v and compute
           the top-1 ARG subspace; record the principal angle to u1.
      3) Aggregate over trials to obtain mean and std for each column
         [baseline, a^2=...].

    Outputs (files)
    ---------------
    Writes four CSV tables to `results/tables/` (index = p values, columns as below):
      - single_normal_mean.csv  : mean principal angles (radians) for Normal runs
      - single_normal_std.csv   : std  of principal angles for Normal runs
      - single_t_mean.csv       : mean principal angles (radians) for t runs
      - single_t_std.csv        : std  of principal angles for t runs

    Columns
    -------
    "baseline"    : top-1 PCA vs u1
    "a^2={value}" : top-1 ARG vs u1 for each a in `a_list` (value is a², formatted)

    Returns
    -------
    None
        Results are saved to disk; nothing is returned.
    """
    # --- paths ---
    base_dir = Path("results")
    tables_dir = base_dir / "tables"
    tables_dir.mkdir(exist_ok=True, parents=True)

    # Column labels (baseline + a^2 columns)
    col_labels = ["baseline"] + [f"a^2={a**2:.2g}" for a in a_list]

    mean_normal = np.zeros((len(p_list), len(col_labels)))
    mean_t      = np.zeros_like(mean_normal)
    std_normal  = np.zeros_like(mean_normal)
    std_t       = np.zeros_like(mean_normal)

    master_rng = np.random.default_rng(master_seed)

    for pi, p in enumerate(tqdm(p_list, desc="single: p sweep", leave=False)):
        # Σ = c1 * p * e1 e1^T + c2 I_p, u1 = e1
        Sigma, u1 = sigma_single_spike(p=p, coef=sigma_coef)
        if u1.ndim == 1:
            u1 = u1[:, None]  # (p,1)

        E = generate_basis(p)  # (p,4)
        mu = np.zeros(p)

        normal_trials, t_trials = [], []
        # Build trials
        for _ in tqdm(range(n_trials), desc=f"single: build trials (p={p})", leave=False):
            seed_n = int(master_rng.integers(0, 2**31 - 1))
            seed_t = int(master_rng.integers(0, 2**31 - 1))
            normal_trials.append(sample_normal(Sigma=Sigma, n=n, mu=mu, seed=seed_n))
            t_trials.append(sample_t(Sigma=Sigma, nu=nu, n=n, mu=mu, seed=seed_t))

        # Baseline (PCA top-1)
        baseline_normal_angles = np.zeros(n_trials)
        baseline_t_angles      = np.zeros(n_trials)
        arg_normal_all = np.zeros((len(a_list), n_trials))
        arg_t_all      = np.zeros((len(a_list), n_trials))

        for i in tqdm(range(n_trials), desc=f"single: baseline eval (p={p})", leave=False):
            U_pc = compute_PC_subspace(samples=normal_trials[i], spike_number=1)
            baseline_normal_angles[i] = compute_principal_angles(u1, U_pc)[0]
            U_pc_t = compute_PC_subspace(samples=t_trials[i], spike_number=1)
            baseline_t_angles[i] = compute_principal_angles(u1, U_pc_t)[0]

        mean_normal[pi, 0] = baseline_normal_angles.mean()
        mean_t[pi, 0]      = baseline_t_angles.mean()
        std_normal[pi, 0]  = baseline_normal_angles.std(ddof=0)
        std_t[pi, 0]       = baseline_t_angles.std(ddof=0)

        # ARG for each a
        for aj, a in enumerate(tqdm(a_list, desc=f"single: ARG loop (p={p})", leave=False)):
            A = np.array([[a, np.sqrt(max(0.0, 1.0 - a*a)), 0.0, 0.0]])
            V = generate_reference_vectors(E, A)  # (p,1)

            for i in range(n_trials):
                U_ARG = compute_ARG_PC_subspace(
                    samples=normal_trials[i], reference_vectors=V, spike_number=1, orthonormal=True
                )
                arg_normal_all[aj, i] = compute_principal_angles(u1, U_ARG)[0]

                U_ARG_t = compute_ARG_PC_subspace(
                    samples=t_trials[i], reference_vectors=V, spike_number=1, orthonormal=True
                )
                arg_t_all[aj, i] = compute_principal_angles(u1, U_ARG_t)[0]

            mean_normal[pi, 1+aj] = arg_normal_all[aj].mean()
            mean_t[pi, 1+aj]      = arg_t_all[aj].mean()
            std_normal[pi, 1+aj]  = arg_normal_all[aj].std(ddof=0)
            std_t[pi, 1+aj]       = arg_t_all[aj].std(ddof=0)

    # Save summaries to results/tables
    idx = p_list
    pd.DataFrame(mean_normal, index=idx, columns=col_labels).to_csv(tables_dir / "single_normal_mean.csv")
    pd.DataFrame(std_normal,  index=idx, columns=col_labels).to_csv(tables_dir / "single_normal_std.csv")
    pd.DataFrame(mean_t,      index=idx, columns=col_labels).to_csv(tables_dir / "single_t_mean.csv")
    pd.DataFrame(std_t,       index=idx, columns=col_labels).to_csv(tables_dir / "single_t_std.csv")

    print("✅ Single simulation complete. Saved in results/tables")


# ------------------------------------------------------------
# Multi-spike, multi-reference simulation (Normal + t)
# ------------------------------------------------------------
def run_simulation_multi(
    p_list: List[int],
    n: int,
    nu: int,
    n_trials: int,
    sigma_coef: Tuple[float, float, float],
    master_seed: int = 725,
) -> None:
    """
    Run the multi-spike / multi-reference simulation under Normal and t sampling
    and write summary tables (means and stds of principal angles) to disk.

    Parameters
    ----------
    p_list : List[int]
        List of ambient dimensions p to sweep over (e.g., [100, 200, 500, ...]).
    n : int
        Sample size per trial.
    nu : int
        Degrees of freedom for Student-t sampling (used only for the t runs).
    n_trials : int
        Number of Monte Carlo trials per (p, distribution) setting.
    sigma_coef : Tuple[float, float, float]
        Covariance coefficients (c1, c2, c3) for the two-spike model:
        Σ = c1·p·u1u1ᵀ + c2·p·u2u2ᵀ + c3·I_p.
        The true eigenvectors U_m = [u1, u2] form the true 2-D subspace.
    master_seed : int, default=725
        Master seed for reproducibility. A NumPy RNG is initialized with this seed,
        and per-trial seeds are drawn from it for Normal and t runs.

    Procedure
    ---------
    For each p in `p_list`:
      1) Construct Σ and the true 2-D spike subspace U_m using `sigma_multi_spike`.
      2) Generate two fixed reference vectors V = [v1, v2] using `generate_reference_vectors`
         with predefined linear combinations of the orthonormal basis from `generate_basis`.
      3) Repeat for n_trials:
         - Sample X ~ N(0, Σ) and X_t ~ t_ν(0, Σ).
         - Compute the top-2 ARG and PCA subspaces for each dataset.
         - Record the two principal angles between each estimated subspace
           and the true subspace U_m.
      4) Aggregate over trials to compute mean and standard deviation of
         principal angles for each of the following comparisons:
             ARG1, PCA1 : first principal angle
             ARG2, PCA2 : second principal angle

    Outputs (files)
    ---------------
    Writes four CSV tables to `results/tables/` (index = p values, columns as below):
      - multi_normal_mean.csv  : mean principal angles (radians) for Normal runs
      - multi_normal_std.csv   : std  of principal angles for Normal runs
      - multi_t_mean.csv       : mean principal angles (radians) for t runs
      - multi_t_std.csv        : std  of principal angles for t runs

    Columns
    -------
    "ARG1" : first principal angle (ARG vs truth)
    "PCA1" : first principal angle (PCA vs truth)
    "ARG2" : second principal angle (ARG vs truth)
    "PCA2" : second principal angle (PCA vs truth)

    Returns
    -------
    None
        Results are saved to disk; nothing is returned.
    """
    # --- paths ---
    base_dir = Path("results")
    tables_dir = base_dir / "tables"
    tables_dir.mkdir(exist_ok=True, parents=True)

    col_labels = ["ARG1", "PCA1", "ARG2", "PCA2"]
    mean_normal = np.zeros((len(p_list), len(col_labels)))
    std_normal  = np.zeros_like(mean_normal)
    mean_t      = np.zeros_like(mean_normal)
    std_t       = np.zeros_like(mean_normal)

    master_rng = np.random.default_rng(master_seed)

    for pi, p in enumerate(tqdm(p_list, desc="multi: p sweep", leave=False)):
        Sigma, U_m = sigma_multi_spike(p=p, coef=sigma_coef)
        E = generate_basis(p)
        mu = np.zeros(p)

        A = np.array([
            [0.5, 0.5, 0.5, 0.5],
            [1/np.sqrt(2), 0.0, -1/np.sqrt(2), 0.0],
        ])
        V = generate_reference_vectors(E, A)  # (p,2)

        angle_normal = np.zeros((n_trials, 4))  # [ARG1, PCA1, ARG2, PCA2]
        angle_t      = np.zeros((n_trials, 4))

        for i in tqdm(range(n_trials), desc=f"multi: trials (p={p})", leave=False):
            seed_n = int(master_rng.integers(0, 2**31 - 1))
            seed_t = int(master_rng.integers(0, 2**31 - 1))

            Xn = sample_normal(Sigma=Sigma, n=n, mu=mu, seed=seed_n)
            Xt = sample_t(Sigma=Sigma, nu=nu, n=n, mu=mu, seed=seed_t)

            # Normal
            U_ARG = compute_ARG_PC_subspace(Xn, V, spike_number=2, orthonormal=True)
            U_PCA = compute_PC_subspace(Xn, spike_number=2)
            th_ARG = compute_principal_angles(U_ARG, U_m)
            th_PCA = compute_principal_angles(U_PCA, U_m)
            angle_normal[i, :] = [th_ARG[0], th_PCA[0], th_ARG[1], th_PCA[1]]

            # t
            U_ARG_t = compute_ARG_PC_subspace(Xt, V, spike_number=2, orthonormal=True)
            U_PCA_t = compute_PC_subspace(Xt, spike_number=2)
            th_ARG_t = compute_principal_angles(U_ARG_t, U_m)
            th_PCA_t = compute_principal_angles(U_PCA_t, U_m)
            angle_t[i, :] = [th_ARG_t[0], th_PCA_t[0], th_ARG_t[1], th_PCA_t[1]]

        mean_normal[pi, :] = angle_normal.mean(axis=0)
        mean_t[pi, :]      = angle_t.mean(axis=0)
        std_normal[pi, :]  = angle_normal.std(axis=0, ddof=0)
        std_t[pi, :]       = angle_t.std(axis=0, ddof=0)

    # Save summaries to results/tables
    idx = p_list
    pd.DataFrame(mean_normal, index=idx, columns=col_labels).to_csv(tables_dir / "multi_normal_mean.csv")
    pd.DataFrame(std_normal,  index=idx, columns=col_labels).to_csv(tables_dir / "multi_normal_std.csv")
    pd.DataFrame(mean_t,      index=idx, columns=col_labels).to_csv(tables_dir / "multi_t_mean.csv")
    pd.DataFrame(std_t,       index=idx, columns=col_labels).to_csv(tables_dir / "multi_t_std.csv")

    print("✅ Multi simulation complete. Saved in results/tables")


# ------------------------------------------------------------
# Convergence-rate simulation for Theorem 4
# ------------------------------------------------------------
def run_simulation_convergence_rate(
    p_list: List[int],
    n: int,
    n_trials: int,
    powers: List[float],
    snr_list: List[float],
    master_seed: int = 725,
) -> None:
    """
    Estimate convergence-rate curves for the inner product u1ᵀ d1 for varying SNR levels
    under single-spike, single-reference case, and write normalized summary tables to disk.

    For each SNR in `snr_list`, set Σ = c1·p·e1e1ᵀ + c2·I_p with (c1, c2) = (1, 1/SNR),
    construct a single reference vector v = (e1 + e2)/√2, compute the (normalized) first
    negative-ridge discriminant d1 for each dataset, and record
        M_α(p) = p^α · |u1ᵀ d1|
    averaged over trials. Each α-curve is then normalized by its value at the
    **first p in `p_list`**.

    Parameters
    ----------
    p_list : List[int]
        List of dimensions p to sweep over (e.g., [100, 200, 500, ...]).
        The first element p_list[0] is used as the normalization baseline.
    n : int
        Sample size per trial.
    n_trials : int
        Number of Monte Carlo trials at each (SNR, p) point.
    powers : List[float]
        Exponents α to evaluate in M_α(p). Columns in the output are labeled "alpha={α}".
    snr_list : List[float]
        Positive SNR values (SNR > 0). For each SNR, we use (c1, c2) = (1, 1/SNR).
    master_seed : int, default=725
        Master seed for reproducibility. A NumPy RNG is initialized with this seed,
        and per-trial seeds are drawn from it when sampling datasets.

    Procedure
    ---------
    For each SNR in `snr_list`:
      1) For each p in `p_list`:
         - Build Σ = c1·p·e1e1ᵀ + c2·I_p and set u1 = e1.
         - Form a single reference vector v = (e1 + e2)/√2 via `generate_reference_vectors`.
         - Repeat for `n_trials`:
             · Sample X ~ N(0, Σ) of shape (n, p).
             · Compute the (normalized) negative-ridge discriminant d1 (spike_number=1).
             · Compute |u1ᵀ d1| and then p^α·|u1ᵀ d1| for each α in `powers`.
         - Aggregate across trials to get the mean for each α at this p.
      2) Normalize each α-curve by its value at p_list[0].

    Outputs (files)
    ---------------
    For each SNR value, write one CSV to `results/tables/`:
      - convergence_rate_snr{SNR}.csv
        * Index: p values from `p_list`
        * Columns: "alpha={α}" for α in `powers`
        * Entries: normalized mean M_α(p) / M_α(p_list[0])

    Returns
    -------
    None
        Results are saved to disk; nothing is returned.
    """
    # --- paths ---
    base_dir = Path("results")
    tables_dir = base_dir / "tables"
    tables_dir.mkdir(exist_ok=True, parents=True)

    def _snr_tag(snr: float) -> str:
        # format like 0.001, 0.025, 0.5, 1 (no scientific notation)
        return f"{snr:g}"

    # Outer loop over SNR settings
    for snr in tqdm(snr_list, desc="convergence: SNR sweep", leave=False):
        if snr <= 0:
            raise ValueError(f"SNR must be positive; got {snr}")
        c1 = 1.0
        c2 = c1 / snr

        mean_mat = np.zeros((len(p_list), len(powers)))
        std_mat  = np.zeros_like(mean_mat)   # kept in case

        master_rng = np.random.default_rng(master_seed)

        for pi, p in enumerate(tqdm(p_list, desc=f"  p sweep (snr={_snr_tag(snr)})", leave=False)):
            # Σ = c1 * p * e1 e1^T + c2 I_p, u1 = e1
            Sigma, u1 = sigma_single_spike(p=p, coef=(c1, c2))
            if u1.ndim == 1:
                u1 = u1[:, None]  # (p,1)

            # v1 = (e1 + e2)/sqrt(2)
            E = generate_basis(p)
            A = np.array([[1/np.sqrt(2), 1/np.sqrt(2), 0.0, 0.0]])  # (1,4)
            V = generate_reference_vectors(E, A)                     # (p,1)

            mu = np.zeros(p)
            vals = np.zeros((n_trials, len(powers)))  # trial × power

            for t in range(n_trials):
                seed_t = int(master_rng.integers(0, 2**31 - 1))
                X = sample_normal(Sigma=Sigma, n=n, mu=mu, seed=seed_t)  # (n,p)

                # d1 (normalized)
                D = compute_negative_ridge_discriminants(
                    samples=X,
                    reference_vectors=V,
                    spike_number=1,
                    normalize=True,
                )  # (p,1)
                d1 = D[:, 0]

                # |u1^T d1|
                ip = float(np.abs(u1[:, 0].T @ d1))

                # p^α * |u1^T d1|
                for j, alpha in enumerate(powers):
                    vals[t, j] = (p ** alpha) * ip

            # Aggregate at fixed p
            mean_mat[pi, :] = vals.mean(axis=0)
            std_mat[pi, :]  = vals.std(axis=0, ddof=0)

        # Normalize using the first row (p=100) as baseline
        baseline = np.where(mean_mat[0, :] == 0.0, 1.0, mean_mat[0, :])
        mean_norm = mean_mat / baseline[None, :]

        # Save normalized table for this SNR
        col_labels = [f"alpha={a:g}" for a in powers]
        idx = p_list
        out_csv = tables_dir / f"convergence_rate_snr{_snr_tag(snr)}.csv"
        pd.DataFrame(mean_norm, index=idx, columns=col_labels).to_csv(out_csv)

    print("✅ Convergence-rate simulation complete. Saved in results/tables")


# ------------------------------------------------------------------
# Run simulation
# ------------------------------------------------------------------
if __name__ == "__main__":
    run_simulation_single(
        p_list=[100, 200, 500, 1000, 2000],
        a_list=[0.0, 0.5, 1/np.sqrt(2), np.sqrt(3)/2, 1.0],
        n=40,
        nu=5,
        n_trials=100,
        sigma_coef=(1.0, 40.0),
        master_seed=725,
    )

    run_simulation_multi(
        p_list=[100, 200, 500, 1000, 2000],
        n=40,
        nu=5,
        n_trials=100,
        sigma_coef=(2.0, 1.0, 40.0),
        master_seed=725,
    )

    run_simulation_convergence_rate(
        p_list=[100, 200, 500, 1000, 2000],
        n=40,
        n_trials=100,
        powers=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        snr_list=[0.01, 0.025, 0.05, 0.1, 0.25, 0.5],
        master_seed=725,
    )