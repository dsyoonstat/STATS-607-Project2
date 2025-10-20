from typing import List, Tuple, Dict, Any
import json
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm

# === import from the files ===
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
    Single-spike, single-reference simulation with both Normal and t sampling.
    Saves raw trial arrays (NPZ) to results/raw and summary CSVs (mean/std) to results/tables.
    """
    # --- paths ---
    base_dir = Path("results")
    raw_dir = base_dir / "raw"
    tables_dir = base_dir / "tables"
    raw_dir.mkdir(exist_ok=True, parents=True)
    tables_dir.mkdir(exist_ok=True, parents=True)

    # --- config dict (for reproducibility) ---
    config: Dict[str, Any] = {
        "p_list": p_list,
        "a_list": a_list,
        "n": n,
        "nu": nu,
        "n_trials": n_trials,
        "sigma_coef": list(sigma_coef),
        "master_seed": master_seed,
    }

    # Column labels
    col_labels = ["baseline"] + [f"a^2={a**2:.2g}" for a in a_list]

    mean_normal = np.zeros((len(p_list), len(col_labels)))
    mean_t      = np.zeros_like(mean_normal)
    std_normal  = np.zeros_like(mean_normal)
    std_t       = np.zeros_like(mean_normal)

    master_rng = np.random.default_rng(master_seed)

    for pi, p in enumerate(tqdm(p_list, desc="single: p sweep", leave=False)):
        Sigma, u1 = sigma_single_spike(p=p, coef=sigma_coef)
        if u1.ndim == 1:
            u1 = u1[:, None]  # (p,1)

        E = generate_basis(p)  # (p,4)
        mu = np.zeros(p)

        normal_trials, t_trials = [], []
        # build trials
        for _ in tqdm(range(n_trials), desc=f"single: build trials (p={p})", leave=False):
            seed_n = int(master_rng.integers(0, 2**31 - 1))
            seed_t = int(master_rng.integers(0, 2**31 - 1))
            normal_trials.append(sample_normal(Sigma=Sigma, n=n, mu=mu, seed=seed_n))
            t_trials.append(sample_t(Sigma=Sigma, nu=nu, n=n, mu=mu, seed=seed_t))

        baseline_normal_angles = np.zeros(n_trials)
        baseline_t_angles      = np.zeros(n_trials)
        arg_normal_all = np.zeros((len(a_list), n_trials))
        arg_t_all      = np.zeros((len(a_list), n_trials))

        # baseline
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
                U_ARG = compute_ARG_PC_subspace(normal_trials[i], V, spike_number=1, orthonormal=True)
                arg_normal_all[aj, i] = compute_principal_angles(u1, U_ARG)[0]
                U_ARG_t = compute_ARG_PC_subspace(t_trials[i], V, spike_number=1, orthonormal=True)
                arg_t_all[aj, i] = compute_principal_angles(u1, U_ARG_t)[0]

            mean_normal[pi, 1+aj] = arg_normal_all[aj].mean()
            mean_t[pi, 1+aj]      = arg_t_all[aj].mean()
            std_normal[pi, 1+aj]  = arg_normal_all[aj].std(ddof=0)
            std_t[pi, 1+aj]       = arg_t_all[aj].std(ddof=0)

        # raw save per p
        raw_normal = np.vstack([baseline_normal_angles[None, :], arg_normal_all])
        raw_t      = np.vstack([baseline_t_angles[None, :],      arg_t_all])
        np.savez_compressed(
            raw_dir / f"single_p{p}.npz",
            raw_normal=raw_normal,   # (1+len(a_list), n_trials)
            raw_t=raw_t,             # (1+len(a_list), n_trials)
            col_labels=np.array(col_labels, dtype=object),
            config=json.dumps(config),
        )

    # save summaries to results/tables
    idx = p_list
    pd.DataFrame(mean_normal, index=idx, columns=col_labels).to_csv(tables_dir / "single_normal_mean.csv")
    pd.DataFrame(std_normal,  index=idx, columns=col_labels).to_csv(tables_dir / "single_normal_std.csv")
    pd.DataFrame(mean_t,      index=idx, columns=col_labels).to_csv(tables_dir / "single_t_mean.csv")
    pd.DataFrame(std_t,       index=idx, columns=col_labels).to_csv(tables_dir / "single_t_std.csv")

    print("✅ Single simulation complete. Tables -> results/tables, raw -> results/raw")


# ------------------------------------------------------------
# Multi-spike, multi-reference simulation (Normal + t)
# ------------------------------------------------------------
def run_simulation_multi(
    p_list: List[int],
    n: int,
    nu: int,
    n_trials: int,
    sigma_coef: Tuple[float, float, float] = (2.0, 1.0, 40.0),
    master_seed: int = 725,
) -> None:
    """
    Multi-spike, multi-reference simulation.
    Saves raw trial arrays (NPZ) to results/raw and summary CSVs (mean/std) to results/tables.
    """
    base_dir = Path("results")
    raw_dir = base_dir / "raw"
    tables_dir = base_dir / "tables"
    raw_dir.mkdir(exist_ok=True, parents=True)
    tables_dir.mkdir(exist_ok=True, parents=True)

    config: Dict[str, Any] = {
        "p_list": p_list,
        "n": n,
        "nu": nu,
        "n_trials": n_trials,
        "sigma_coef": list(sigma_coef),
        "master_seed": master_seed,
    }

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

        # raw per p
        np.savez_compressed(
            raw_dir / f"multi_p{p}.npz",
            angle_normal=angle_normal,  # (n_trials, 4)
            angle_t=angle_t,            # (n_trials, 4)
            col_labels=np.array(col_labels, dtype=object),
            config=json.dumps(config),
        )

    # save summaries to results/tables
    idx = p_list
    pd.DataFrame(mean_normal, index=idx, columns=col_labels).to_csv(tables_dir / "multi_normal_mean.csv")
    pd.DataFrame(std_normal,  index=idx, columns=col_labels).to_csv(tables_dir / "multi_normal_std.csv")
    pd.DataFrame(mean_t,      index=idx, columns=col_labels).to_csv(tables_dir / "multi_t_mean.csv")
    pd.DataFrame(std_t,       index=idx, columns=col_labels).to_csv(tables_dir / "multi_t_std.csv")

    print("✅ Multi simulation complete. Tables -> results/tables, raw -> results/raw")


# ------------------------------------------------------------------
# Example direct execution (you can delete or keep for quick runs)
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