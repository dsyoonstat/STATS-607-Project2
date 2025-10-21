# plot.py
# - Single-spike, single-reference case: trend over a^2
# - Multi-spike, multi-reference case: ARG/PCA × θ1/θ2
# - Convergence: normalized p^α |u1ᵀ d1| curves

from pathlib import Path
import re
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt


# ---------------------- aesthetics ----------------------
def _set_pub_style():
    """Set matplotlib parameters for a clean, publication-quality style."""
    mpl.rcParams.update({
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "font.size": 10.5,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "legend.fontsize": 10,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "axes.grid": True,
        "grid.linestyle": "--",
        "grid.linewidth": 0.5,
        "grid.alpha": 0.3,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.8,
        "figure.constrained_layout.use": True,
    })


def _ensure_dirs():
    """Ensure expected folders exist."""
    base = Path("results")
    (base / "tables").mkdir(exist_ok=True, parents=True)
    (base / "figures").mkdir(exist_ok=True, parents=True)


def _save(fig: mpl.figure.Figure, stem: str):
    """Save figure as PDF and SVG into results/figures."""
    out_pdf = Path("results/figures") / f"{stem}.pdf"
    out_svg = Path("results/figures") / f"{stem}.svg"
    fig.savefig(out_pdf)
    fig.savefig(out_svg)
    print(f"Saved: {out_pdf}\n       {out_svg}")


# ---------------------- helpers ----------------------
def _parse_a2_columns(df: pd.DataFrame):
    """
    Extract a^2 columns from table columns like:
      'baseline', 'a^2=0.25', 'a^2=0.5', 'a^2=0.75', 'a^2=1'
    Returns (xs, cols) sorted by a^2 ascending.
    """
    vals, cols = [], []
    for c in df.columns:
        m = re.match(r"a\^2=([\d\.eE+-]+)$", c)
        if m:
            vals.append(float(m.group(1)))
            cols.append(c)
    if not vals:
        raise ValueError("No a^2 columns found (expected 'a^2=0.25', etc).")
    idx = np.argsort(vals)
    xs = np.asarray(vals, float)[idx]
    cols = [cols[i] for i in idx]
    return xs, cols


# ---------------------- single: normal / t ----------------------
def make_single_plots(dist: str):
    """
    dist ∈ {'normal','t'}
    Inputs:
      results/tables/single_{dist}_mean.csv
      results/tables/single_{dist}_std.csv   (read but not used in the current figure)
    Output:
      results/figures/single_{dist}_trend.(pdf|svg)
    """
    from matplotlib.lines import Line2D

    mean_path = Path(f"results/tables/single_{dist}_mean.csv")
    std_path  = Path(f"results/tables/single_{dist}_std.csv")
    if not (mean_path.exists() and std_path.exists()):
        raise FileNotFoundError(f"Missing single-{dist} tables: {mean_path}, {std_path}")

    df_mean = pd.read_csv(mean_path, index_col=0)
    _ = pd.read_csv(std_path,  index_col=0)  # kept for completeness; not plotted

    # x = a^2 (exclude 'baseline'); ticks exactly match the data
    a2_x, a2_cols = _parse_a2_columns(df_mean)
    x_min, x_max = float(np.min(a2_x)), float(np.max(a2_x))

    # Trend plot: lines + 'o' markers only (no std shading)
    fig, ax = plt.subplots(figsize=(6.2, 3.9))

    # draw ARG curves and matching-color PCA baselines (horizontal dotted lines)
    for p_str, row in df_mean.iterrows():
        p = int(p_str)
        y_arg = row[a2_cols].to_numpy(float)
        # 1) ARG curve (colored by cycle)
        (line,) = ax.plot(a2_x, y_arg, marker="o", linewidth=1.9, label=f"p={p}")
        color = line.get_color()
        # 2) PCA baseline = row["baseline"] as horizontal dotted line with same color
        y_pca = float(row["baseline"])
        ax.hlines(y_pca, x_min, x_max, colors=color, linestyles=":", linewidth=1.6, alpha=0.9, zorder=1)

    ax.set_xlabel(r"$a^2$")
    ax.set_ylabel("Mean angle (radians)")
    ax.set_title(f"Single-spike, single-reference ({dist}) — trend over $a^2$")
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(bottom=0)

    # Ticks exactly at available a^2 values (e.g., 0, 0.25, 0.5, 0.75, 1)
    ax.set_xticks(a2_x)
    ax.set_xticklabels([f"{t:.2g}" for t in a2_x])

    # Legend: keep p-curves + add a proxy handle describing PCA baseline (dotted line)
    handles, labels = ax.get_legend_handles_labels()
    baseline_proxy = Line2D([], [], color="k", linestyle=":", linewidth=1.6, label="PCA baseline")
    handles.append(baseline_proxy)
    labels.append("PCA baseline")
    ax.legend(handles=handles, labels=labels, ncol=2, loc="lower left", frameon=False)

    _save(fig, f"single_{dist}_trend")
    plt.close(fig)


# ---------------------- multi: normal / t ----------------------
def make_multi_plots(dist: str):
    """
    dist ∈ {'normal','t'}
    Inputs:
      results/tables/multi_{dist}_mean.csv
      results/tables/multi_{dist}_std.csv
    Output:
      results/figures/multi_{dist}_angles.(pdf|svg)
    Spec:
      - One figure combining θ1/θ2 across methods:
          * colors: θ1 = first, θ2 = second
          * linestyles: ARG = solid, PCA = dashed
      - Log x-axis with fixed ticks [100, 200, 500, 1000, 2000]; no minor ticks.
    """
    from matplotlib.ticker import FixedLocator, ScalarFormatter, NullLocator
    from matplotlib.lines import Line2D

    mean_path = Path(f"results/tables/multi_{dist}_mean.csv")
    std_path  = Path(f"results/tables/multi_{dist}_std.csv")
    if not (mean_path.exists() and std_path.exists()):
        raise FileNotFoundError(f"Missing multi-{dist} tables: {mean_path}, {std_path}")

    df_mean = pd.read_csv(mean_path, index_col=0)
    _ = pd.read_csv(std_path, index_col=0)

    # x values and fixed major ticks
    p_vals = df_mean.index.astype(int).to_numpy()
    p_ticks = [100, 200, 500, 1000, 2000]

    # Data series
    y_ARG1 = df_mean["ARG1"].to_numpy(float)
    y_PCA1 = df_mean["PCA1"].to_numpy(float)
    y_ARG2 = df_mean["ARG2"].to_numpy(float)
    y_PCA2 = df_mean["PCA2"].to_numpy(float)

    # Colors: θ1 = first, θ2 = second
    default_colors = mpl.rcParams["axes.prop_cycle"].by_key().get("color", ["#1f77b4", "#ff7f0e"])
    c_th1, c_th2 = default_colors[0], default_colors[1] if len(default_colors) > 1 else "#ff7f0e"

    fig, ax = plt.subplots(figsize=(6.8, 4.0))

    # θ1: solid for ARG, dashed for PCA
    ax.plot(p_vals, y_ARG1, color=c_th1, linestyle="-",  marker="o", linewidth=1.9, label=r"$\theta_1$ (ARG)")
    ax.plot(p_vals, y_PCA1, color=c_th1, linestyle="--", marker="o", linewidth=1.9, label=r"$\theta_1$ (PCA)")

    # θ2: solid for ARG, dashed for PCA
    ax.plot(p_vals, y_ARG2, color=c_th2, linestyle="-",  marker="o", linewidth=1.9, label=r"$\theta_2$ (ARG)")
    ax.plot(p_vals, y_PCA2, color=c_th2, linestyle="--", marker="o", linewidth=1.9, label=r"$\theta_2$ (PCA)")

    # Log x-axis with fixed ticks, no minor ticks
    ax.set_xscale("log", base=10)
    ax.xaxis.set_major_locator(FixedLocator(p_ticks))
    ax.xaxis.set_major_formatter(ScalarFormatter())
    ax.xaxis.set_minor_locator(NullLocator())
    ax.minorticks_off()

    ax.grid(axis="y", which="major", linestyle="--", linewidth=0.5, alpha=0.3)
    ax.grid(axis="x", which="both", visible=False)

    ax.set_xlabel(r"Dimension $p$ (log scale)")
    ax.set_ylabel(r"Mean principal angle (radians)")
    ax.set_title(f"Two-spike, two-reference ({dist}) — θ₁ / θ₂ across methods")
    ax.set_ylim(bottom=0)

    # Custom legend grouped by theta and method
    handles = [
        Line2D([], [], color=c_th1, linestyle="-",  marker="o", linewidth=2.2, label=r"$\theta_1$ ARG"),
        Line2D([], [], color=c_th1, linestyle="--", marker="o", linewidth=2.2, label=r"$\theta_1$ PCA"),
        Line2D([], [], color=c_th2, linestyle="-",  marker="o", linewidth=2.2, label=r"$\theta_2$ ARG"),
        Line2D([], [], color=c_th2, linestyle="--", marker="o", linewidth=2.2, label=r"$\theta_2$ PCA"),
    ]
    ax.legend(handles=handles,
              loc="lower left",
              frameon=False,
              ncol=2,
              handlelength=2.8,
              handletextpad=0.8,
              columnspacing=1.8)

    _save(fig, f"multi_{dist}_angles")
    plt.close(fig)


# ---------------------- convergence (Theorem 4) ----------------------
def make_convergence_plots():
    """
    Inputs:
      results/tables/convergence_rate_snr*.csv
        - rows: p in {100, 200, 500, 1000, 2000}
        - cols: alpha={value} (normalized by p=100)

    Output:
      results/figures/convergence_rate_all.pdf / .svg

    Spec:
      - One figure with multiple subplots (2×3 grid)
      - Each subplot: one SNR value, showing normalized curves over p
      - Log x-axis with ticks [100, 200, 500, 1000, 2000]; no minor ticks.
      - 'o' markers; one color per alpha.
    """
    from matplotlib.ticker import FixedLocator, ScalarFormatter, NullLocator
    import glob
    import re

    tables_dir = Path("results/tables")
    plot_dir   = Path("results/figures")
    plot_dir.mkdir(parents=True, exist_ok=True)

    paths = sorted(glob.glob(str(tables_dir / "convergence_rate_snr*.csv")))
    if not paths:
        raise FileNotFoundError(f"No convergence_rate_snr*.csv found in {tables_dir}")

    snr_re = re.compile(r"convergence_rate_snr([0-9\.eE+-]+)\.csv$")
    p_ticks = [100, 200, 500, 1000, 2000]

    # --- figure layout: 2×3 grid (adjust if different number of SNRs) ---
    n_files = len(paths)
    n_rows, n_cols = 2, 3
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 7))
    axes = axes.flatten()

    for i, path in enumerate(paths):
        if i >= len(axes):
            break  # in case more than 6 SNRs
        ax = axes[i]

        m = snr_re.search(path)
        snr_tag = m.group(1) if m else "?"

        df_norm = pd.read_csv(path, index_col=0)
        p_vals = df_norm.index.astype(int).to_numpy()

        # Plot each α curve
        for col in df_norm.columns:
            y = df_norm[col].to_numpy(float)
            ax.plot(p_vals, y, marker="o", markersize = 4, linewidth=1.2, label=col.replace("alpha=", r"$\alpha$="))

        # Log x-axis
        ax.set_xscale("log", base=10)
        ax.xaxis.set_major_locator(FixedLocator(p_ticks))
        ax.xaxis.set_major_formatter(ScalarFormatter())
        ax.xaxis.set_minor_locator(NullLocator())
        ax.minorticks_off()

        # Style
        ax.set_xlim(min(p_ticks), max(p_ticks))
        ax.set_ylim(bottom=0)
        ax.set_title(f"SNR = {snr_tag}")
        ax.grid(axis="y", which="major", linestyle="--", linewidth=0.5, alpha=0.3)
        ax.grid(axis="x", which="both", visible=False)

        if i // n_cols == n_rows - 1:
            ax.set_xlabel(r"Dimension $p$ (log scale)")
        if i % n_cols == 0:
            ax.set_ylabel(r"Normalized $p^{\alpha}|u_1^\top d_1|$")

    # Remove unused subplots (if any)
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    # Shared legend below all subplots
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=10, frameon=False, fontsize=9)
    fig.suptitle(r"Convergence-rate behavior across SNR values", fontsize=13)
    fig.tight_layout(rect=[0, 0.05, 1, 0.97])

    _save(fig, "convergence_rate")
    plt.close(fig)


# ---------------------- main ----------------------
def main():
    _ensure_dirs()
    _set_pub_style()

    # Single (normal / t)
    for dist in ("normal", "t"):
        make_single_plots(dist)

    # Multi (normal / t)
    for dist in ("normal", "t"):
        make_multi_plots(dist)

    # Convergence (Theorem 4)
    make_convergence_plots()

if __name__ == "__main__":
    main()