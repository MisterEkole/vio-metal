#!/usr/bin/env python3
"""Plot optimizer cost function values over time."""

import pandas as pd
import matplotlib.pyplot as plt
import sys
import os

def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "results/cost_log.csv"
    if not os.path.exists(path):
        print(f"Cost log not found: {path}")
        sys.exit(1)

    # Derive output path: cost_log_XXX.csv -> cost_plot_XXX.png (same directory)
    dirname = os.path.dirname(path)
    basename = os.path.basename(path)
    out_name = basename.replace("cost_log", "cost_plot").replace(".csv", ".png")
    out_path = os.path.join(dirname, out_name) if dirname else out_name

    df = pd.read_csv(path)
    t0 = df["timestamp"].iloc[0]
    df["time_s"] = (df["timestamp"] - t0) * 1e-9

    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

    # Cost evolution
    ax = axes[0]
    ax.semilogy(df["time_s"], df["initial_cost"], label="Initial cost", alpha=0.7)
    ax.semilogy(df["time_s"], df["final_cost"], label="Final cost", alpha=0.7)
    ax.set_ylabel("Cost (log)")
    ax.set_title("Optimizer Cost per Keyframe")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Cost reduction ratio
    ax = axes[1]
    ratio = df["final_cost"] / df["initial_cost"].clip(lower=1e-12)
    ax.plot(df["time_s"], ratio, color="green", alpha=0.7)
    ax.axhline(y=1.0, color="red", linestyle="--", alpha=0.5)
    ax.set_ylabel("Final / Initial")
    ax.set_title("Cost Reduction Ratio (< 1 = improvement)")
    ax.grid(True, alpha=0.3)

    # Iterations and convergence
    ax = axes[2]
    colors = ["green" if c else "red" for c in df["converged"]]
    ax.bar(df["time_s"], df["iterations"], width=0.3, color=colors, alpha=0.7)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Iterations")
    ax.set_title("Solver Iterations (green=converged, red=not)")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    print(f"Saved: {out_path}")
    if os.environ.get("DISPLAY") or sys.platform == "darwin":
        plt.show()

if __name__ == "__main__":
    main()
