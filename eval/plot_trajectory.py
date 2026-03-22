"""Plot estimated vs ground truth trajectories as 3D point clouds.

Generates two figures:
  1. Full-scale 3D view showing divergence extent
  2. Zoomed 2D projections (XY, XZ, YZ) around the GT bounding box
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt


def load_tum(path):
    """Load TUM-format trajectory: timestamp tx ty tz qx qy qz qw"""
    data = np.loadtxt(path)
    return data[:, 1:4]  # xyz only


def main():
    parser = argparse.ArgumentParser(description="Plot trajectory point clouds against ground truth")
    parser.add_argument("--gt", required=True, help="Ground truth TUM file")
    parser.add_argument("--est", required=True, nargs="+", help="Estimated TUM file(s)")
    parser.add_argument("--labels", nargs="+", help="Labels for estimated trajectories")
    parser.add_argument("--subsample", type=int, default=1, help="Plot every Nth point (default: 1)")
    parser.add_argument("--output", help="Save plot prefix (generates _full.png and _zoom.png)")
    args = parser.parse_args()

    labels = args.labels or [f"est_{i}" for i in range(len(args.est))]
    if len(labels) != len(args.est):
        parser.error("Number of labels must match number of estimated trajectories")

    gt = load_tum(args.gt)
    estimates = [(label, load_tum(path)) for label, path in zip(labels, args.est)]

    s = args.subsample
    est_colors = ["#2ecc71", "#3498db", "#e67e22", "#9b59b6"]
    gt_color = "#e74c3c"

    # --- Figure 1: Full-scale 3D ---
    fig1 = plt.figure(figsize=(14, 10))
    ax = fig1.add_subplot(111, projection="3d")
    ax.scatter(gt[::s, 0], gt[::s, 1], gt[::s, 2],
               c=gt_color, s=2, alpha=0.5, label="ground truth")
    for i, (label, est) in enumerate(estimates):
        ax.scatter(est[::s, 0], est[::s, 1], est[::s, 2],
                   c=est_colors[i % len(est_colors)], s=1, alpha=0.3, label=label)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_zlabel("z (m)")
    ax.legend(markerscale=5)
    ax.set_title("Full-Scale Trajectory (shows divergence extent)")
    plt.tight_layout()

    # --- Figure 2: Zoomed 2D projections around GT extent ---
    gt_min = gt.min(axis=0)
    gt_max = gt.max(axis=0)
    gt_range = (gt_max - gt_min).max()
    margin = gt_range * 0.5
    xlim = (gt_min[0] - margin, gt_max[0] + margin)
    ylim = (gt_min[1] - margin, gt_max[1] + margin)
    zlim = (gt_min[2] - margin, gt_max[2] + margin)

    proj = [("x", "y", 0, 1, xlim, ylim),
            ("x", "z", 0, 2, xlim, zlim),
            ("y", "z", 1, 2, ylim, zlim)]

    fig2, axes = plt.subplots(1, 3, figsize=(20, 6))
    for ax, (xlab, ylab, xi, yi, xl, yl) in zip(axes, proj):
        ax.scatter(gt[::s, xi], gt[::s, yi],
                   c=gt_color, s=3, alpha=0.6, label="ground truth", zorder=3)
        for i, (label, est) in enumerate(estimates):
            # clip to zoom region so scatter doesn't distort scale
            mask = ((est[:, xi] >= xl[0]) & (est[:, xi] <= xl[1]) &
                    (est[:, yi] >= yl[0]) & (est[:, yi] <= yl[1]))
            clipped = est[mask]
            ax.scatter(clipped[::s, xi], clipped[::s, yi],
                       c=est_colors[i % len(est_colors)], s=2, alpha=0.4,
                       label=label, zorder=2)
        ax.set_xlabel(f"{xlab} (m)")
        ax.set_ylabel(f"{ylab} (m)")
        ax.set_xlim(xl)
        ax.set_ylim(yl)
        ax.set_aspect("equal")
        ax.legend(markerscale=4, fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_title(f"{xlab.upper()}-{ylab.upper()} projection")

    fig2.suptitle("Zoomed to Ground Truth Region", fontsize=13)
    plt.tight_layout()

    # show interactive window, save on close
    plt.show()

    if args.output:
        prefix = args.output.replace(".png", "")
        fig1.savefig(f"{prefix}_full.png", dpi=150)
        fig2.savefig(f"{prefix}_zoom.png", dpi=150)
        print(f"Saved: {prefix}_full.png, {prefix}_zoom.png")


if __name__ == "__main__":
    main()
