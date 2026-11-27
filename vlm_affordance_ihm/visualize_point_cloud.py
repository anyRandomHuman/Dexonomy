"""Visualize a point cloud saved by point_cloud_generation.py.

Example (from the repository root):
  python vlm_affordance_ihm/visualize_point_cloud.py \
      --input output/screwdriver.ply --output output/screwdriver.png

Add --show to open an interactive 3D window when a display is available.
"""

import argparse
from pathlib import Path

import numpy as np
import trimesh


def load_points(path: Path) -> np.ndarray:
    if path.suffix.lower() == ".npy":
        points = np.load(path, allow_pickle=False)
    elif path.suffix.lower() == ".ply":
        points = np.asarray(trimesh.load(path).vertices)
    else:
        raise ValueError("Input must be a .ply or .npy point cloud")

    if points.ndim != 2 or points.shape[1] != 3 or len(points) == 0:
        raise ValueError("Point cloud must be a nonempty array of shape (N, 3)")
    if not np.isfinite(points).all():
        raise ValueError("Point cloud contains nonfinite coordinates")
    return points


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True,
                        help="Point cloud file (.ply or .npy)")
    parser.add_argument("--output", type=Path,
                        help="PNG image path (default: input filename with .png suffix)")
    parser.add_argument("--show", action="store_true",
                        help="Open an interactive Matplotlib window")
    parser.add_argument("--max-points", type=int, default=50_000,
                        help="Maximum plotted points (default: 50000)")
    parser.add_argument("--point-size", type=float, default=1.0,
                        help="Marker area in points squared (default: 1)")
    parser.add_argument("--elevation", type=float, default=25.0)
    parser.add_argument("--azimuth", type=float, default=-60.0)
    parser.add_argument("--seed", type=int, default=0,
                        help="Seed for display subsampling (default: 0)")
    args = parser.parse_args()

    if not args.input.is_file():
        parser.error(f"Point cloud not found: {args.input}")
    if args.max_points < 1:
        parser.error("--max-points must be positive")
    if args.point_size <= 0:
        parser.error("--point-size must be positive")
    output = args.output or args.input.with_suffix(".png")
    if output.suffix.lower() != ".png":
        parser.error("--output must end in .png")

    try:
        points = load_points(args.input)
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
    if len(points) > args.max_points:
        indices = np.random.default_rng(args.seed).choice(
            len(points), args.max_points, replace=False
        )
        points = points[indices]

    import matplotlib

    if not args.show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(*points.T, c=points[:, 2], cmap="viridis",
               s=args.point_size, linewidths=0, depthshade=False)
    ax.set(xlabel="X", ylabel="Y", zlabel="Z",
           title=f"{args.input.stem} ({len(points):,} points shown)")
    center = (points.min(axis=0) + points.max(axis=0)) / 2
    half_range = np.maximum((points.max(axis=0) - points.min(axis=0)) / 2, 1e-6)
    for setter, mid, half in zip((ax.set_xlim, ax.set_ylim, ax.set_zlim),
                                 center, half_range):
        setter(mid - half * 1.05, mid + half * 1.05)
    ax.set_box_aspect(half_range)
    ax.view_init(elev=args.elevation, azim=args.azimuth)
    fig.tight_layout()

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=200)
    print(f"Saved {len(points)} plotted points to {output.resolve()}")
    if args.show:
        plt.show()
    plt.close(fig)


if __name__ == "__main__":
    main()
