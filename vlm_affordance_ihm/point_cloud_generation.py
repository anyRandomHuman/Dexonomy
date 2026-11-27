"""Sample a point cloud from a Dexonomy object scene or an OBJ mesh.

Examples (from the repository root):
  python vlm_affordance_ihm/point_cloud_generation.py \
      --object assets/object/DGN_5k/scene_cfg/ddg_gd_screwdriver_poisson_006 \
      --scale scale017 --output output/screwdriver.ply
  python vlm_affordance_ihm/point_cloud_generation.py \
      --object assets/object/DGN_5k/processed_data/ddg_gd_screwdriver_poisson_006/mesh/simplified.obj \
      --output output/screwdriver_unscaled.npy

Scene files apply their object scale and pose. Direct OBJ input uses mesh coordinates.
"""

import argparse
from pathlib import Path

import numpy as np
import trimesh
from scipy.spatial.transform import Rotation


def resolve_scene(directory: Path, scale: str | None) -> Path:
    floating = directory / "floating"
    search_dir = floating if floating.is_dir() else directory
    scenes = sorted(search_dir.glob("*.npy"))
    if scale:
        name = scale if scale.startswith("scale") else f"scale{scale}"
        scene = search_dir / f"{name.removesuffix('.npy')}.npy"
        if not scene.is_file():
            raise ValueError(f"Scene scale {scale!r} not found in {search_dir}")
        return scene
    if len(scenes) != 1:
        raise ValueError(
            f"Found {len(scenes)} scene files in {search_dir}; "
            "pass --scale to select one"
        )
    return scenes[0]


def load_object_mesh(object_path: Path, scale: str | None) -> trimesh.Trimesh:
    if object_path.is_dir():
        object_path = resolve_scene(object_path, scale)
    elif scale:
        raise ValueError("--scale is only used with a scene directory")

    if object_path.suffix.lower() == ".obj":
        return trimesh.load(object_path, force="mesh")
    if object_path.suffix.lower() != ".npy":
        raise ValueError("--object must be a scene .npy, an OBJ mesh, or a scene directory")

    scene = np.load(object_path, allow_pickle=True).item()
    object_name = scene["task"]["obj_name"]
    obj = scene["scene"][object_name]
    if obj["type"] != "rigid_object":
        raise ValueError(f"Expected a rigid_object, got {obj['type']!r}")

    mesh_path = (object_path.parent / obj["file_path"]).resolve()
    mesh = trimesh.load(mesh_path, force="mesh")
    scale_xyz = np.asarray(obj["scale"], dtype=float)
    pose = np.asarray(obj["pose"], dtype=float)  # xyz translation, wxyz quaternion
    mesh.vertices = Rotation.from_quat(pose[[4, 5, 6, 3]]).apply(
        mesh.vertices * scale_xyz
    ) + pose[:3]
    return mesh


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--object", type=Path, required=True,
                        help="Scene .npy, OBJ mesh, or scene directory")
    parser.add_argument("--output", type=Path, required=True,
                        help="Output point cloud (.ply or .npy)")
    parser.add_argument("--scale", help="Scene filename such as scale017 when --object is a directory")
    parser.add_argument("--points", type=int, default=10_000,
                        help="Number of surface points (default: 10000)")
    parser.add_argument("--seed", type=int, default=0,
                        help="Random seed for sampling (default: 0)")
    args = parser.parse_args()

    if not args.object.exists():
        parser.error(f"Object not found: {args.object}")
    if args.points < 1:
        parser.error("--points must be positive")
    if args.output.suffix.lower() not in (".ply", ".npy"):
        parser.error("--output must end in .ply or .npy")

    try:
        mesh = load_object_mesh(args.object.resolve(), args.scale)
        points, _ = trimesh.sample.sample_surface(mesh, args.points, seed=args.seed)
    except (ValueError, KeyError, OSError) as exc:
        parser.error(str(exc))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.suffix.lower() == ".npy":
        np.save(args.output, points)
    else:
        trimesh.points.PointCloud(points).export(args.output)
    print(f"Saved {len(points)} points to {args.output.resolve()}")


if __name__ == "__main__":
    main()
