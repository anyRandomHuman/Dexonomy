"""Render six axis-aligned views of a Dexonomy object, without a hand.

Run from the Dexonomy repository root, for example:
  MUJOCO_GL=egl python vlm_affordance_ihm/render_object_views.py \
      --scene assets/object/custom/scene_cfg/PuzzlePiece_B/floating/scale100.npy \
      --output output/puzzle_piece_views

The scene can also be an absolute path to a converted object bundle. Only the
active rigid object is loaded; no hand, grasp, or other scene objects are needed.
The output directory also receives camera_params.json, keyed by image stem, with
intrinsics and camera-to-world matrices in the OpenCV convention (+Z forward,
+Y down) used by AffordDex/NAA/point_utils.py.

The default views are left (-X), right (+X), front (-Y), back (+Y), top (+Z),
and bottom (-Z), all looking toward the object. Use --view-mode angles for the
previous azimuth/elevation grid.
"""

import argparse
import json
from pathlib import Path

import imageio.v3 as iio
import mujoco
import numpy as np

from dexonomy.util.file_util import load_scene_cfg


AXIS_VIEWS = (
    ("front", 90, 0),
    ("back", 270, 0),
    ("left", 0, 0),
    ("right", 180, 0),
    ("top", 0, -90),
    ("bottom", 0, 90),
)


def camera_parameters(scene: mujoco.MjvScene, width: int, height: int) -> dict:
    """Return AffordDex-compatible intrinsics and c2w for a rendered view."""
    camera = mujoco.mjv_averageCamera(scene.camera[0], scene.camera[1])
    forward = np.asarray(camera.forward, dtype=float)
    forward /= np.linalg.norm(forward)
    up = np.asarray(camera.up, dtype=float)
    right = np.cross(forward, up)
    right /= np.linalg.norm(right)
    down = np.cross(forward, right)

    c2w = np.eye(4)
    c2w[:3, :3] = np.column_stack((right, down, forward))
    c2w[:3, 3] = camera.pos

    focal = height / mujoco.mjv_frustumHeight(scene)
    intrinsics = np.array([
        [focal, 0.0, width / 2],
        [0.0, focal, height / 2],
        [0.0, 0.0, 1.0],
    ])
    return {"intrinsics": intrinsics.tolist(), "c2w": c2w.tolist()}


def load_object_model(scene_path: Path) -> mujoco.MjModel:
    scene = load_scene_cfg(str(scene_path))
    name = scene["task"]["obj_name"]
    obj = scene["scene"][name]
    if obj["type"] != "rigid_object":
        raise ValueError(f"Expected a rigid_object, got {obj['type']!r}")

    spec = mujoco.MjSpec.from_file(str(Path(obj["xml_path"]).resolve()))
    scale = np.asarray(obj["scale"], dtype=float)
    for mesh in spec.meshes:
        mesh.scale[:] *= scale

    # The converted PuzzlePiece_B MJCF has one root body named "object".
    pose = np.asarray(obj["pose"], dtype=float)
    body = spec.body("object")
    body.pos[:] = pose[:3]
    body.quat[:] = pose[3:]

    spec.add_texture(
        type=mujoco.mjtTexture.mjTEXTURE_SKYBOX,
        builtin=mujoco.mjtBuiltin.mjBUILTIN_GRADIENT,
        rgb1=[0.75, 0.82, 0.9],
        rgb2=[0.88, 0.92, 0.96],
        width=512,
        height=512,
    )
    spec.worldbody.add_light(
        pos=[0.2, -0.5, 1.0], diffuse=[0.45, 0.45, 0.45], castshadow=False
    )
    model = spec.compile()
    model.vis.headlight.ambient[:] = [0.2, 0.2, 0.2]
    model.vis.headlight.diffuse[:] = [0.45, 0.45, 0.45]
    return model


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", type=Path, required=True, help="Dexonomy scene .npy file")
    parser.add_argument("--output", type=Path, required=True, help="Directory for PNG images")
    parser.add_argument("--camera-json", type=Path,
                        help="Camera JSON path (default: <output>/camera_params.json)")
    parser.add_argument("--width", type=int, default=200)
    parser.add_argument("--height", type=int, default=200)
    parser.add_argument("--distance", type=float, help="Camera distance in metres (auto by default)")
    parser.add_argument("--view-mode", choices=("axes", "angles"), default="axes",
                        help="Six axis views (default) or an azimuth/elevation grid")
    parser.add_argument("--azimuths", type=float, nargs="+",
                        help="Azimuth angles in degrees for --view-mode angles")
    parser.add_argument("--elevations", type=float, nargs="+",
                        help="Elevation angles in degrees for --view-mode angles")
    args = parser.parse_args()

    scene_path = args.scene.resolve()
    if not scene_path.is_file():
        parser.error(f"Scene not found: {scene_path}")
    if args.width < 1 or args.height < 1:
        parser.error("Image dimensions must be positive")
    if args.distance is not None and args.distance <= 0:
        parser.error("Camera distance must be positive")
    if args.view_mode == "axes":
        if args.azimuths is not None or args.elevations is not None:
            parser.error("--azimuths and --elevations require --view-mode angles")
        views = AXIS_VIEWS
    else:
        azimuths = args.azimuths or (0, 90, 180, 270)
        elevations = args.elevations or (-25, 25)
        views = tuple((f"az_{azimuth:g}_el_{elevation:g}", azimuth, elevation)
                      for elevation in elevations for azimuth in azimuths)

    model = load_object_model(scene_path)
    model.vis.global_.offwidth = max(model.vis.global_.offwidth, args.width)
    model.vis.global_.offheight = max(model.vis.global_.offheight, args.height)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    option = mujoco.MjvOption()
    option.geomgroup[3] = 0  # Show the visual mesh, not the collision hulls.
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.lookat[:] = model.stat.center
    camera.distance = args.distance or max(0.1, model.stat.extent)

    args.output.mkdir(parents=True, exist_ok=True)
    camera_json = args.camera_json or args.output / "camera_params.json"
    camera_json.parent.mkdir(parents=True, exist_ok=True)
    camera_params = {}
    with mujoco.Renderer(model, height=args.height, width=args.width) as renderer:
        for name, azimuth, elevation in views:
            camera.azimuth = azimuth
            camera.elevation = elevation
            renderer.update_scene(data, camera=camera, scene_option=option)
            destination = args.output / f"{name}.png"
            camera_params[name] = camera_parameters(
                renderer.scene, args.width, args.height
            )
            iio.imwrite(destination, renderer.render().copy())
            print(destination)
    with camera_json.open("w") as file:
        json.dump(camera_params, file, indent=2)
    print(camera_json)


if __name__ == "__main__":
    main()
