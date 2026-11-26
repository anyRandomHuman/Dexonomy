"""Render the object in a Dexonomy scene from several camera angles, without a hand.

Run from the Dexonomy repository root, for example:
  MUJOCO_GL=egl python getting_started/render_object_views.py \
      --scene assets/object/custom/scene_cfg/PuzzlePiece_B/floating/scale100.npy \
      --output output/puzzle_piece_views

The scene can also be an absolute path to a converted object bundle. Only the
active rigid object is loaded; no hand, grasp, or other scene objects are needed.
"""

import argparse
from pathlib import Path

import imageio.v3 as iio
import mujoco
import numpy as np

from dexonomy.util.file_util import load_scene_cfg


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
    parser.add_argument("--width", type=int, default=200)
    parser.add_argument("--height", type=int, default=200)
    parser.add_argument("--distance", type=float, help="Camera distance in metres (auto by default)")
    parser.add_argument("--azimuths", type=float, nargs="+", default=(0, 90, 180, 270))
    parser.add_argument("--elevations", type=float, nargs="+", default=(-25, 25))
    args = parser.parse_args()

    scene_path = args.scene.resolve()
    if not scene_path.is_file():
        parser.error(f"Scene not found: {scene_path}")
    if args.width < 1 or args.height < 1:
        parser.error("Image dimensions must be positive")
    if args.distance is not None and args.distance <= 0:
        parser.error("Camera distance must be positive")

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
    with mujoco.Renderer(model, height=args.height, width=args.width) as renderer:
        for elevation in args.elevations:
            for azimuth in args.azimuths:
                camera.azimuth = azimuth
                camera.elevation = elevation
                renderer.update_scene(data, camera=camera, scene_option=option)
                destination = args.output / f"az_{azimuth:g}_el_{elevation:g}.png"
                iio.imwrite(destination, renderer.render().copy())
                print(destination)


if __name__ == "__main__":
    main()


"""
python vlm_affordance_ihm/render_object_views.py --scene assets/object/DGN_5k/scene_cfg/   --output output/

"""