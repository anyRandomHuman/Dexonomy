"""Render Dexonomy-generated grasp .npy files to PNGs with MuJoCo.

Run this from the Dexonomy repository root, for example:
  MUJOCO_GL=egl python /path/to/render_dexonomy_grasps.py \
      --source output/trial_shadow/succ_grasp --output output/trial_shadow/png --limit 10
"""

import argparse
from itertools import islice
from pathlib import Path

import imageio.v3 as iio
import mujoco
import numpy as np

from dexonomy.sim import HandCfg, MuJoCo_OptCfg, MuJoCo_VisEnv
from dexonomy.util.file_util import load_scene_cfg


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True, help="A .npy file or a grasp directory")
    parser.add_argument("--output", type=Path, required=True, help="Directory for PNG files")
    parser.add_argument("--limit", type=int, default=10, help="Maximum images; 0 means all")
    parser.add_argument("--width", type=int, default=800)
    parser.add_argument("--height", type=int, default=800)
    parser.add_argument("--distance", type=float, default=0.4, help="Camera distance in metres")
    args = parser.parse_args()

    repo = Path.cwd()
    hand_xml = repo / "assets/hand/shadow/right.xml"
    if not hand_xml.is_file():
        parser.error("Run this script from the Dexonomy repository root")
    source = args.source.resolve()
    if not source.exists():
        parser.error(f"Source does not exist: {source}")
    files = [source] if source.is_file() else source.rglob("*.npy")
    if args.limit > 0:
        files = islice(files, args.limit)

    for grasp_file in files:
        grasp = np.load(grasp_file, allow_pickle=True).item()
        scene_path = Path(str(grasp["scene_path"]))
        if not scene_path.is_absolute():
            scene_path = repo / scene_path
        scene_cfg = load_scene_cfg(str(scene_path))
        env = MuJoCo_VisEnv(
            HandCfg(xml_path=str(hand_xml), freejoint=True),
            scene_cfg=scene_cfg,
            sim_cfg=MuJoCo_OptCfg(),
            vis_mode="visual",
        )
        hand_qpos = np.asarray(grasp["grasp_qpos"])[0]

        # Add a background and light to the assembled hand/object scene.
        env._spec.add_texture(
            type=mujoco.mjtTexture.mjTEXTURE_SKYBOX,
            builtin=mujoco.mjtBuiltin.mjBUILTIN_GRADIENT,
            rgb1=[0.75, 0.82, 0.9],
            rgb2=[0.88, 0.92, 0.96],
            width=512,
            height=512,
        )
        env._spec.worldbody.add_light(pos=[0.2, -0.5, 1.0], castshadow=False)
        model = env._spec.compile()
        model.vis.headlight.ambient[:] = [0.65, 0.65, 0.65]
        model.vis.headlight.diffuse[:] = [0.8, 0.8, 0.8]
        data = mujoco.MjData(model)
        data.qpos[:] = np.concatenate([hand_qpos, env._obj_init_qpos])
        mujoco.mj_forward(model, data)

        camera = mujoco.MjvCamera()
        camera.type = mujoco.mjtCamera.mjCAMERA_FREE
        camera.lookat[:] = np.asarray(grasp["obj_pose"])[:3]
        camera.distance = args.distance
        camera.azimuth = 135
        camera.elevation = -25

        option = mujoco.MjvOption()
        option.geomgroup[3] = 0  # Hide collision geometry; retain visual meshes.
        model.vis.global_.offwidth = max(model.vis.global_.offwidth, args.width)
        model.vis.global_.offheight = max(model.vis.global_.offheight, args.height)
        with mujoco.Renderer(model, height=args.height, width=args.width) as renderer:
            renderer.update_scene(data, camera=camera, scene_option=option)
            pixels = renderer.render().copy()

        relative = grasp_file.name if source.is_file() else grasp_file.relative_to(source)
        destination = (args.output / relative).with_suffix(".png")
        destination.parent.mkdir(parents=True, exist_ok=True)
        iio.imwrite(destination, pixels)
        print(destination)


if __name__ == "__main__":
    main()
