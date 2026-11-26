"""Render Dexonomy grasps with a hand-mounted or object-centred camera.

Run this from the Dexonomy repository root, for example:
  MUJOCO_GL=egl python /path/to/render_dexonomy_grasps.py \
      --source output/trial_shadow/succ_grasp --output output/trial_shadow/png --limit 10
"""

import argparse
from itertools import islice
from pathlib import Path
import time

import imageio.v3 as iio
import mujoco
import numpy as np

from dexonomy.sim import HandCfg, MuJoCo_OptCfg, MuJoCo_VisEnv
from dexonomy.util.file_util import load_scene_cfg


def camera_quat(position, target):
    """Quaternion for a MuJoCo camera looking from position toward target."""
    backward = np.asarray(position, dtype=float) - np.asarray(target, dtype=float)
    length = np.linalg.norm(backward)
    if length < 1e-9:
        raise ValueError("Camera position and target must differ")
    backward /= length
    up = np.array([0.0, 1.0, 0.0])
    right = np.cross(up, backward)
    if np.linalg.norm(right) < 1e-9:
        raise ValueError("Camera direction must not be parallel to hand +Y")
    right /= np.linalg.norm(right)
    up = np.cross(backward, right)
    rotation = np.column_stack((right, up, backward))
    quat = np.empty(4)
    mujoco.mju_mat2Quat(quat, rotation.ravel())
    return quat


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True, help="A .npy file or a grasp directory")
    parser.add_argument("--output", type=Path, help="Directory for PNG files (required unless --preview)")
    parser.add_argument("--limit", type=int, default=10, help="Maximum images; 0 means all")
    parser.add_argument("--width", type=int, default=800)
    parser.add_argument("--height", type=int, default=800)
    parser.add_argument("--distance", type=float, default=0.4, help="Camera distance in metres")
    parser.add_argument("--hand-xml", type=Path, default=Path("assets/hand/shadow/right.xml"),
                        help="Hand MJCF path, relative to the Dexonomy root")
    parser.add_argument("--camera-frame", choices=("hand", "object"), default="hand",
                        help="Attach camera to hand or use the original object-centred view")
    parser.add_argument("--camera-aim", choices=("object", "hand"), default="object",
                        help="With a hand camera, dynamically aim at the object or keep a fixed hand-local direction")
    parser.add_argument("--camera-pos", type=float, nargs=3, default=(0.0, 0.17, 0.45),
                        metavar=("X", "Y", "Z"), help="Camera position in the hand root frame (metres)")
    parser.add_argument("--camera-target", type=float, nargs=3, default=(0.0, 0.17, 0.0),
                        metavar=("X", "Y", "Z"), help="Hand-local aim point when --camera-aim hand")
    parser.add_argument("--fovy", type=float, default=45.0, help="Hand camera vertical field of view (degrees)")
    parser.add_argument("--preview", action="store_true",
                        help="Open the first grasp in an interactive viewer; press P to print hand-local camera coordinates")
    args = parser.parse_args()
    if not args.preview and args.output is None:
        parser.error("--output is required unless --preview is set")

    repo = Path.cwd()
    hand_xml = (repo / args.hand_xml).resolve()
    if not hand_xml.is_file():
        parser.error(f"Hand XML not found: {hand_xml}; run from the Dexonomy repository root")
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
        if args.camera_frame == "hand":
            hand_root = env._spec.body(f"{env._cfg.hand_prefix}world")
            camera_kwargs = dict(name="hand_camera", pos=args.camera_pos, fovy=args.fovy)
            if args.camera_aim == "object":
                camera_kwargs.update(
                    mode=mujoco.mjtCamLight.mjCAMLIGHT_TARGETBODYCOM,
                    targetbody=env._model.body(env._obj_active_bodyid).name,
                )
            else:
                camera_kwargs["quat"] = camera_quat(args.camera_pos, args.camera_target)
            hand_root.add_camera(**camera_kwargs)
        model = env._spec.compile()
        model.vis.headlight.ambient[:] = [0.65, 0.65, 0.65]
        model.vis.headlight.diffuse[:] = [0.8, 0.8, 0.8]
        data = mujoco.MjData(model)
        data.qpos[:] = np.concatenate([hand_qpos, env._obj_init_qpos])
        mujoco.mj_forward(model, data)

        if args.camera_frame == "hand":
            camera = "hand_camera"
        else:
            camera = mujoco.MjvCamera()
            camera.type = mujoco.mjtCamera.mjCAMERA_FREE
            camera.lookat[:] = np.asarray(grasp["obj_pose"])[:3]
            camera.distance = args.distance
            camera.azimuth = 135
            camera.elevation = -25

        if args.preview:
            from mujoco import viewer as mujoco_viewer

            requested = {"capture": False}

            def on_key(keycode):
                if keycode in (ord("P"), ord("p")):
                    requested["capture"] = True

            with mujoco_viewer.launch_passive(model, data, key_callback=on_key) as viewer:
                with viewer.lock():
                    viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
                    viewer.cam.lookat[:] = np.asarray(grasp["obj_pose"])[:3]
                    viewer.cam.distance = args.distance
                    viewer.cam.azimuth = 135
                    viewer.cam.elevation = -25
                    viewer.opt.geomgroup[3] = 0
                viewer.sync()
                print("Drag in the viewer to adjust the view; press P in the viewer to capture it.", flush=True)
                while viewer.is_running() and not requested["capture"]:
                    viewer.sync()
                    time.sleep(0.05)
                if requested["capture"]:
                    with viewer.lock():
                        if viewer.cam.type != mujoco.mjtCamera.mjCAMERA_FREE:
                            raise RuntimeError("Select the free camera before pressing P")
                        preview_scene = mujoco.MjvScene(model, maxgeom=1000)
                        mujoco.mjv_updateCamera(model, data, viewer.cam, preview_scene)
                        world_pos = np.asarray(preview_scene.camera[0].pos).copy()
                        world_target = np.asarray(viewer.cam.lookat).copy()
                    root = data.body(f"{env._cfg.hand_prefix}world")
                    hand_pos = root.xpos.copy()
                    hand_rotation = root.xmat.reshape(3, 3).copy()
                    local_pos = hand_rotation.T @ (world_pos - hand_pos)
                    local_target = hand_rotation.T @ (world_target - hand_pos)
                    fmt = lambda values: " ".join(f"{value:.5f}" for value in values)
                    print("Hand-mounted camera settings for this view:")
                    print(f"  --camera-pos {fmt(local_pos)} --camera-aim object")
                    print(f"  --camera-pos {fmt(local_pos)} --camera-target {fmt(local_target)} --camera-aim hand")
            return

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