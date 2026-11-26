# Generate Shadow-hand grasps and render PNGs

This guide covers a small Dexonomy run using the DGN object assets, then renders
still images of the successful grasps. Run commands from the repository root.
The example uses GPU 0, grasp type `1_Large_Diameter`, and experiment name `trial`.

## 1. Install Dexonomy and prepare objects

Follow the main [README](../README.md) for the environment and submodules:

```bash
git submodule update --init --recursive --progress
conda create -n dexonomy python=3.10
conda activate dexonomy
pip install -e .
pip install -U huggingface_hub
```

The [Hugging Face dataset](https://huggingface.co/datasets/JiayiChenPKU/Dexonomy/tree/main)
provides processed objects and precomputed grasps as separate `.tar.gz` files.
For new grasp synthesis, download `DGN_5k_processed.tar.gz`:

```bash
mkdir -p data/hf assets/object
hf download JiayiChenPKU/Dexonomy DGN_5k_processed.tar.gz \
  --repo-type dataset --local-dir data/hf
tar -tzf data/hf/DGN_5k_processed.tar.gz | head
tar -xzf data/hf/DGN_5k_processed.tar.gz -C assets/object
```

Check that `assets/object/DGN_5k/scene_cfg` and
`assets/object/DGN_5k/processed_data` exist. The default object path in
`dexonomy/config/op/init.yaml` points into that `scene_cfg` directory. If the
archive has an extra enclosing directory, move its contents to this layout.
The `*_vision` archives contain point clouds for learning and are not needed
for this synthesis example. The released `Dexonomy_GRASP_shadow.tar.gz`
contains grasps that have **already been generated**; it is not required to
run the synthesis pipeline.

## 2. Convert annotations and synthesize grasps

```bash
dexrun op=tmpl hand=shadow
dexsyn hand=shadow exp_name=trial 'tmpl_name=[1_Large_Diameter]' \
  'init_gpu=[0]' +init.object.n_cfg=20
```

The first command writes templates to `assets/hand/shadow/init_tmpl`.
`dexsyn` runs object-pose initialization on the GPU, grasp refinement in
MuJoCo, and simulation evaluation. See the [running tutorial](2_running_tutorial.md)
for individual `init`, `grasp`, and `eval` commands and tuning options.

For this example, inspect:

| Path | Contents |
| --- | --- |
| `output/trial_shadow/init_data` | Candidate initializations |
| `output/trial_shadow/grasp_data` | Refined grasps before evaluation |
| `output/trial_shadow/succ_grasp` | Grasps that passed evaluation |

The files are NumPy `.npy` dictionaries. In the current format, `scene_path`
points to a scene configuration containing the object assets;
`grasp_qpos[0]` contains the Shadow-hand pose. There are also
`pregrasp_qpos` and `squeeze_qpos` arrays. `succ_grasp` entries may be symbolic
links to the corresponding files in `grasp_data`.

## 3. Render still images with MuJoCo

Dexonomy's `op=v3d` exports OBJ meshes and `op=vusd` exports USD; neither
writes a PNG per grasp. Use the companion script to reconstruct each saved
scene, set `grasp_qpos`, call MuJoCo forward kinematics, and capture an
offscreen frame:

```bash
MUJOCO_GL=egl python getting_started/render_grasps.py \
  --source output/trial_shadow/succ_grasp \
  --output output/trial_shadow/png \
  --limit 10 \
  --camera-frame hand --camera-aim object
```

Use `--limit 0` for all files. The PNG tree mirrors the `.npy` tree. To
render a single grasp, pass its `.npy` path to `--source`. To see unvalidated
results, use `--source output/trial_shadow/grasp_data`. Run the script from
the repository root. It defaults to `assets/hand/shadow/right.xml`; for
Artus grasp data, install its hand assets and add
`--hand-xml assets/hand/artus/right.xml`. The XML must match the joint layout
in `grasp_qpos`.

### Choose a camera view

The default `--camera-frame hand --camera-aim object` mounts the camera on
the hand root and turns it toward the object's center of mass for each
grasp. This keeps the camera position relative to the hand while the aim
changes with the object. `--camera-pos X Y Z` is in metres in the hand root
frame. Move it sideways when the hand hides the object, move it farther away
to fit both in view, and widen `--fovy` (degrees) if needed. For Artus,
`--camera-pos 0 0.17 0.55 --fovy 65` is a wider starting view.

For a direction that stays fixed relative to the hand, use
`--camera-aim hand --camera-target X Y Z`. The target coordinates are also
in the hand root frame. To reproduce the original view centered on the
object in world coordinates, use `--camera-frame object`. For PNG output,
this mode uses `--distance` (default 0.4 metres); preview also uses it for
the starting view.

### Tune the view interactively

Run the preview from a graphical desktop terminal with a grasp file or
directory. The preview opens the first grasp and does not write PNGs:

```bash
python getting_started/render_grasps.py \
  --source output/trial_shadow/succ_grasp \
  --preview
```

Drag in the MuJoCo viewer to choose a view, then press **P** in the viewer.
The terminal prints two sets of hand-local camera arguments. The first
uses `--camera-aim object` to keep tracking the object in other grasps;
the second uses `--camera-aim hand` and `--camera-target` to preserve the
preview's direction relative to the hand. Copy one set into the PNG command
above. Use the same `--hand-xml` in preview and rendering. If you close the
viewer without pressing **P**, no camera coordinates are printed. The
viewer needs a desktop display; use `MUJOCO_GL=egl` for headless PNG rendering.

`MUJOCO_GL=egl` selects headless GPU rendering on Linux. For a 3D editor,
the following command exports hand and object OBJ files under
`output/trial_shadow/vis_3d`:

```bash
dexrun op=v3d hand=shadow exp_name=trial op.data=succ_grasp op.n_max=10
```

## Troubleshooting

- If `op=tmpl` prints a Warp CUDA warning but reports `Finish template
  conversion`, check that `.npy` files were written under
  `assets/hand/shadow/init_tmpl`. The conversion can succeed despite the
  warning. GPU synthesis still requires working CUDA in both PyTorch and
  Warp.
- Check GPU access with `nvidia-smi` and
  `python -c "import torch; print(torch.cuda.is_available())"`. For Warp, run
  `python -c "import warp as wp; wp.init(); print(wp.is_cuda_available())"`.
  The repo pins Warp 1.8.0; NVIDIA reports that its
  [`cuDeviceGetUuid` issue](https://github.com/NVIDIA/warp/issues/940) was fixed
  in 1.8.1.
- If rendering raises an OpenGL context error on a headless machine, set
  `MUJOCO_GL=egl` **before** starting Python. MuJoCo needs an EGL-capable
  graphics driver for offscreen GPU rendering.
- The Hugging Face dataset viewer does not parse these archives as a standard
  `load_dataset(...)` dataset. Download the archive files directly.
