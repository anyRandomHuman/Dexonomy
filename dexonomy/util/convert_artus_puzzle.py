"""Prepare the supplied Artus URDF and PuzzlePiece_B USD for Dexonomy.

This converts geometry and actuation only. A contact template and skeleton
still need to be annotated before Artus grasp synthesis.
"""

from pathlib import Path
import argparse
import json
import shutil

from lxml import etree
import mujoco
import numpy as np
import trimesh
from pxr import Usd, UsdGeom


HAND_URDF = Path(
    "/home/ric-ct/ihm/configuration/configuration/assets/hands/artus_hand/description/artus.urdf"
)
OBJECT_USD = Path(
    "/home/ric-ct/ihm/configuration/configuration/assets/objects/puzzle_pieces/mixed_pieces/PuzzlePiece_B.usd"
)


def convert_hand(bundle: Path) -> Path:
    out = bundle / "assets/hand/artus"
    meshes = out / "meshes"
    meshes.mkdir(parents=True, exist_ok=True)

    # MuJoCo's URDF parser strips mesh path prefixes by default. Supply its
    # mesh directory and retain both visual and collision geometries.
    urdf = etree.parse(str(HAND_URDF)).getroot()
    extension = etree.SubElement(urdf, "mujoco")
    etree.SubElement(
        extension,
        "compiler",
        meshdir=str(HAND_URDF.parent / "meshes"),
        strippath="true",
        discardvisual="false",
    )
    intermediate = bundle / "artus_import.urdf"
    intermediate.write_bytes(etree.tostring(urdf))
    spec = mujoco.MjSpec.from_file(str(intermediate))
    spec.compile()
    xml = etree.fromstring(spec.to_xml().encode())
    xml.find("compiler").set("meshdir", "meshes")

    for geom in xml.xpath(".//geom"):
        geom.set("group", "2" if geom.get("contype") == "0" else "3")

    actuator = etree.SubElement(xml, "actuator")
    joints = xml.xpath(".//joint[@name]")
    for joint in joints:
        name = joint.get("name")
        etree.SubElement(
            actuator,
            "position",
            name=f"act_{name}",
            joint=name,
            kp="5",
            forcerange="-3 3",
        )

    for asset in xml.xpath("./asset/mesh[@file]"):
        name = asset.get("file")
        shutil.copy2(HAND_URDF.parent / "meshes" / name, meshes / name)

    out_xml = out / "right.xml"
    etree.ElementTree(xml).write(
        str(out_xml), encoding="utf-8", xml_declaration=True, pretty_print=True
    )
    config_dir = bundle / "dexonomy/config/hand"
    config_dir.mkdir(parents=True, exist_ok=True)
    (config_dir / "artus.yaml").write_text(
        "xml_path: ${asset_root}/hand/artus/right.xml\n"
        "color: [0.65, 0.65, 0.7, 1.0]\n"
        "arm_hand:\n"
    )
    (out / "body_group.yaml").write_text(
        "# Starter contact groups; verify them against Artus link semantics.\n"
        "body_group:\n"
        "  - [d1_1_urdf, d2_1_urdf, d3_1_urdf, d4_1_urdf] # pinky\n"
        "  - [d1_2_urdf, d2_2_urdf, d3_2_urdf, d4_2_urdf] # ring\n"
        "  - [d1_3_urdf, d2_3_urdf, d3_3_urdf, d4_3_urdf] # middle\n"
        "  - [d1_4_urdf, d2_4_urdf, d3_4_urdf, d4_4_urdf] # index\n"
        "  - [t1_urdf, t2_urdf, t3_urdf, t4_urdf]\n"
        "  - [world]\n"
    )
    intermediate.unlink()
    return out_xml


def convert_object(bundle: Path) -> Path:
    root = bundle / "assets/object/custom"
    obj = root / "processed_data/PuzzlePiece_B"
    mesh_dir = obj / "mesh"
    collision_dir = obj / "collision"
    mjcf_dir = obj / "mjcf"
    info_dir = obj / "info"
    scene_dir = root / "scene_cfg/PuzzlePiece_B/floating"
    for path in (mesh_dir, collision_dir, mjcf_dir, info_dir, scene_dir):
        path.mkdir(parents=True, exist_ok=True)

    stage = Usd.Stage.Open(str(OBJECT_USD))
    mesh_prims = [prim for prim in stage.Traverse() if prim.GetTypeName() == "Mesh"]
    if len(mesh_prims) != 1:
        raise ValueError(f"Expected one object mesh, found {len(mesh_prims)}")
    mesh_prim = mesh_prims[0]
    usd_mesh = UsdGeom.Mesh(mesh_prim)
    face_counts = np.asarray(usd_mesh.GetFaceVertexCountsAttr().Get())
    if not np.all(face_counts == 3):
        raise ValueError("Object mesh must already be triangulated")
    transform = UsdGeom.XformCache().GetLocalToWorldTransform(mesh_prim)
    points = np.asarray(
        [tuple(transform.Transform(point)) for point in usd_mesh.GetPointsAttr().Get()],
        dtype=float,
    )
    points *= UsdGeom.GetStageMetersPerUnit(stage)
    faces = np.asarray(usd_mesh.GetFaceVertexIndicesAttr().Get()).reshape(-1, 3)
    visual = trimesh.Trimesh(vertices=points, faces=faces, process=True)
    visual.fix_normals()
    visual.export(mesh_dir / "simplified.obj")
    pieces = trimesh.decomposition.convex_decomposition(
        visual, maxConvexHulls=32, resolution=200000
    )
    collision = [trimesh.Trimesh(**piece, process=True) for piece in pieces]

    xml = etree.Element("mujoco", model="PuzzlePiece_B")
    etree.SubElement(xml, "compiler", angle="radian", meshdir=".")
    asset = etree.SubElement(xml, "asset")
    etree.SubElement(asset, "mesh", name="visual", file="../mesh/simplified.obj")
    body = etree.SubElement(etree.SubElement(xml, "worldbody"), "body", name="object")
    etree.SubElement(
        body,
        "geom",
        name="object_visual",
        type="mesh",
        mesh="visual",
        contype="0",
        conaffinity="0",
        group="2",
        density="0",
        rgba="0.75 0.78 0.82 1",
    )
    for index, part in enumerate(collision):
        filename = f"hull_{index:03}.obj"
        part.export(collision_dir / filename)
        mesh_name = f"hull_{index:03}"
        etree.SubElement(asset, "mesh", name=mesh_name, file=f"../collision/{filename}")
        etree.SubElement(
            body,
            "geom",
            name=f"collision_{index:03}",
            type="mesh",
            mesh=mesh_name,
            group="3",
            density="1000",
        )
    object_xml = mjcf_dir / "object.xml"
    etree.ElementTree(xml).write(
        str(object_xml), encoding="utf-8", xml_declaration=True, pretty_print=True
    )

    density = 1000.0
    collision_volume = sum(part.volume for part in collision)
    info = {
        "mass": collision_volume * density,
        "density": density,
        "scale": 1.0,
        "gravity_center": visual.center_mass.tolist(),
        "obb": visual.extents.tolist(),
    }
    (info_dir / "simplified.json").write_text(json.dumps(info, indent=2) + "\n")

    relative = "../../../processed_data/PuzzlePiece_B"
    name = "PuzzlePiece_B"
    scene = {
        "scene_id": f"{name}/floating/scale100",
        "scene": {
            name: {
                "type": "rigid_object",
                "file_path": f"{relative}/mesh/simplified.obj",
                "xml_path": f"{relative}/mjcf/object.xml",
                "info_path": f"{relative}/info/simplified.json",
                "scale": np.ones(3, dtype=np.float32),
                "pose": np.array([0, 0, 0, 1, 0, 0, 0], dtype=np.float32),
            }
        },
        "task": {"type": "force_closure", "obj_name": name},
    }
    scene_path = scene_dir / "scale100.npy"
    np.save(scene_path, scene)
    return scene_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    bundle = args.output.resolve()
    bundle.mkdir(parents=True, exist_ok=True)
    print("hand:", convert_hand(bundle))
    print("scene:", convert_object(bundle))

if __name__ == "__main__":
    main()
