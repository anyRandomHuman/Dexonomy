#!/usr/bin/env python3
"""Run Dexonomy template conversion, synthesis, evaluation, and statistics.

Run from any directory with the Dexonomy conda environment active. By default,
all configured hands with a model, annotations, body groups, and skeleton are
included. Each template gets its own experiment so runs can be resumed safely.

Examples:
  python scripts/run_full_evaluation.py --dry-run --scene-glob '/path/to/scenes/**/floating/*.npy'
  python scripts/run_full_evaluation.py --run-name dedup --scene-glob '/path/to/dgn_reduced/scene_cfg/**/floating/*.npy'

The script does not render PNGs or synthesize arm trajectories. For those,
use getting_started/render_grasps.py and the separate trajectory pipeline.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
from glob import glob
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys


REPO = Path(__file__).resolve().parents[1]
HAND_CONFIGS = REPO / "dexonomy/config/hand"
ASSETS = REPO / "assets"
REPORT_FIELDS = [
    "hand", "template", "experiment", "status", "scene_glob",
    "scene_candidates", "epochs",
    "init_grasps", "refined_grasps", "successful_grasps",
    "refinement_rate", "simulation_success_rate",
    "init_objects", "refined_objects", "successful_objects", "updated_utc",
]


def resolve_scene_glob(value: str) -> str:
    path = Path(value).expanduser()
    return str(path if path.is_absolute() else REPO / path)


def hand_readiness(name: str) -> tuple[bool, str]:
    config = HAND_CONFIGS / f"{name}.yaml"
    if not config.is_file():
        return False, f"missing {config}"
    match = re.search(r"^xml_path:\s*(\S+)", config.read_text(), re.MULTILINE)
    if not match:
        return False, "hand config has no top-level xml_path"
    xml_path = match.group(1).replace("${asset_root}", str(ASSETS))
    if not Path(xml_path).is_absolute():
        xml_path = str(REPO / xml_path)
    if not Path(xml_path).is_file():
        return False, f"missing hand XML: {xml_path}"
    hand_dir = ASSETS / "hand" / name
    if hand_dir.is_symlink() and hand_dir.resolve().name != name:
        return False, (
            f"{hand_dir} shares template storage with {hand_dir.resolve()}; "
            "give this hand a separate asset directory before batch conversion"
        )
    for filename in ("body_group.yaml", "skeleton.yaml"):
        if not (hand_dir / filename).is_file():
            return False, f"missing {hand_dir / filename}"
    if not list((hand_dir / "raw_anno").glob("*.yaml")):
        return False, f"no raw annotations in {hand_dir / 'raw_anno'}"
    return True, "ready"


def choose_hands(requested: list[str] | None) -> list[str]:
    names = requested or sorted(path.stem for path in HAND_CONFIGS.glob("*.yaml"))
    ready = []
    for name in names:
        ok, reason = hand_readiness(name)
        if ok:
            ready.append(name)
        elif requested:
            raise ValueError(f"Hand {name}: {reason}")
        else:
            print(f"Skipping hand {name}: {reason}", flush=True)
    if not ready:
        raise ValueError("No hands have all required files")
    return ready


def raw_templates(hand: str, requested: list[str] | None) -> list[str]:
    directory = ASSETS / "hand" / hand / "raw_anno"
    names = sorted(path.stem for path in directory.glob("*.yaml"))
    if requested:
        names = [name for name in names if name in requested]
        missing = set(requested) - set(names)
        if missing:
            raise ValueError(f"Hand {hand} has no annotations for: {sorted(missing)}")
    slugs = [re.sub(r"[^A-Za-z0-9_-]+", "_", name).strip("_") for name in names]
    if any(not slug for slug in slugs) or len(slugs) != len(set(slugs)):
        raise ValueError(f"Hand {hand} has ambiguous template names for experiment paths")
    return names


def run_command(arguments: list[str], dry_run: bool) -> None:
    print("$ " + shlex.join(arguments), flush=True)
    if not dry_run:
        subprocess.run(arguments, cwd=REPO, check=True)


def dexrun(op: str, hand: str, experiment: str | None, workers: int) -> list[str]:
    command = [sys.executable, "-m", "dexonomy.main", f"op={op}", f"hand={hand}"]
    if experiment:
        command.append(f"exp_name={experiment}")
    command.extend((f"n_worker={workers}", "skip_traj=True", "skip_done=True"))
    return command


def count_outputs(directory: Path, template: str) -> tuple[int, int]:
    root = directory / template
    if not root.is_dir():
        return 0, 0
    paths = list(root.rglob("*.npy"))
    objects = {path.relative_to(root).parts[0] for path in paths}
    return len(paths), len(objects)


def log_tail(path: Path, start_offset: int = 0, size: int = 8192) -> str:
    if not path.is_file():
        return ""
    with path.open("rb") as file:
        end = file.seek(0, os.SEEK_END)
        # Hydra can append to or replace a previous main.log.
        if end < start_offset:
            start_offset = 0
        file.seek(max(start_offset, end - size))
        return file.read().decode("utf-8", errors="replace")


def check_initialization_finished(experiment_dir: Path, previous_log_size: int) -> None:
    log_path = experiment_dir / "log/init_0/main.log"
    if "Finish initialization." not in log_tail(log_path, previous_log_size):
        raise RuntimeError(
            f"Initialization did not report completion in {log_path}. "
            "Dexonomy may log an error while returning exit code 0."
        )


def report_row(hand: str, template: str, experiment: str, base: Path,
               status: str, scene_glob: str, scene_candidates: int,
               epochs: int) -> dict[str, str | int]:
    initial, initial_objects = count_outputs(base / "init_data", template)
    refined, refined_objects = count_outputs(base / "grasp_data", template)
    successful, successful_objects = count_outputs(base / "succ_grasp", template)
    return {
        "hand": hand, "template": template, "experiment": experiment,
        "status": status, "scene_glob": scene_glob,
        "scene_candidates": scene_candidates, "epochs": epochs,
        "init_grasps": initial, "refined_grasps": refined,
        "successful_grasps": successful, "init_objects": initial_objects,
        "refinement_rate": round(refined / initial, 4) if initial else 0,
        "simulation_success_rate": round(successful / refined, 4) if refined else 0,
        "refined_objects": refined_objects, "successful_objects": successful_objects,
        "updated_utc": datetime.now(timezone.utc).isoformat(),
    }


def save_report(path: Path, rows: dict[tuple[str, str], dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    with temporary.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=REPORT_FIELDS)
        writer.writeheader()
        for key in sorted(rows):
            writer.writerow(rows[key])
    temporary.replace(path)


def load_report(path: Path) -> dict[tuple[str, str], dict]:
    if not path.is_file():
        return {}
    with path.open(newline="") as file:
        return {(row["hand"], row["template"]): row for row in csv.DictReader(file)}


def check_gpu(gpu: int) -> None:
    code = (
        "import torch, warp as wp; wp.init(); "
        f"assert torch.cuda.device_count() > {gpu}, 'PyTorch cannot see GPU {gpu}'; "
        "assert wp.is_cuda_available(), 'Warp CUDA is unavailable'"
    )
    subprocess.run([sys.executable, "-c", code], cwd=REPO, check=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scene-glob", required=True,
                        help="Glob for floating scene .npy files, such as the reduced DGN set")
    parser.add_argument("--hands", nargs="+", help="Hands to run; default is every ready hand")
    parser.add_argument("--templates", nargs="+", help="Template names for each chosen hand")
    parser.add_argument("--run-name", default="batch",
                        help="Stable experiment prefix; reuse it to resume a run")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--n-scenes", type=int, default=0,
                        help="Number of scene files to sample; 0 processes all matches")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without running")
    parser.add_argument("--skip-template-conversion", action="store_true")
    args = parser.parse_args()

    if not re.fullmatch(r"[A-Za-z0-9_-]+", args.run_name):
        parser.error("--run-name may contain only letters, digits, underscores, and hyphens")
    if args.gpu < 0 or args.workers < 1 or args.epochs < 1 or args.n_scenes < 0:
        parser.error("--gpu must be nonnegative; workers and epochs positive; n-scenes nonnegative")

    scene_glob = resolve_scene_glob(args.scene_glob)
    scenes = sorted(path for path in glob(scene_glob, recursive=True) if path.endswith(".npy"))
    if not scenes:
        parser.error(f"No scene configs match: {scene_glob}")
    hands = choose_hands(args.hands)
    templates = {hand: raw_templates(hand, args.templates) for hand in hands}
    hands = [hand for hand in hands if templates[hand]]
    if not hands:
        parser.error("No requested templates were found")
    attempts = sum(len(templates[hand]) for hand in hands)
    scene_candidates = min(len(scenes), args.n_scenes) if args.n_scenes else len(scenes)
    print(f"Planning {attempts} hand/template runs over {scene_candidates} scene files "
          f"({args.epochs} epochs each).", flush=True)
    print(f"Scene glob: {scene_glob}", flush=True)

    if not args.dry_run:
        check_gpu(args.gpu)

    report_path = REPO / "output" / f"{args.run_name}_evaluation.csv"
    rows = {} if args.dry_run else load_report(report_path)
    for row in rows.values():
        if (row.get("scene_glob") != scene_glob
                or int(row.get("scene_candidates", -1)) != scene_candidates
                or int(row.get("epochs", -1)) != args.epochs):
            raise ValueError(
                f"{report_path} belongs to a different scene set or epoch count. "
                "Use a different --run-name."
            )
    for hand in hands:
        if not args.skip_template_conversion:
            run_command(dexrun("tmpl", hand, None, args.workers), args.dry_run)
        if not args.dry_run:
            missing = [name for name in templates[hand]
                       if not (ASSETS / "hand" / hand / "init_tmpl" / f"{name}.npy").is_file()]
            if missing:
                raise RuntimeError(f"Template conversion failed for {hand}: {missing}")

        for template in templates[hand]:
            slug = re.sub(r"[^A-Za-z0-9_-]+", "_", template).strip("_")
            experiment = f"{args.run_name}_{slug}"
            base = REPO / "output" / f"{experiment}_{hand}"
            common = (f"tmpl_name={template}",)
            init = dexrun("init", hand, experiment, args.workers) + list(common) + [
                f"init_gpu=[{args.gpu}]", f"op.epoch={args.epochs}",
                f"op.object.cfg_path={scene_glob}",
                f"op.object.n_cfg={args.n_scenes}", "op.object.batch_size=1",
            ]
            stages = [
                ("init", init),
                ("grasp", dexrun("grasp", hand, experiment, args.workers)),
                ("eval", dexrun("eval", hand, experiment, args.workers)),
                ("stat", dexrun("stat", hand, experiment, args.workers)),
            ]
            print(f"\n[{hand} / {template}] -> {base}", flush=True)
            for stage, command in stages:
                init_log = base / "log/init_0/main.log"
                old_size = init_log.stat().st_size if stage == "init" and init_log.is_file() else 0
                try:
                    run_command(command, args.dry_run)
                    if stage == "init" and not args.dry_run:
                        check_initialization_finished(base, old_size)
                except (subprocess.CalledProcessError, RuntimeError):
                    if not args.dry_run:
                        rows[(hand, template)] = report_row(
                            hand, template, experiment, base, f"failed_{stage}",
                            scene_glob, scene_candidates, args.epochs,
                        )
                        save_report(report_path, rows)
                    raise
            if not args.dry_run:
                row = report_row(hand, template, experiment, base, "complete",
                                 scene_glob, scene_candidates, args.epochs)
                rows[(hand, template)] = row
                save_report(report_path, rows)
                print(f"Summary: {row['successful_grasps']} successful grasps across "
                      f"{row['successful_objects']} objects", flush=True)

    if not args.dry_run:
        completed = [row for row in rows.values() if row["status"] == "complete"]
        total_refined = sum(int(row["refined_grasps"]) for row in completed)
        total_successful = sum(int(row["successful_grasps"]) for row in completed)
        print(f"Completed {len(completed)} hand/template evaluations: "
              f"{total_successful}/{total_refined} refined grasps passed simulation.",
              flush=True)
        print(f"\nEvaluation report: {report_path}", flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, subprocess.CalledProcessError, RuntimeError) as error:
        print(f"Evaluation stopped: {error}", file=sys.stderr)
        raise SystemExit(1)
