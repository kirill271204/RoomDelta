"""The whole pipeline, step by step. run.py is the command line front end.

    frames -> VGGT reconstruction -> level + scale -> change detection -> move matching
           -> Claude descriptions (optional) -> robot map

Frames and the reconstruction are cached in the output folder (the slow parts);
everything after them is recomputed on each run (about a minute).
"""

import json
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.ndimage import distance_transform_edt

from . import viz
from .align import prepare_scene
from .changes import detect, summary
from .crops import best_crops
from .frames import extract_frames, video_name
from .match import embed, match_objects
from .navmap import (FREE, OCCUPIED, UNKNOWN, MapParams, astar, cost_map, grid_bounds, occupancy,
                     path_length, reachable, snap_to_free, update)
from .objects import cluster_changes
from .scene import Scene

N_VIEWS = 8   # crops per object used for appearance matching
SNAP_M = 0.5  # a start/goal too close to obstacles moves to the nearest safe spot within this distance


@dataclass
class Options:
    frames_per_video: int = 30
    camera_height_m: float = 1.4
    reconstruction: Path | None = None  # precomputed VGGT .npz (e.g. from the Colab notebook)
    goal: tuple[float, float] | None = None
    goal_name: str = "goal"
    robot_radius_m: float = 0.25
    describe: bool = True


# ---------------------------------------------------------------- frames + reconstruction

def step_frames(videos: list[Path], names: list[str], frames_dir: Path, n: int) -> None:
    for video, name in zip(videos, names):
        d = frames_dir / name
        if len(list(d.glob("*.jpg"))) == n:
            print(f"[frames] {name}: cached ({n} frames)")
            continue
        for old in d.glob("*.jpg"):
            old.unlink()
        written = extract_frames(video, d, n)
        print(f"[frames] {name}: {len(written)} sharpest frames from {video.name}")


def step_reconstruct(frames_dir: Path, names: list[str], n: int, out: Path) -> None:
    from .reconstruct import memory_warning, run_vggt  # imports torch/VGGT only when needed

    if out.exists():
        print(f"[vggt] cached: {out}")
        return
    warning = memory_warning(n * len(names))
    if warning:
        print(f"[vggt] WARNING: {warning}")
    run_vggt([frames_dir / name for name in names], names, n, out)


# ---------------------------------------------------------------- change detection + matching

def _box(o) -> dict:
    return {"centre_m": o.centroid.round(3).tolist(), "bbox_min_m": o.bbox_min.round(3).tolist(),
            "bbox_max_m": o.bbox_max.round(3).tolist(), "size_m": o.size.round(3).tolist()}


def _change_list(objects, crops, matches, out_dir: Path) -> list[dict]:
    """Merge matched removed/added pairs into 'moved' entries and save each object's crops."""
    (out_dir / "objects").mkdir(parents=True, exist_ok=True)

    def save(kind, i, tag):
        """The 3 best crops plus one context frame (whole view, object boxed)."""
        paths = []
        for j, c in enumerate(crops[kind][i][:3]):
            path = out_dir / "objects" / f"{tag}_{kind}_view{j}.jpg"
            c.image.save(path, quality=92)
            paths.append(str(path))
        context = out_dir / "objects" / f"{tag}_{kind}_context.jpg"
        crops[kind][i][0].context.save(context, quality=88)
        return paths + [str(context)]

    changes, used = [], {"removed": set(), "added": set()}
    for m in matches:
        r, a = objects["removed"][m.removed], objects["added"][m.added]
        tag = f"change{len(changes) + 1:02d}"
        d = a.centroid - r.centroid
        changes.append({
            "id": tag, "type": "moved", "from": _box(r), "to": _box(a),
            "distance_on_floor_m": round(float(np.linalg.norm(d[:2])), 3),
            "height_change_m": round(float(d[2]), 3),
            "match": {"appearance": round(m.appearance, 3), "size": round(m.size, 3), "score": round(m.score, 3)},
            "crops_before": save("removed", m.removed, tag), "crops_after": save("added", m.added, tag),
            "_objects": (r, a),
        })
        used["removed"].add(m.removed)
        used["added"].add(m.added)
    for kind in ("removed", "added"):
        for i, o in enumerate(objects[kind]):
            if i in used[kind]:
                continue
            tag = f"change{len(changes) + 1:02d}"
            changes.append({"id": tag, "type": kind, ("from" if kind == "removed" else "to"): _box(o),
                            f"crops_{'before' if kind == 'removed' else 'after'}": save(kind, i, tag),
                            "_objects": (o,)})
    return changes


def step_detect(scene: Scene, frames_dir: Path, out_dir: Path) -> list[dict]:
    results, objects, crops, emb = [], {}, {}, {}
    for source, kind in ((0, "removed"), (1, "added")):
        labels = detect(scene, source)
        objects[kind] = cluster_changes(labels, kind)
        crops[kind] = [best_crops(scene, o, source, frames_dir, k=N_VIEWS) for o in objects[kind]]
        emb[kind] = [embed([c.masked() for c in cs]) for cs in crops[kind]]
        results.append((kind, labels, objects[kind]))
        print(f"[detect] {scene.video_names[source]} points: {summary(labels)}")

    matches, _ = match_objects(objects["removed"], objects["added"], emb["removed"], emb["added"])
    changes = _change_list(objects, crops, matches, out_dir)
    for ch in changes:
        if ch["type"] == "moved":
            print(f"[detect] {ch['id']} MOVED {ch['distance_on_floor_m']:.2f} m "
                  f"(appearance {ch['match']['appearance']:.2f}, size {ch['match']['size']:.2f})")
        else:
            box = ch["from" if ch["type"] == "removed" else "to"]
            print(f"[detect] {ch['id']} {ch['type'].upper()} at {np.round(box['centre_m'], 2).tolist()} m, "
                  f"size {np.round(box['size_m'], 2).tolist()} m")

    viz.changes_topdown(results, changes, scene.video_names, out_dir / "changes_topdown.png")
    viz.changes_rrd(results, changes, scene.video_names, out_dir / "changes.rrd")
    record = {"before": scene.video_names[0], "after": scene.video_names[1], "units": "metres, floor at z = 0, z up",
              "changes": [{k: v for k, v in ch.items() if not k.startswith("_")} for ch in changes]}
    (out_dir / "changes.json").write_text(json.dumps(record, indent=2) + "\n")
    return changes


def step_describe(out_dir: Path) -> Path | None:
    from dotenv import load_dotenv

    load_dotenv(Path.cwd() / ".env")
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("[describe] skipped: ANTHROPIC_API_KEY not set (copy .env.example to .env and add your key)")
        return None
    from .describe import describe_all

    record = describe_all(out_dir / "changes.json")
    out = out_dir / "changes_described.json"
    out.write_text(json.dumps(record, indent=2) + "\n")
    for ch in record["changes"]:
        print(f"[describe] {ch['id']:10s} {ch['type'].upper():8s} {ch['object_label']}: {ch['description']}")
    print(f"[describe] summary: {record['summary']}")
    return out


# ---------------------------------------------------------------- robot map

def _plan(grid, start, goal, p):
    cost = cost_map(grid, p)
    ends = {}
    for name, xy in (("start", start), ("goal", goal)):
        cell = snap_to_free(cost, grid.to_cell(xy), int(SNAP_M / grid.res))
        if cell is None:
            return {"reachable": False, "reason": f"no safe spot for the robot within {SNAP_M} m of the {name}"}
        ends[name] = cell
        ends[name + "_moved_m"] = round(float(np.linalg.norm(grid.to_world(cell) - np.asarray(xy))), 2)
    path, _ = astar(cost, ends["start"], ends["goal"])
    if path is None:
        return {"reachable": False, "reason": "start and goal are not connected by a gap the robot fits through"}
    clearance = distance_transform_edt(grid.cells != OCCUPIED) * grid.res
    rows, cols = np.array(path).T
    return {"reachable": True, "length_m": round(path_length(grid, path), 2),
            "min_clearance_m": round(float(clearance[rows, cols].min()), 2),
            "unknown_cells_crossed": int((grid.cells[rows, cols] == UNKNOWN).sum()),
            "start_moved_m": ends["start_moved_m"], "goal_moved_m": ends["goal_moved_m"], "_path": path}


def step_navigation(scene: Scene, opt: Options, changes: list[dict], out_dir: Path) -> dict:
    p = MapParams(robot_radius=opt.robot_radius_m)
    # The robot starts where the first recording started (typically the entrance)
    nav = {"start": scene.camera_centers()[scene.frames_of(0)[0], :2].round(2).tolist(),
           "start_name": "start of 1st recording"}
    if opt.goal is not None:
        nav.update(goal=list(opt.goal), goal_name=opt.goal_name)

    clouds = [scene.cloud(v)[0] for v in (0, 1)]
    floor_clouds = [scene.cloud(v, conf_pct=0)[0] for v in (0, 1)]  # all pixels, for floor evidence
    origin, shape = grid_bounds(clouds, p.res)
    before, after_only = (occupancy(c, origin, shape, p, floor_points=f) for c, f in zip(clouds, floor_clouds))
    updated = update(before, after_only)

    routes = {}
    if opt.goal is not None:
        routes = {name: _plan(g, nav["start"], nav["goal"], p) for name, g in (("before", before), ("after", updated))}
    # Only free <-> occupied is a real change; unknown -> occupied is just floor seen for the first time
    masks = {"newly_blocked": (before.cells == FREE) & (updated.cells == OCCUPIED),
             "newly_free": (before.cells == OCCUPIED) & (updated.cells == FREE)}
    # Observed free floor the robot can drive to from the start. Unknown cells may be crossed (as in
    # planning) but are not counted: they include space behind walls nobody has seen.
    for name, g in (("before", before), ("after", updated)):
        cost = cost_map(g, p)
        start = snap_to_free(cost, g.to_cell(nav["start"]), int(SNAP_M / g.res))
        connected = reachable(cost, start) if start is not None else np.zeros(shape, bool)
        masks[f"reach_{name}"] = connected & (g.cells == FREE)
    # Floor that is still empty now but the robot can no longer get to (cut off by the new obstacles)
    masks["lost_reach"] = masks["reach_before"] & (updated.cells == FREE) & ~masks["reach_after"]

    viz.navmap(before, updated, routes, masks, nav, changes, p, out_dir / "navmap.png")
    area = p.res ** 2
    result = {**{k: v for k, v in nav.items()}, "robot_radius_m": p.robot_radius,
              **{f"route_{k}": {kk: vv for kk, vv in r.items() if not kk.startswith("_")} for k, r in routes.items()},
              "newly_blocked_floor_m2": round(float(masks["newly_blocked"].sum() * area), 2),
              "newly_free_floor_m2": round(float(masks["newly_free"].sum() * area), 2),
              "reachable_floor_before_m2": round(float(masks["reach_before"].sum() * area), 2),
              "reachable_floor_after_m2": round(float(masks["reach_after"].sum() * area), 2),
              "free_floor_cut_off_m2": round(float(masks["lost_reach"].sum() * area), 2)}
    (out_dir / "navigation.json").write_text(json.dumps(result, indent=2) + "\n")
    print(f"[robot] reachable floor {result['reachable_floor_before_m2']} -> {result['reachable_floor_after_m2']} m2, "
          f"newly blocked {result['newly_blocked_floor_m2']} m2, cut off {result['free_floor_cut_off_m2']} m2")
    for k, r in routes.items():
        print(f"[robot] route {k}: " + (f"{r['length_m']} m" if r["reachable"] else f"none ({r['reason']})"))
    return result


# ---------------------------------------------------------------- everything

def run(before: Path, after: Path, out_dir: Path, opt: Options) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    frames_dir = out_dir / "frames"

    if opt.reconstruction is not None:
        # Names must match the frame folders the reconstruction was made from
        names = [str(n) for n in np.load(opt.reconstruction)["video_names"]]
        npz = opt.reconstruction
        n = int(np.bincount(np.load(opt.reconstruction)["video_id"]).max())
        if n != opt.frames_per_video:
            print(f"[vggt] using {n} frames per video, as in {npz}")
        opt.frames_per_video = n
    else:
        names = [video_name(before), video_name(after)]
        npz = out_dir / f"vggt_{opt.frames_per_video}f.npz"
    if names[0] == names[1]:
        names = [names[0] + "_before", names[1] + "_after"]

    step_frames([before, after], names, frames_dir, opt.frames_per_video)
    if opt.reconstruction is None:
        step_reconstruct(frames_dir, names, opt.frames_per_video, npz)

    scene, scale = prepare_scene(npz, opt.camera_height_m)
    print(f"[scale] floor levelled; camera height {opt.camera_height_m} m -> {scale:.3f} m per VGGT unit")
    viz.scene_rrd(scene, out_dir / "scene.rrd")

    changes = step_detect(scene, frames_dir, out_dir)
    described = step_describe(out_dir) if opt.describe else None
    if described is not None:
        changes = json.loads(described.read_text())["changes"]
    step_navigation(scene, opt, changes, out_dir)

    (out_dir / "run_settings.json").write_text(json.dumps({
        "before": str(before), "after": str(after), "reconstruction": str(npz),
        "frames_per_video": opt.frames_per_video, "camera_height_m": opt.camera_height_m,
        "robot_radius_m": opt.robot_radius_m, "goal": opt.goal, "goal_name": opt.goal_name}, indent=2) + "\n")
    print(f"\nDone. Results in {out_dir}/ ; 3D views: uv run rerun {out_dir}/scene.rrd {out_dir}/changes.rrd")
