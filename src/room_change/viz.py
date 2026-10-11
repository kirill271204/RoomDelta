"""Figures and Rerun recordings for the reconstruction, the changes and the robot map."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import rerun as rr
import rerun.blueprint as rrb
from matplotlib.colors import ListedColormap

from .changes import CHANGED, TRANSIENT, UNCHANGED, UNKNOWN
from .scene import Scene

VIDEO_COLORS = [(255, 140, 0), (30, 120, 255)]  # orange = first recording, blue = second
KIND_COLOR = {"removed": (230, 30, 30), "added": (30, 180, 30), "moved": (150, 60, 220)}
LABEL_COLOR = {UNCHANGED: (190, 190, 190), UNKNOWN: (140, 165, 255), TRANSIENT: (255, 200, 50)}
MAP_CMAP = ListedColormap(["#c9d3ff", "#ffffff", "#3a3a3a"])  # unknown, free, occupied


def label_colors(labels, kind):
    c = np.array([LABEL_COLOR.get(int(l), KIND_COLOR[kind]) for l in range(4)], np.uint8)
    return c[labels.label]


def scene_rrd(scene: Scene, out: Path, max_points: int = 400_000) -> None:
    """Both recordings' point clouds and cameras, in two tabs: real colours and which recording."""
    names = scene.video_names
    # Entity filters only support a trailing /**, so each excluded path is listed explicitly.
    blueprint = rrb.Blueprint(
        rrb.Tabs(
            rrb.Spatial3DView(name="Real colors", origin="world",
                              contents=["+ /world/**"] + [f"- /world/{n}/points_by_video" for n in names]),
            rrb.Spatial3DView(name="Which video (orange 1st, blue 2nd)", origin="world",
                              contents=["+ /world/**"] + [f"- /world/{n}/points_rgb" for n in names]),
        ),
        collapse_panels=True,
    )
    rr.init("room_change_scene")
    rr.save(out, default_blueprint=blueprint)
    rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)
    radius = rr.Radius.ui_points(2.0)
    for v, name in enumerate(names):
        pts, cols, _ = scene.cloud(v, max_points=max_points)
        rr.log(f"world/{name}/points_rgb", rr.Points3D(pts, colors=cols, radii=radius), static=True)
        rr.log(f"world/{name}/points_by_video", rr.Points3D(pts, colors=VIDEO_COLORS[v], radii=radius), static=True)
    h, w = scene.images.shape[1:3]
    for i in range(scene.n_frames):
        v = scene.video_id[i]
        E = scene.extrinsic[i]
        path = f"world/{names[v]}/cam_{i:02d}"
        rr.log(path, rr.Transform3D(translation=E[:, 3], mat3x3=E[:, :3],
                                    relation=rr.TransformRelation.ChildFromParent), static=True)
        rr.log(path, rr.Pinhole(image_from_camera=scene.intrinsic[i], resolution=[w, h],
                                image_plane_distance=0.07, color=VIDEO_COLORS[v]), static=True)
        # Frames live outside /world so they are not drawn inside the 3D frustums
        rr.log(f"frames/{names[v]}/{i:02d}", rr.Image(scene.images[i]).compress(jpeg_quality=80), static=True)


def changes_topdown(results, changes, names, out: Path, verdict: str = "good") -> None:
    """Before (removed) and after (added) point maps with change boxes and move arrows."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    for ax, (kind, labels, _), name in zip(axes, results, names):
        keep = labels.points[:, 2] > 0.03  # hide the floor so objects stand out
        order = np.argsort(labels.label[keep] == CHANGED)  # changed points on top
        P, C = labels.points[keep][order], label_colors(labels, kind)[keep][order]
        ax.scatter(P[:, 0], P[:, 1], s=0.3, c=C / 255)
        for ch in changes:
            if ch["type"] not in ("moved", kind):
                continue
            o = ch["_objects"][0 if kind == "removed" else -1]
            (x0, y0), (x1, y1) = o.bbox_min[:2], o.bbox_max[:2]
            color = np.array(KIND_COLOR[ch["type"]]) / 255
            ax.add_patch(plt.Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, lw=2, ec=color))
            ax.text(x0, y1 + 0.04, f"{ch['id']} {ch['type']}", fontsize=9, color=color, weight="bold")
            if ch["type"] == "moved" and kind == "added":
                (fx, fy), (tx, ty) = ch["_objects"][0].centroid[:2], ch["_objects"][1].centroid[:2]
                ax.annotate("", (tx, ty), (fx, fy), arrowprops=dict(arrowstyle="->", lw=2.5, color=color))
                ax.text((fx + tx) / 2, (fy + ty) / 2, f"{ch['distance_on_floor_m']:.2f} m", color=color, fontsize=9)
        ax.set_title(f"{name}: {'BEFORE' if kind == 'removed' else 'AFTER'} | purple moved, red removed, green added\n"
                     "grey unchanged, blue not observed, yellow transient", fontsize=10)
        ax.set_xlabel("x (m)"), ax.set_ylabel("y (m)"), ax.set_aspect("equal"), ax.grid(True, alpha=0.3)
    if verdict != "good":
        fig.suptitle(f"Reconstruction quality: {verdict.upper()} (see quality.json); treat these changes with caution",
                     color="#c0392b", fontsize=13, weight="bold")
    plt.tight_layout()
    plt.savefig(out, dpi=90)
    plt.close(fig)


def changes_rrd(results, changes, names, out: Path) -> None:
    """3D before/after views with change boxes and move arrows."""
    blueprint = rrb.Blueprint(
        rrb.Horizontal(*[rrb.Spatial3DView(name=f"{n}: {'before' if k == 'removed' else 'after'}", origin="world",
                                           contents=[f"+ /world/{k}/**", "+ /world/moves/**"])
                         for (k, _, _), n in zip(results, names)]),
        collapse_panels=True,
    )
    rr.init("room_change_changes")
    rr.save(out, default_blueprint=blueprint)
    rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)
    for kind, labels, _ in results:
        rr.log(f"world/{kind}/points", rr.Points3D(labels.points, colors=label_colors(labels, kind),
                                                    radii=rr.Radius.ui_points(2.0)), static=True)
        boxes = [(ch, ch["_objects"][0 if kind == "removed" else -1]) for ch in changes
                 if ch["type"] in ("moved", kind)]
        if boxes:
            rr.log(f"world/{kind}/objects", rr.Boxes3D(
                mins=[o.bbox_min for _, o in boxes], sizes=[o.size for _, o in boxes],
                labels=[f"{ch['id']} {ch['type']}" for ch, _ in boxes],
                colors=[KIND_COLOR[ch["type"]] for ch, _ in boxes]), static=True)
    moves = [ch for ch in changes if ch["type"] == "moved"]
    if moves:
        rr.log("world/moves", rr.Arrows3D(
            origins=[ch["_objects"][0].centroid for ch in moves],
            vectors=[ch["_objects"][1].centroid - ch["_objects"][0].centroid for ch in moves],
            labels=[f"{ch['id']}: {ch['distance_on_floor_m']:.2f} m" for ch in moves],
            colors=KIND_COLOR["moved"]), static=True)


def navmap(before, updated, routes, masks, nav, changes, p, out: Path) -> None:
    """Before map + route | updated map + route | what changed for the robot."""
    fig, axes = plt.subplots(1, 3, figsize=(21, 8))
    goal_name = nav.get("goal_name")
    for ax, (name, grid) in zip(axes[:2], (("before", before), ("after", updated))):
        ax.imshow(grid.cells, origin="lower", extent=grid.extent, cmap=MAP_CMAP, vmin=0, vmax=2)
        title = "BEFORE: robot's original map" if name == "before" else "AFTER: map updated with 2nd recording"
        r = routes.get(name)
        if r is None:
            ax.set_title(title, fontsize=11)
            continue
        if r["reachable"]:
            xy = grid.to_world(np.array(r["_path"]))
            ax.plot(xy[:, 0], xy[:, 1], "-", color="#1f77ff", lw=3)
            info = f"route {r['length_m']} m, closest obstacle {r['min_clearance_m']:.2f} m"
        else:
            info = f"NO ROUTE: {r['reason']}"
        ax.set_title(f"{title}\n{nav['start_name']} -> {goal_name}: {info}", fontsize=11)

    shape = updated.cells.shape
    diff = np.zeros(shape + (4,))
    diff[masks["lost_reach"]] = (1.0, 0.65, 0.2, 0.55)
    diff[masks["newly_blocked"]] = (0.9, 0.1, 0.1, 1)
    diff[masks["newly_free"]] = (0.1, 0.7, 0.1, 1)
    axes[2].imshow(updated.cells, origin="lower", extent=updated.extent, cmap=MAP_CMAP, vmin=0, vmax=2, alpha=0.35)
    axes[2].imshow(diff, origin="lower", extent=updated.extent)
    for ch in changes:
        box = ch.get("to") or ch.get("from")
        if box:
            axes[2].annotate(ch.get("object_label", f"{ch['id']} {ch['type']}"), box["centre_m"][:2], fontsize=9,
                             color="#7a1fa2", weight="bold", xytext=(4, 4), textcoords="offset points")
    area = p.res ** 2
    axes[2].set_title(f"What changed for the robot: reachable floor {masks['reach_before'].sum() * area:.1f} -> "
                      f"{masks['reach_after'].sum() * area:.1f} m2\nred: newly blocked, green: newly free, "
                      f"orange: still free but cut off from {nav['start_name']}", fontsize=11)
    for ax in axes:
        points = [("start", "o")] + ([("goal", "*")] if "goal" in nav else [])
        for key, mk in points:
            ax.plot(*nav[key], mk, color="#ff8c00", ms=14, mec="black")
            ax.annotate(nav[f"{key}_name"], nav[key], xytext=(8, 6), textcoords="offset points", fontsize=10)
        ax.set_xlabel("x (m)"), ax.set_ylabel("y (m)"), ax.grid(True, alpha=0.25)
    fig.text(0.01, 0.01, f"white free, dark occupied ({p.obstacle_z[0] * 100:.0f} cm to {p.obstacle_z[1]} m high), "
             f"lilac never observed; robot radius {p.robot_radius} m; {p.res * 100:.0f} cm cells", fontsize=10)
    plt.tight_layout(rect=(0, 0.03, 1, 1))
    plt.savefig(out, dpi=80)
    plt.close(fig)
