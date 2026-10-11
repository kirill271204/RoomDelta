"""What changed in the room? One command from two phone videos to a change report.

Usage:
    uv run python run.py BEFORE_VIDEO AFTER_VIDEO [options]

Examples:
    uv run python run.py data/IMG_1763.MOV "data/IMG_1765 2.MOV"
    uv run python run.py before.mov after.mov --goal 1.5 2.0 --goal-name kitchen
    uv run python run.py before.mov after.mov --reconstruction vggt_output.npz   # VGGT run elsewhere (Colab)

Steps: sharpest frames -> VGGT 3D reconstruction of both videos together -> floor levelling and
metric scale -> visibility aware change detection -> moved / added / removed matching ->
Claude descriptions (if ANTHROPIC_API_KEY is set) -> robot map before vs after.
"""

import argparse
from pathlib import Path

from room_change.frames import video_name
from room_change.pipeline import Options, run


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("before", type=Path, help="video of the room before the changes")
    p.add_argument("after", type=Path, help="video of the room after the changes")
    p.add_argument("--out", type=Path, help="output folder (default: outputs/<before>_vs_<after>)")
    p.add_argument("--frames-per-video", type=int, default=30,
                   help="frames per video given to VGGT (default 30; more = better coverage, more memory)")
    p.add_argument("--camera-height", type=float, default=1.4,
                   help="roughly how high the phone was held, metres; sets the real-world scale (default 1.4)")
    p.add_argument("--reconstruction", type=Path, help="use a precomputed VGGT .npz instead of running VGGT here")
    p.add_argument("--goal", type=float, nargs=2, metavar=("X", "Y"), help="robot goal on the map, metres")
    p.add_argument("--goal-name", default="goal", help="name of the goal for the plots")
    p.add_argument("--robot-radius", type=float, default=0.25, help="robot radius, metres (default 0.25)")
    p.add_argument("--no-describe", action="store_true", help="skip the Claude descriptions")
    a = p.parse_args()

    for v in (a.before, a.after):
        if not v.exists():
            p.error(f"video not found: {v}")
    out = a.out or Path("outputs") / f"{video_name(a.before)}_vs_{video_name(a.after)}"
    run(a.before, a.after, out, Options(
        frames_per_video=a.frames_per_video, camera_height_m=a.camera_height, reconstruction=a.reconstruction,
        goal=tuple(a.goal) if a.goal else None, goal_name=a.goal_name, robot_radius_m=a.robot_radius,
        describe=not a.no_describe))


if __name__ == "__main__":
    main()
