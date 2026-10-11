"""Is the reconstruction trustworthy? Warn instead of reporting changes that are really VGGT errors.

When VGGT misplaces cameras, the two recordings do not line up and walls, doors
and floors start to look "changed". Four signals catch this; the thresholds sit in
the gap between a good reconstruction (a bedroom, about 4 by 3 m) and a broken one
(a 12 m hallway with identical white doors, 30 frames over a minute of walking):

    signal                                   good        broken      warn when
    transient points (inconsistent within    1 to 4%     21 to 24%   > 10%
      their own recording)
    unchanged points                         79 to 90%   40 to 55%   < 65%
    largest camera jump between frames       0.7-1.2 m   10.3 m      > 3 m
    typical camera step between frames       0.2-0.3 m   0.9-1.0 m   > 0.6 m
"""

from dataclasses import dataclass

import numpy as np

from .changes import TRANSIENT, UNCHANGED, PointLabels
from .scene import Scene


@dataclass
class QualityParams:
    max_transient: float = 0.10
    min_unchanged: float = 0.65
    max_camera_jump_m: float = 3.0
    max_median_step_m: float = 0.6


def assess(scene: Scene, labels: list[PointLabels], p: QualityParams = QualityParams()) -> dict:
    """Metrics per recording, plain language warnings, and a verdict: good, questionable or unreliable."""
    metrics, warnings = {}, []
    for v, lab in enumerate(labels):
        name = scene.video_names[v]
        centers = scene.camera_centers()[scene.frames_of(v)]
        steps = np.linalg.norm(np.diff(centers, axis=0), axis=1)
        m = {"transient_fraction": round(float((lab.label == TRANSIENT).mean()), 3),
             "unchanged_fraction": round(float((lab.label == UNCHANGED).mean()), 3),
             "largest_camera_jump_m": round(float(steps.max()), 2),
             "median_camera_step_m": round(float(np.median(steps)), 2)}
        metrics[name] = m
        if m["transient_fraction"] > p.max_transient:
            warnings.append(f"{name}: {m['transient_fraction']:.0%} of points disagree with other frames of the same "
                            "recording, so VGGT placed some of its cameras inconsistently")
        if m["unchanged_fraction"] < p.min_unchanged:
            warnings.append(f"{name}: only {m['unchanged_fraction']:.0%} of points match the other recording, so the "
                            "two recordings may not line up (walls and doors would then look changed)")
        if m["largest_camera_jump_m"] > p.max_camera_jump_m:
            warnings.append(f"{name}: the camera jumps {m['largest_camera_jump_m']:.1f} m between two frames, which is "
                            "not possible while walking; some frames are misplaced")
        if m["median_camera_step_m"] > p.max_median_step_m:
            warnings.append(f"{name}: frames are about {m['median_camera_step_m']:.1f} m apart, too far for VGGT to "
                            "connect them reliably; record more slowly or use more frames per video")
    verdict = "good" if not warnings else "questionable" if len(warnings) <= 2 else "unreliable"
    advice = None
    if warnings:
        advice = ("Treat the changes below with caution. Usually this means the space is larger than a room or has "
                  "few distinct features: record a smaller area, walk more slowly, or raise --frames-per-video.")
    return {"verdict": verdict, "warnings": warnings, "advice": advice, "metrics": metrics}
