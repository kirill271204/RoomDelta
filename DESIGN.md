# Design note

## The problem as I framed it

A robot keeps a map of its space and uses it to plan. When someone moves a chair, the map is wrong in two places: where the chair was and where it is now. "What changed?" is really "which parts of my map can I still trust?". That framing drove three decisions that run through the whole system:

1. **Saying "I don't know" is a valid answer.** A part of the room that only one recording saw is not a change, it is unknown. Most false alarms in change detection come from treating "not seen" as "gone".
2. **Geometry finds changes, vision explains them.** Where something changed is a measurable 3D fact. What it is and whether it is the same object as before is a recognition problem. Each is handled by the tool that is good at it.
3. **The output should end in a consequence for the robot,** not only a list: which floor is newly blocked and whether a route still exists.

## Key decisions

**One shared VGGT pass instead of two reconstructions and an alignment step.** The frames of both recordings go into VGGT together, so both come out in one coordinate frame. The alternative, COLMAP or two VGGT runs aligned with ICP, adds a step that fails exactly when it matters: when many objects moved, ICP can lock onto the moved furniture. Joint inference lets everything that stayed put anchor the alignment. The cost is memory: all frames must fit in one pass.

**Camera only, with real scale from an assumption.** A humanoid sees mostly through cameras, so the core pipeline uses video alone and runs with any phone. VGGT has no metric scale, and my thresholds are in metres (6 cm depth tolerance, 25 cm robot radius). I first measured a known object (a 2.2 m bed) by hand, which was accurate but did not carry over to a new reconstruction. I replaced it with an assumed phone height of 1.4 m: no input needed and about 10% error, which the thresholds tolerate. For the bedroom both methods agreed within 1% (camera height 1.39 m from the bed measurement).

**Change detection by visibility, not by point distance.** The obvious approach is to compare the two point clouds and flag points without a close neighbour. It fails on everything this challenge asks about, because a region seen from a new angle, or not seen at all, has no close neighbour either. Instead every point from one recording is projected into every camera of the other and compared with the depth that camera measured. A point is changed only if the other recording looked at that spot and saw through it to something behind. Points no camera of the other recording saw are unknown. Points that other frames of their own recording do not confirm (a hand in front of the lens, depth noise at edges) are transient and ignored.

**Moved objects by appearance and size, matched one to one.** Each removed object is paired with at most one added object using DINOv2 features and 3D size, with an optimal assignment. Two details mattered. First, the crops are masked to the object, because unmasked crops matched the office chair to the blue chair (0.69) more than to itself (0.48): the desk behind both dominated. Second, similarity is the best match over all pairs of views, because the first recording saw the office chair only from behind and the second from the front.

**Claude as a checker, not a detector.** Claude never decides where something changed. It receives crops of places the geometry already flagged, from both recordings, and answers questions geometry cannot: is this a distinct object or a door edge seen from another angle, is this moved pair really the same object, and did the clustering merge two objects (a backpack on a chair). Rejected detections stay in the output with the reason, so nothing disappears silently.

**A quality check before trusting any of it.** If VGGT misplaces cameras, the recordings do not line up and walls look changed. Four measurements catch this and the run is rated good, questionable or unreliable.

**A robot map with three states.** Free, occupied and never observed. The robot keeps its old map and overwrites only cells the new recording observed, as a real robot would. It reports newly blocked floor, floor that is still free but cut off, and an A* route.

## What failed, and what it taught me

**A 12 m hallway.** Thirty frames over a minute of walking down a corridor of identical white doors gave camera jumps of 10 m between consecutive frames. The change list was mostly door edges and picture frames, and Claude itself flagged most of them as mistaken matches. I did not tune the detector to hide this. I added the quality check instead, with thresholds set in the wide gap between this run and the bedroom:

| Measurement | Bedroom | Hallway | Warn when |
| :--- | :--- | :--- | :--- |
| Points inconsistent within their own recording | 1 to 4% | 21 to 24% | above 10% |
| Points matching the other recording | 79 to 90% | 40 to 55% | below 65% |
| Largest camera jump between frames | 0.7 to 1.2 m | 10.3 m | above 3 m |
| Typical camera step between frames | 0.2 to 0.3 m | 0.9 to 1.0 m | above 0.6 m |

The lesson: the honest output for a broken reconstruction is a warning, not a list of changes.

**VGGT's confidence saturates on plain surfaces.** About 61% of pixels have confidence exactly at VGGT's minimum, mostly plain floor and wall. Filtering by confidence threw away the very floor that proves a chair is gone, and the moved office chair came out as unknown. The fix was to keep confidence filtering for building point clouds but use all pixels as evidence of empty space, where multi view voting handles the noise.

**Filters tuned on one room.** The size filters were set on the bedroom, where the changes were chairs of thousands of voxels. In the kitchen, a small caddy on a far counter gave about 80 voxels and was dropped. Size alone could not separate it from noise, but the evidence could: for real small changes, 84 to 95% of the cameras that looked at the spot saw through it, against mostly 60 to 79% for noise. Small clusters are now kept on that evidence.

**One object, two clusters.** Occlusion split the baby seat into two clusters. The larger one was matched as moved and the leftover was reported as a separate removal. Pieces within 15 cm of a moved object are now merged into it.

**Claude could not tell "new" from "always there".** A cap on a shelf, seen from a new angle, passed the new small object rule, and Claude confirmed it as added. It had only been shown the after recording. Giving Claude the same place in the other recording fixed it ("the same caps show up on the shelf in both recordings"). The lesson: a verifier is only as good as the evidence it is shown.

**My own schema was ambiguous.** I asked Claude for `same_object` plus a confidence, and it reported "not the same, 0.2" meaning a 20% chance of being the same. My code read it as 20% sure of "not the same" and kept bad pairs. One field with an explicit meaning, `same_object_probability`, removed the ambiguity.

## Results

| Scene | Found | Missed | False alarms | Type correct |
| :--- | :--- | :--- | :--- | :--- |
| Bedroom | 4 of 4 | 0 | 0 | 4 of 4 |
| Kitchen and dining area | 3 of 3 | 0 | 0 | 3 of 3 |
| Hallway | not scored: its measurements fail all four quality thresholds | | | |

Scores come from `evaluate.py`, which I checked against a deliberately wrong answer key: it reported the planted miss, false alarm and wrong type.

## Tradeoffs I accepted

* **Frames versus memory.** Thirty frames per video fit a free 15 GB GPU; a 16 GB Mac fits about fifteen, which still finds large objects but missed the bin on the bed in an early test.
* **Video size.** Phones record 4K at 60 fps, but the pipeline needs about 1024 pixels. Shrinking first turned a 30 minute Colab step into a couple of minutes.
* **Claude in the loop** costs a few cents per run and needs an API key. Everything else runs without it, and the change list is still produced from geometry alone.
* **Touching objects** share one 3D box. Claude reports them as separate changes, but the backpack's position is the chair's position.

## What I would do next

* Run VGGT on a larger cloud GPU so 60 to 100 frames per video fit, which should help larger spaces.
* Use iPhone LiDAR (Record3D) as a second input path, both for metric scale and to measure how far the camera only result is off.
* Split touching objects with a segmentation model, so each gets its own 3D box.
* Keep a map over many sessions instead of comparing two, so the robot learns which objects move often.
* Record a control pair with nothing changed, to measure the false alarm rate directly.
