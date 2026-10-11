# RoomDelta

**What changed in the room?** A tool that compares two phone videos of the same indoor space and tells you what changed, in 3D.

Record a space, move things around, record it again, and run one command. You get:

* a **shared 3D model** of both recordings
* every object that was **moved, added or removed**, with its position and size in metres
* a **plain English description** of each change (optional, uses Claude)
* what the changes mean **for a robot**: newly blocked floor, floor that is now cut off, and a replanned route

Changes are only reported where both recordings actually looked. Anything one recording never saw is marked unknown instead of changed, so a different camera path, occlusion or a missed corner does not produce false alarms.

```bash
uv run python run.py before.mov after.mov
```

## Setup

Tested on macOS (Apple Silicon) with Python 3.11. Linux should work the same way without the Homebrew step, but is untested.

```bash
git clone https://github.com/kirill271204/RoomDelta.git
cd RoomDelta
brew install uv libusb          # macOS: uv manages Python, Open3D needs libusb
uv sync                         # creates .venv with all Python packages
git clone https://github.com/facebookresearch/vggt.git third_party/vggt
```

On the first run VGGT downloads its weights (about 5 GB) and DINOv2 about 90 MB.

For the descriptions, add an Anthropic API key (optional; without it everything else still runs):

```bash
cp .env.example .env            # then put your key in .env
```

## Record your space

1. Film the space once: walk slowly, keep the phone at a steady height, and cover it from at least two directions. 20 to 60 seconds is plenty.
2. Move, add or remove some objects.
3. Film it again. Take a different path if you like: overlap between the recordings is what matters, not identical viewpoints.

Works best for a room sized space with some texture (furniture, objects on surfaces). Plain white walls alone give VGGT little to work with.

The pipeline only needs about 1024 pixels on the long side. Phone clips in 4K at 60 fps are large and slow to upload to Colab, so shrinking them first saves a lot of time (use the shrunk files for every later step, so the frames match):

```bash
ffmpeg -i before.MOV -vf "scale='if(gt(iw,ih),1024,-2)':'if(gt(iw,ih),-2,1024)',fps=15" -an before_small.mp4
```

## Run

```bash
uv run python run.py BEFORE_VIDEO AFTER_VIDEO
```

| Option | Default | What it does |
| :--- | :--- | :--- |
| `--frames-per-video N` | 30 | Frames per video given to VGGT. More frames give better coverage and need more memory |
| `--camera-height M` | 1.4 | Roughly how high the phone was held, in metres. Sets the real world scale |
| `--reconstruction FILE` | | Use a VGGT result computed elsewhere (see Hardware) instead of running VGGT here |
| `--goal X Y` | | Robot goal on the map, in metres. Without it the map still shows reachable floor |
| `--goal-name NAME` | goal | Label for the goal in the plots |
| `--robot-radius M` | 0.25 | Robot radius in metres; obstacles are padded by this |
| `--no-describe` | | Skip the Claude descriptions |
| `--out DIR` | `outputs/BEFORE_vs_AFTER` | Output folder |

A good way to choose `--goal`: run once without it, read the coordinates off `navmap.png`, then run again. Frames and the VGGT reconstruction are cached in the output folder, so the second run only redoes the fast steps (about a minute).

## Outputs

| File | Contents |
| :--- | :--- |
| `changes_described.json` | The confirmed changes with labels, descriptions and a summary for the robot, plus the detections the visual check `rejected` and why |
| `quality.json` | Whether the 3D reconstruction can be trusted, with the measurements behind the verdict |
| `changes.json` | The change list from geometry alone: type, position, size, distance moved, match scores |
| `changes_topdown.png` | Top view of both recordings: removed (red), added (green), moved (purple arrows) |
| `navmap.png` | Robot map before and after, route, newly blocked and cut off floor |
| `navigation.json` | Route lengths, clearances and floor areas |
| `changes.rrd`, `scene.rrd` | Interactive 3D views: `uv run rerun OUTPUT_DIR/changes.rrd` |
| `objects/` | Crops of every changed object, before and after, plus a context frame |
| `run_settings.json` | The settings used, for reproducing the run |

Coordinates are in metres, with the floor at z = 0, z pointing up, and the start of the first recording near the origin.

## How it works

1. **Frames.** From each video, the sharpest frame in each of N equal time windows (motion blur is skipped, coverage is kept).
2. **One shared 3D model.** All frames of *both* videos go through [VGGT](https://github.com/facebookresearch/vggt) in a single pass. Everything that did not change anchors both recordings in the same coordinate frame, so no separate alignment step is needed.
3. **Level and scale.** The floor is found with RANSAC and becomes z = 0. VGGT has no real world scale, so the scene is scaled from the assumed phone height (about 10% accuracy, enough for thresholds in metres).
4. **Quality check.** When VGGT misplaces cameras, the recordings do not line up and walls start to look changed. Four measurements catch this (points that disagree with their own recording, points that match the other recording, camera jumps between frames, and spacing between frames). The run is rated good, questionable or unreliable, and a poor rating is printed, written to `quality.json`, shown on the figures and passed to the summary.
5. **Change detection that knows what it did not see.** Every 3D point from one recording is projected into every camera of the other and compared with the depth that camera measured there:
   * same depth: the surface is still there (**unchanged**)
   * the camera saw *through* the point to something behind it: the object is gone (**changed**)
   * something was in front, or the spot was never in view: no evidence either way (**unknown**)

   A point is only called changed when the other recording actually looked at that spot and saw empty space. Points that are not confirmed by other frames of their own video (a hand in front of the lens, depth noise at edges) are dropped as **transient**.
6. **Objects.** Changed points are clustered into objects (DBSCAN); specks, thin lines and thin sheets below the depth tolerance are dropped. Small clusters, such as a box on a far counter, are kept only on strong evidence: most of the cameras that looked at the spot must have seen through it.
7. **Moved, or added and removed?** Each object is cut out of the frames that see it best, with the background masked out. Appearance (DINOv2 features, best match over all pairs of views, so an object seen from behind in one video still matches its side view in the other) and 3D size decide an optimal one to one pairing. A pair becomes **moved**, with distance and height change. When occlusion splits one object into two clusters, the leftover piece next to a moved object is merged into it.
8. **Visual check and descriptions.** Claude sees each change from both recordings: the object, and the same place in the other recording. It returns a label and a description, and three checks the geometry cannot do alone:
   * **real change:** is a distinct object present in one recording and absent in the other? A spot seen from a new angle, a door edge or a bare wall is rejected (kept in the output with the reason).
   * **same object:** for a move, how likely the two crops show the same object. Unlikely pairs are split into a removal and an addition.
   * **merged objects:** anything the clustering joined, such as a bag put down on a chair.

   The figures are redrawn afterwards, so they show only confirmed changes.
9. **Robot map.** Each recording becomes a floor map with three states: free, occupied (10 cm to 1.5 m high) and never observed. The robot keeps its old map and overwrites only the cells the second recording observed. It then reports newly blocked floor, floor that is still free but now cut off, and an A* route to an optional goal.

## Hardware

VGGT is the only heavy step. It runs on an NVIDIA GPU, on Apple Silicon or (slowly) on the CPU.

| Machine | Frames per video | Tested |
| :--- | :--- | :--- |
| NVIDIA T4, 15 GB (free Colab) | 30 | yes |
| MacBook M1 Pro, 16 GB | 15 | yes, VGGT takes about 3 minutes |

On a Mac the memory need is about 11 GB plus 0.23 GB per frame, and `run.py` warns before starting if it will not fit. Fewer frames still find large objects but can miss small ones, which need more views.

**No suitable GPU?** Open [`notebooks/vggt_colab.ipynb`](notebooks/vggt_colab.ipynb) in Google Colab, upload the two videos, and download the result. Then finish on your machine:

```bash
uv run python run.py before.mov after.mov --reconstruction vggt_30f.npz
```

## Examples

Each example folder holds the two input videos, the answer key (`ground_truth.json`) and the full output.

### Bedroom

[`examples/bedroom/`](examples/bedroom/) holds two recordings of a bedroom, the [answer key](examples/bedroom/input/ground_truth.json) and the [full output](examples/bedroom/output/). Between the recordings two chairs were moved, a backpack was put on one of them, and a bin was moved from the floor onto the bed. The system finds all four:

| Detected | Claude's description (shortened) | Answer key |
| :--- | :--- | :--- |
| MOVED 1.84 m | dark purple gaming chair, from the desk to the corner by the door | office chair moved ✅ |
| MOVED 0.82 m | blue chair, from under the desk to the open floor near the bed | blue chair moved ✅ |
| ADDED | dark navy backpack, now on the blue chair's seat | backpack added ✅ |
| MOVED 2.06 m, 0.5 m higher | grey wastebasket, from the floor by the shelf to the top of the bed | bin moved ✅ |

The geometry also flagged caps on a shelf seen from a new angle. The visual check rejected it ("the same caps show up on the shelf in both recordings, so they were there all along"), and it is listed under `rejected` instead of as a change.

![Changes found in the example](examples/bedroom/output/changes_topdown.png)

For a robot starting at the door, the changes cut the floor it can reach from 2.4 m² to 0.8 m², while a route to the desk still exists:

![Robot map before and after in the example](examples/bedroom/output/navmap.png)

To run it yourself:

```bash
uv run python run.py examples/bedroom/input/before.mp4 examples/bedroom/input/after.mp4 --goal 0.3 2.2 --goal-name desk
```

The shipped output was made from the full resolution recordings with a Colab reconstruction at 30 frames per video; the input videos here are smaller 720p copies of those. In an earlier test with only 15 frames per video (a 16 GB Mac), both chairs were still found but the bin on the bed was missed.

### Kitchen and dining area

[`examples/kitchen/`](examples/kitchen/): a baby seat was moved to the kitchen counter, a vase of flowers along the dining table, and a small caddy with an oil bottle from the corner of the counter to its middle. All three are found and nothing else is reported:

| Detected | Claude's description (shortened) | Answer key |
| :--- | :--- | :--- |
| MOVED 1.14 m | grey baby high chair, to in front of the kitchen sink cabinets | baby seat moved ✅ |
| MOVED 0.65 m | glass vase of white roses, from the wall end of the table toward its middle | flowers moved ✅ |
| MOVED 0.50 m | dark wooden pepper mill, along the back of the kitchen counter | caddy with oil moved ✅ |

Two things this scene needed that the bedroom did not. The caddy sits on a counter at the back, seen only from a distance, so it gives few 3D points and is kept by the strong evidence rule for small objects. The baby seat was split by occlusion into two clusters, and the leftover piece is merged into the move instead of being reported as a separate removal. Claude names the caddy by the pepper mill standing in it.

![Changes found in the kitchen example](examples/kitchen/output/changes_topdown.png)

The input videos are the 1024 pixel, 15 fps copies that the Colab reconstruction was made from:

```bash
uv run python run.py examples/kitchen/input/before.mp4 examples/kitchen/input/after.mp4
```

### A failure case

A 12 m hallway with identical white doors (recordings not included): VGGT misplaced cameras (jumps of 10 m between frames) and the change list was mostly door edges and frames. This failure is why the quality check exists: its thresholds were set between this run and the bedroom, and this run's measurements fail all four of them.

## Limitations

* **Scale** comes from the assumed phone height, so absolute distances can be off by about 10%. Relative positions are unaffected.
* **Touching objects** form one cluster in 3D (for example a bag on a chair). The visual check reports them as separate changes, but they share one 3D box.
* **Coverage** decides what can be judged: anything only one recording saw is reported as unknown, not as a change.
* **Thin or small objects** (under about 5 cm thick, or seen in very few frames) are below what VGGT depth can reliably separate from noise.
* **Large or repetitive spaces** (long corridors, rows of identical doors) can defeat VGGT within the frames that fit in one pass. The quality check flags this, but cannot fix it: record a smaller area.

## Repository layout

```
run.py                     command line entry point
src/room_change/
  pipeline.py              the steps, in order
  frames.py                sharpest frame selection
  reconstruct.py           VGGT on CUDA, Apple Silicon or CPU
  scene.py                 posed depth maps and point clouds
  align.py                 floor levelling and scale
  quality.py               is the reconstruction trustworthy?
  changes.py               visibility aware change detection
  objects.py               clustering changed points into objects
  crops.py                 best views and masks of each object
  match.py                 moved vs added/removed pairing (DINOv2)
  describe.py              Claude labels, checks and descriptions
  navmap.py                occupancy maps, A*, reachable floor
  viz.py                   figures and Rerun recordings
notebooks/vggt_colab.ipynb VGGT on a free Colab GPU
examples/bedroom/          example input videos, answer key and full output
examples/kitchen/          second example
```
