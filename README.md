# What Changed in the Room?

Two phone videos of the same room go in. Out comes a shared 3D model, every object that was **moved, added or removed**, a plain English description of each change, and what the changes mean for a robot that has to drive through the room.

```bash
uv run python run.py before.mov after.mov
```

![Changes found in the example room](examples/bedroom/output/changes_topdown.png)

## Result on the example

Between the two example recordings I moved two chairs, put a backpack on one of them, and moved a bin from the floor onto the bed. The system finds all four, with no false alarms:

| Detected | Claude's description (shortened) | Ground truth |
| :--- | :--- | :--- |
| MOVED 1.84 m | dark purple gaming chair, from the desk to the corner by the door | office chair moved ✅ |
| MOVED 0.82 m | blue chair, from under the desk to the open floor near the bed | blue chair moved ✅ |
| ADDED | dark navy backpack, now on the blue chair's seat | backpack added ✅ |
| MOVED 2.06 m, 0.5 m higher | grey wastebasket, from the floor by the shelf to the top of the bed | bin moved ✅ |

Full output: [`examples/bedroom/output/`](examples/bedroom/output/). The answer key is [`ground_truth.json`](examples/bedroom/input/ground_truth.json).

For a robot, the changes cut the floor it can reach from the doorway from 2.4 m² to 0.8 m², while a route to the desk still exists:

![Robot map before and after](examples/bedroom/output/navmap.png)

## How it works

1. **Frames.** From each video, the sharpest frame in each of 30 equal time windows (motion blur is skipped, coverage is kept).
2. **One shared 3D model.** All frames of *both* videos go through [VGGT](https://github.com/facebookresearch/vggt) in a single pass. The walls and furniture that did not change anchor both recordings in the same coordinate frame, so no separate alignment step is needed.
3. **Level and scale.** The floor is found with RANSAC and becomes z = 0. VGGT has no real world scale, so the scene is scaled assuming the phone was held about 1.4 m above the floor (about 10% accuracy, enough for thresholds in metres).
4. **Change detection that knows what it did not see.** Every 3D point from one recording is projected into every camera of the other and compared with the depth that camera measured there:
   * same depth: the surface is still there (**unchanged**)
   * the camera saw *through* the point to something behind it: the object is gone (**changed**)
   * something was in front, or the spot was never in view: no evidence either way (**unknown**)

   A point is only called changed when the other recording actually looked at that spot and saw empty space. This is what separates real changes from a different viewpoint, occlusion or incomplete coverage. Points that are not confirmed by other frames of their own video (a hand in front of the lens, depth noise at edges) are dropped as **transient**.
5. **Objects.** Changed points are clustered into objects (DBSCAN); specks, thin lines and thin sheets below the depth tolerance are dropped.
6. **Moved, or added and removed?** Each object is cut out of the frames that see it best, with the background masked out. Appearance (DINOv2 features, best match over all pairs of views, so a chair seen from behind in one video still matches its side view in the other) and 3D size decide an optimal one to one pairing. A pair becomes **moved**, with distance and height change.
7. **Descriptions.** Claude looks at the crops of each change and returns a label, a description, a check that a moved pair really is the same object, and anything the clustering merged (this is how the backpack on the chair is found).
8. **Robot map.** Each recording becomes a floor map with three states: free, occupied (10 cm to 1.5 m high) and never observed. The robot keeps its old map and overwrites only the cells the second recording observed. It then reports newly blocked floor, floor that is still free but now cut off, and an A* route to an optional goal.

## Setup

Tested on macOS (Apple Silicon) with Python 3.11. Linux should work the same way without the Homebrew step, but is untested.

```bash
git clone https://github.com/YOUR_GITHUB_USER/room-change.git
cd room-change
brew install uv libusb          # macOS: uv manages Python, Open3D needs libusb
uv sync                         # creates .venv with all Python packages
git clone https://github.com/facebookresearch/vggt.git third_party/vggt
```

On the first run VGGT downloads its weights (about 5 GB) and DINOv2 about 90 MB.

For the descriptions, add an Anthropic API key (optional; without it everything else still runs):

```bash
cp .env.example .env            # then put your key in .env
```

## Run

```bash
uv run python run.py BEFORE_VIDEO AFTER_VIDEO
```

To run the example (the shipped output was made from the full resolution recordings with a Colab reconstruction; the input videos here are smaller 720p copies of those):

```bash
uv run python run.py examples/bedroom/input/before.mp4 examples/bedroom/input/after.mp4 --goal 0.3 2.2 --goal-name desk
```

| Option | Default | What it does |
| :--- | :--- | :--- |
| `--frames-per-video N` | 30 | Frames per video given to VGGT. More frames give better coverage and need more memory |
| `--camera-height M` | 1.4 | Roughly how high the phone was held, in metres. Sets the real world scale |
| `--reconstruction FILE` | | Use a VGGT result computed elsewhere (see below) instead of running VGGT here |
| `--goal X Y` | | Robot goal on the map, in metres. Without it the map still shows reachable floor |
| `--goal-name NAME` | goal | Label for the goal in the plots |
| `--robot-radius M` | 0.25 | Robot radius in metres; obstacles are padded by this |
| `--no-describe` | | Skip the Claude descriptions |
| `--out DIR` | `outputs/BEFORE_vs_AFTER` | Output folder |

Frames and the VGGT reconstruction are cached in the output folder, so a second run only redoes the fast steps (about a minute).

### Recording tips

Walk slowly, keep the phone at a steady height, and film the whole room from at least two directions. Use a different path the second time: overlap is what matters, not identical viewpoints.

## Hardware

VGGT is the only heavy step. It runs on an NVIDIA GPU, on Apple Silicon or (slowly) on the CPU.

| Machine | Frames per video | Tested |
| :--- | :--- | :--- |
| NVIDIA T4, 15 GB (free Colab) | 30 | yes (this is how the example was made) |
| MacBook M1 Pro, 16 GB | 15 | yes, VGGT takes about 3 minutes |

On a Mac the memory need is about 11 GB plus 0.23 GB per frame, and `run.py` warns before starting if it will not fit. With 15 frames per video the example still finds both chairs but misses the bin on the bed: small objects need more views.

**No suitable GPU?** Open [`notebooks/vggt_colab.ipynb`](notebooks/vggt_colab.ipynb) in Google Colab, upload the two videos, and download the result. Then finish on your machine:

```bash
uv run python run.py before.mov after.mov --reconstruction vggt_30f.npz
```

## Outputs

| File | Contents |
| :--- | :--- |
| `changes_described.json` | The change list with labels, descriptions and a summary for the robot |
| `changes.json` | The change list from geometry alone: type, position, size, distance moved, match scores |
| `changes_topdown.png` | Top view of both recordings: removed (red), added (green), moved (purple arrows) |
| `navmap.png` | Robot map before and after, route, newly blocked and cut off floor |
| `navigation.json` | Route lengths, clearances and floor areas |
| `changes.rrd`, `scene.rrd` | Interactive 3D views: `uv run rerun OUTPUT_DIR/changes.rrd` |
| `objects/` | Crops of every changed object, before and after, plus a context frame |

Coordinates are in metres, with the floor at z = 0, z pointing up, and the start of the first recording near the origin.

## Limitations

* **Scale** comes from the assumed phone height, so absolute distances can be off by about 10%. Relative positions are unaffected.
* **Touching objects** are one cluster in 3D (the backpack on the chair); the visual check recovers them as separate changes, but they share one 3D box.
* **Coverage** decides what can be judged: anything only one recording saw is reported as unknown, not as a change.
* **Thin or small objects** (under about 5 cm thick, or seen in very few frames) are below what VGGT depth can reliably separate from noise.

## Repository layout

```
run.py                     command line entry point
src/room_change/
  pipeline.py              the steps, in order
  frames.py                sharpest frame selection
  reconstruct.py           VGGT on CUDA, Apple Silicon or CPU
  scene.py                 posed depth maps and point clouds
  align.py                 floor levelling and scale
  changes.py               visibility aware change detection
  objects.py               clustering changed points into objects
  crops.py                 best views and masks of each object
  match.py                 moved vs added/removed pairing (DINOv2)
  describe.py              Claude labels, checks and descriptions
  navmap.py                occupancy maps, A*, reachable floor
  viz.py                   figures and Rerun recordings
notebooks/vggt_colab.ipynb VGGT on a free Colab GPU
examples/bedroom/          example input videos, answer key and full output
```
