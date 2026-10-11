"""Score a run of run.py against an answer key of what was really changed.

Usage:
    uv run python evaluate.py OUTPUT_DIR ANSWER_KEY

Example:
    uv run python evaluate.py examples/kitchen/output examples/kitchen/input/ground_truth.json

The answer key is a JSON file listing the changes you made, for example:
    {"changes": [{"object": "baby seat", "type": "moved", "from": "the table", "to": "the kitchen counter"}]}

Writes OUTPUT_DIR/score.json and adds the score to OUTPUT_DIR/report.html. Needs ANTHROPIC_API_KEY (Claude pairs the answer key with the detections).
"""

import argparse
import json
from pathlib import Path

from dotenv import load_dotenv

from room_change.evaluate import print_score, score
from room_change.report import write_report


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("output_dir", type=Path, help="output folder of run.py")
    p.add_argument("answer_key", type=Path, help="JSON answer key of the real changes")
    a = p.parse_args()
    load_dotenv(Path.cwd() / ".env")
    s = score(a.output_dir, a.answer_key)
    (a.output_dir / "score.json").write_text(json.dumps(s, indent=2) + "\n")
    print_score(s)
    write_report(a.output_dir)  # adds the score to the results page
    print(f"saved {a.output_dir / 'score.json'} and updated {a.output_dir / 'report.html'}")


if __name__ == "__main__":
    main()
