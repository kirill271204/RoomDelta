"""Name, verify and describe detected changes with Claude (vision).

Geometry decides WHERE something changed; Claude only looks at the crops of
those places. For each change it returns an object label, room relative
locations, a one sentence description, and two checks the geometry cannot do
on its own:

* same_object: for a "moved" pair, is it really the same physical object?
  A confident "no" splits the pair back into "removed" + "added".
* other_changes: changes visible in the crops that clustering merged into the
  main object (e.g. a backpack now resting on a chair).
"""

import base64
import io
import json
import os
from pathlib import Path

import anthropic
from PIL import Image

MODEL = "claude-opus-5-5"
MAX_SIDE = 768  # px; enough detail for labelling, keeps requests small

SYSTEM = (
    "You are the perception module of a mobile robot. A 3D change detector compared two "
    "recordings of the same room (BEFORE and AFTER) and found a region that changed. You see "
    "image crops of that region and its measured 3D geometry. Name the object precisely but "
    "briefly, describe locations relative to visible landmarks (furniture, doors, windows, walls), "
    "and report only what the images support. State facts; do not recommend tidying actions. "
    "The 3D detector cannot separate objects that touch, so check the crops yourself: if an object "
    "is on, in or against the main object in one recording but absent in the other (for example a "
    "bag now resting on a chair), list it in other_changes, even if you also mention it in the "
    "description. Call the report_change tool exactly once."
)

TOOL = {
    "name": "report_change",
    "description": "Report the verified label and description of one detected change.",
    "input_schema": {
        "type": "object",
        "properties": {
            "object_label": {"type": "string", "description": "Short noun phrase, e.g. 'black office chair'."},
            "same_object": {"type": "boolean", "description": "Moved changes only: are BEFORE and AFTER the same physical object? true for other types."},
            "same_object_confidence": {"type": "number", "minimum": 0, "maximum": 1},
            "location_before": {"type": "string", "description": "Where it was, relative to landmarks. Empty if added."},
            "location_after": {"type": "string", "description": "Where it is now, relative to landmarks. Empty if removed."},
            "other_changes": {
                "type": "array",
                "description": "Objects touching the main object that appear in only one recording (e.g. an item now resting on it). Empty if none.",
                "items": {
                    "type": "object",
                    "properties": {
                        "object_label": {"type": "string"},
                        "type": {"type": "string", "enum": ["added", "removed"]},
                        "note": {"type": "string"},
                    },
                    "required": ["object_label", "type", "note"],
                },
            },
            "description": {"type": "string", "description": "One plain factual sentence: what changed and where."},
        },
        "required": ["object_label", "same_object", "same_object_confidence", "location_before",
                     "location_after", "other_changes", "description"],
    },
}


def _image_block(path: str) -> dict:
    im = Image.open(path).convert("RGB")
    im.thumbnail((MAX_SIDE, MAX_SIDE))
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=88)
    return {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg",
                                        "data": base64.b64encode(buf.getvalue()).decode()}}


def _facts(change: dict) -> str:
    facts = {k: change[k] for k in ("type", "from", "to", "distance_on_floor_m", "height_change_m") if k in change}
    return ("Measured geometry (metres, floor at z = 0, z up; 'centre_m' z is the object's mid height):\n"
            + json.dumps(facts, indent=1))


def describe_change(client: anthropic.Anthropic, change: dict) -> dict:
    content = [{"type": "text", "text": f"Detected change type: {change['type'].upper()}.\n{_facts(change)}"}]
    for side, key in (("BEFORE", "crops_before"), ("AFTER", "crops_after")):
        if key not in change:
            continue
        *crops, context = change[key]
        content.append({"type": "text", "text": f"{side} recording: close-up crops of the region"})
        content += [_image_block(p) for p in crops]
        content.append({"type": "text", "text": f"{side} recording: whole frame, region boxed in magenta"})
        content.append(_image_block(context))
    resp = client.messages.create(
        model=MODEL, max_tokens=4096, system=SYSTEM, tools=[TOOL],
        # Forcing a specific tool is not supported on this model; the system prompt requires it instead
        tool_choice={"type": "auto"},
        messages=[{"role": "user", "content": content}],
    )
    for block in resp.content:
        if block.type == "tool_use" and block.name == TOOL["name"]:
            return block.input
    raise RuntimeError(f"Claude did not call {TOOL['name']} for {change['id']} (stop_reason={resp.stop_reason})")


def summarize(client: anthropic.Anthropic, changes: list[dict]) -> str:
    lines = [f"- {c['type']}: {c['description']}" for c in changes]
    resp = client.messages.create(
        model=MODEL, max_tokens=4096,  # the model thinks first; thinking counts toward max_tokens
        messages=[{"role": "user", "content": (
            "These changes were detected between two recordings of a room:\n" + "\n".join(lines)
            + "\n\nWrite a 2 to 3 sentence summary for a mobile robot, ending with what matters for "
              "navigating the room. Plain prose, no lists, no dashes.")}],
    )
    text = "".join(b.text for b in resp.content if b.type == "text").strip()
    if not text:
        raise RuntimeError(f"Empty summary from Claude (stop_reason={resp.stop_reason})")
    return text


def describe_all(changes_json: Path, min_split_confidence: float = 0.6) -> dict:
    """Describe every change in changes.json; returns the enriched record (also suitable to save)."""
    record = json.loads(changes_json.read_text())
    # Reads ANTHROPIC_API_KEY. Keys not scoped to a workspace also need the workspace to bill.
    workspace = os.environ.get("ANTHROPIC_WORKSPACE_ID")
    client = anthropic.Anthropic(default_headers={"anthropic-workspace-id": workspace} if workspace else None)
    out = []
    for ch in record["changes"]:
        d = describe_change(client, ch)
        if ch["type"] == "moved" and not d["same_object"] and d["same_object_confidence"] >= min_split_confidence:
            # Claude is confident these are different objects: report removed + added instead of a move
            out.append({**{k: v for k, v in ch.items() if k not in ("to", "crops_after", "distance_on_floor_m",
                                                                   "height_change_m", "match")},
                        "type": "removed", "note": "pair rejected by visual check", **d})
            out.append({**{k: v for k, v in ch.items() if k not in ("from", "crops_before", "distance_on_floor_m",
                                                                   "height_change_m", "match")},
                        "id": ch["id"] + "b", "type": "added", "note": "pair rejected by visual check", **d})
            continue
        out.append({**ch, **d})
        for j, extra in enumerate(d["other_changes"]):
            where = ch.get("to" if extra["type"] == "added" else "from")
            out.append({"id": f"{ch['id']}.{j + 1}", "type": extra["type"], "object_label": extra["object_label"],
                        "description": extra["note"], ("to" if extra["type"] == "added" else "from"): where,
                        "note": f"found by the visual check inside {ch['id']}; shares its 3D location"})
    record["changes"] = out
    record["summary"] = summarize(client, out)
    return record
