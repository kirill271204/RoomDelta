"""Name, verify and describe detected changes with Claude (vision).

Geometry decides WHERE something changed; Claude only looks at the crops of
those places. For each change it returns an object label, room relative
locations, a one sentence description, and three checks the geometry cannot do
on its own:

* real_change: is there an actual object that changed, or did the geometry flag
  a bare wall, a door edge seen from another angle, or reconstruction noise?
  Rejected detections are kept in the output under "rejected", with the reason.
* same_object_probability: for a "moved" pair, how likely is it the same physical
  object? Below 0.5 the pair is split into "removed" + "added", and each half is
  described on its own.
* other_changes: changes visible in the crops that clustering merged into the
  main object (e.g. a bag now resting on a chair).
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
    "description. For an ADDED or REMOVED change you also see the same place in the other recording: "
    "real_change is true only if a distinct object is present in one recording and absent in the "
    "other; if it is visible in both, it was there all along and real_change is false. "
    "Call the report_change tool exactly once."
)

TOOL = {
    "name": "report_change",
    "description": "Report the verified label and description of one detected change.",
    "input_schema": {
        "type": "object",
        "properties": {
            "object_label": {"type": "string", "description": "Short noun phrase, e.g. 'black office chair'."},
            "real_change": {"type": "boolean", "description": "true if a distinct object was moved, added or removed; false if the region is only bare wall, floor, a door or frame seen from a different angle, or reconstruction noise."},
            "real_change_reason": {"type": "string", "description": "One short sentence justifying real_change."},
            "same_object_probability": {"type": "number", "minimum": 0, "maximum": 1, "description": "MOVED changes only: probability that the BEFORE and AFTER crops show the same physical object (0 = clearly different objects, 1 = clearly the same). Use 1 for added or removed changes."},
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
        "required": ["object_label", "real_change", "real_change_reason", "same_object_probability",
                     "location_before", "location_after", "other_changes", "description"],
    },
}


def make_client() -> anthropic.Anthropic:
    """Reads ANTHROPIC_API_KEY. Keys not scoped to a workspace also need the workspace to bill."""
    workspace = os.environ.get("ANTHROPIC_WORKSPACE_ID")
    return anthropic.Anthropic(default_headers={"anthropic-workspace-id": workspace} if workspace else None)


def call_tool(client: anthropic.Anthropic, tool: dict, content, system: str | None = None) -> dict:
    """One request that must end in a call to `tool`; returns the tool input."""
    kwargs = {"system": system} if system else {}
    resp = client.messages.create(
        model=MODEL, max_tokens=4096, tools=[tool],
        # Forcing a specific tool is not supported on this model; the prompt requires it instead
        tool_choice={"type": "auto"},
        messages=[{"role": "user", "content": content}], **kwargs,
    )
    for block in resp.content:
        if block.type == "tool_use" and block.name == tool["name"]:
            return block.input
    raise RuntimeError(f"Claude did not call {tool['name']} (stop_reason={resp.stop_reason})")


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
        content.append({"type": "text", "text": f"{side} recording: close-up crops of the region (same 3D place in both recordings)"})
        content += [_image_block(p) for p in crops]
        content.append({"type": "text", "text": f"{side} recording: whole frame, region boxed in magenta"})
        content.append(_image_block(context))
    try:
        return call_tool(client, TOOL, content, SYSTEM)
    except RuntimeError as e:
        raise RuntimeError(f"{change['id']}: {e}") from None


def summarize(client: anthropic.Anthropic, changes: list[dict], quality: dict | None = None) -> str:
    lines = [f"- {c['type']}: {c['description']}" for c in changes]
    caveat = ""
    if quality and quality.get("verdict") != "good":
        caveat = (f"\n\nThe 3D reconstruction was rated {quality['verdict']}: " + "; ".join(quality["warnings"])
                  + ". Say so in the first sentence and keep the claims cautious.")
    resp = client.messages.create(
        model=MODEL, max_tokens=4096,  # the model thinks first; thinking counts toward max_tokens
        messages=[{"role": "user", "content": (
            "These changes were detected between two recordings of a room:\n" + "\n".join(lines)
            + caveat + "\n\nWrite a 2 to 3 sentence summary for a mobile robot, ending with what matters for "
              "navigating the room. Plain prose, no lists, no dashes.")}],
    )
    text = "".join(b.text for b in resp.content if b.type == "text").strip()
    if not text:
        raise RuntimeError(f"Empty summary from Claude (stop_reason={resp.stop_reason})")
    return text


def _split(change: dict) -> tuple[dict, dict]:
    """A rejected 'moved' pair as separate 'removed' and 'added' changes."""
    pair_only = ("distance_on_floor_m", "height_change_m", "match")
    removed = {k: v for k, v in change.items() if k not in ("to", "crops_after", *pair_only)}
    added = {k: v for k, v in change.items() if k not in ("from", "crops_before", *pair_only)}
    note = "visual check: the matched BEFORE and AFTER objects are different, so the move was split"
    return ({**removed, "type": "removed", "note": note}, {**added, "id": change["id"] + "b", "type": "added", "note": note})


def describe_all(changes_json: Path, min_same_probability: float = 0.5) -> dict:
    """Describe and check every change in changes.json; returns the enriched record."""
    record = json.loads(changes_json.read_text())
    client = make_client()
    out, rejected = [], []

    def keep(ch: dict, d: dict) -> None:
        if not d["real_change"]:
            rejected.append({**ch, **d})
            return
        out.append({**ch, **d})
        for j, extra in enumerate(d["other_changes"]):
            where = ch.get("to" if extra["type"] == "added" else "from")
            out.append({"id": f"{ch['id']}.{j + 1}", "type": extra["type"], "object_label": extra["object_label"],
                        "description": extra["note"], ("to" if extra["type"] == "added" else "from"): where,
                        "note": f"found by the visual check inside {ch['id']}; shares its 3D location"})

    for ch in record["changes"]:
        d = describe_change(client, ch)
        if ch["type"] == "moved" and d["real_change"] and d["same_object_probability"] < min_same_probability:
            for half in _split(ch):  # each half gets its own label and its own real change check
                keep(half, describe_change(client, half))
        else:
            keep(ch, d)
    record["changes"] = out
    record["rejected"] = rejected
    record["summary"] = summarize(client, out, record.get("reconstruction_quality")) if out else "No changes were confirmed."
    return record
