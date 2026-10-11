"""Score a run against a hand written answer key.

The answer key is in words ("baby seat moved to the kitchen counter") while the
output has Claude's labels and 3D positions ("grey baby high chair, moved 1.1 m"),
so plain string matching fails. Claude pairs each detected change with at most one
answer key entry; the counting is done here, in code:

    found          answer key entries matched by a detected change
    missed         answer key entries nothing matched
    false alarms   detected changes that match no answer key entry
    type correct   matched pairs where moved / added / removed agrees
"""

import json
from pathlib import Path

from .describe import call_tool, make_client

TOOL = {
    "name": "report_matching",
    "description": "Pair detected changes with answer key entries.",
    "input_schema": {
        "type": "object",
        "properties": {
            "pairs": {
                "type": "array",
                "description": "One entry per answer key item that a detected change corresponds to. Each answer key "
                               "index and each change id may appear at most once.",
                "items": {
                    "type": "object",
                    "properties": {
                        "answer_index": {"type": "integer"},
                        "change_id": {"type": "string"},
                        "reason": {"type": "string", "description": "Why they are the same change, in a few words."},
                    },
                    "required": ["answer_index", "change_id", "reason"],
                },
            },
        },
        "required": ["pairs"],
    },
}

PROMPT = """An automatic system compared two recordings of a room and reported the detected changes below. A person \
wrote the answer key of what they actually changed. Pair each answer key entry with the detected change that \
describes the same real event, if there is one.

People and the system name objects differently ("bin" vs "wastebasket", "office chair" vs "gaming chair", a \
container named after what stands in it), so match on what the object is, where it was and where it went. Do not \
pair something just because the type matches. Leave an answer key entry unpaired if no detected change fits, and \
leave detected changes unpaired if they match nothing. Call report_matching once.

ANSWER KEY
{answer}

DETECTED CHANGES
{detected}"""


def _detected(out_dir: Path) -> list[dict]:
    described = out_dir / "changes_described.json"
    if described.exists():
        return json.loads(described.read_text())["changes"]
    # Without descriptions there are no labels; the geometry alone cannot be matched to words
    raise SystemExit(f"{described} not found: scoring needs the Claude descriptions (run without --no-describe)")


def score(out_dir: Path, answer_key: Path) -> dict:
    truth = json.loads(answer_key.read_text())["changes"]
    detected = _detected(out_dir)
    answer_lines = "\n".join(
        f"[{i}] {t['type'].upper()}: {t['object']}"
        + (f", from {t['from']}" if t.get("from") else "") + (f", to {t['to']}" if t.get("to") else "")
        for i, t in enumerate(truth))
    detected_lines = "\n".join(
        f"[{c['id']}] {c['type'].upper()}: {c['object_label']}. {c['description']}" for c in detected)
    result = call_tool(make_client(), TOOL, PROMPT.format(answer=answer_lines, detected=detected_lines))

    # Enforce one to one pairing and valid references, whatever the model returned
    by_id = {c["id"]: c for c in detected}
    pairs, used_answers, used_changes = [], set(), set()
    for p in result["pairs"]:
        i, cid = p["answer_index"], p["change_id"]
        if not (0 <= i < len(truth)) or cid not in by_id or i in used_answers or cid in used_changes:
            continue
        used_answers.add(i)
        used_changes.add(cid)
        pairs.append({"answer": truth[i]["object"], "answer_type": truth[i]["type"], "change_id": cid,
                      "detected": by_id[cid]["object_label"], "detected_type": by_id[cid]["type"],
                      "type_correct": truth[i]["type"] == by_id[cid]["type"], "reason": p["reason"]})

    missed = [t["object"] for i, t in enumerate(truth) if i not in used_answers]
    false_alarms = [{"change_id": c["id"], "detected": c["object_label"], "type": c["type"]}
                    for c in detected if c["id"] not in used_changes]
    n_found = len(pairs)
    return {
        "answer_key_changes": len(truth),
        "detected_changes": len(detected),
        "found": n_found,
        "missed": len(missed),
        "false_alarms": len(false_alarms),
        "type_correct": sum(p["type_correct"] for p in pairs),
        "recall": round(n_found / len(truth), 3) if truth else None,
        "precision": round(n_found / len(detected), 3) if detected else None,
        "pairs": pairs,
        "missed_changes": missed,
        "false_alarm_changes": false_alarms,
    }


def print_score(s: dict) -> None:
    print(f"found {s['found']} of {s['answer_key_changes']} | missed {s['missed']} | false alarms {s['false_alarms']} "
          f"| type correct {s['type_correct']} of {s['found']}")
    for p in s["pairs"]:
        mark = "ok" if p["type_correct"] else "WRONG TYPE"
        print(f"  found   {p['answer']!r} ({p['answer_type']}) as {p['change_id']} {p['detected']!r} "
              f"({p['detected_type']}, {mark})")
    for m in s["missed_changes"]:
        print(f"  MISSED  {m!r}")
    for f in s["false_alarm_changes"]:
        print(f"  FALSE   {f['change_id']} {f['detected']!r} ({f['type']})")
