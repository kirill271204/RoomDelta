"""One self contained HTML page per run: verdict, summary, a card per change, figures and the score."""

import base64
import html
import io
import json
from pathlib import Path

from PIL import Image

TYPE_COLOR = {"moved": "var(--moved)", "added": "var(--added)", "removed": "var(--removed)"}


def _img(path, max_side: int = 520) -> str:
    """Image file as a data URI, downscaled so the page stays small enough to email."""
    if not path or not Path(path).exists():
        return ""
    im = Image.open(path).convert("RGB")
    im.thumbnail((max_side, max_side))
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=85)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def _load(path: Path):
    return json.loads(path.read_text()) if path.exists() else None


def _crop(change: dict, key: str, caption: str) -> str:
    crops = change.get(key)
    if not crops:
        return f'<figure class="crop empty"><div>not visible</div><figcaption>{caption}</figcaption></figure>'
    return f'<figure class="crop"><img src="{_img(crops[0])}" alt="{caption}"><figcaption>{caption}</figcaption></figure>'


def _card(c: dict) -> str:
    kind = c["type"]
    facts = []
    if kind == "moved":
        facts.append(f"moved {c['distance_on_floor_m']:.2f} m")
        if abs(c.get("height_change_m", 0)) >= 0.15:
            facts.append(f"{c['height_change_m']:+.2f} m in height")
    box = c.get("to") or c.get("from")
    if box:
        facts.append("size " + " × ".join(f"{v:.2f}" for v in box["size_m"]) + " m")
    if c.get("note", "").startswith("found by the visual check"):
        facts.append("found by the visual check, shares the 3D box of the object it rests on")
    before_caption = "before" if kind != "added" else "before: same place"
    after_caption = "after" if kind != "removed" else "after: same place"
    crops = "" if c.get("note", "").startswith("found by the visual check") else (
        f'<div class="crops">{_crop(c, "crops_before", before_caption)}{_crop(c, "crops_after", after_caption)}</div>')
    return f"""
    <article class="card">
      <header><span class="badge" style="background:{TYPE_COLOR.get(kind, 'var(--muted)')}">{kind.upper()}</span>
        <h3>{html.escape(c.get('object_label', c['id']))}</h3><span class="id">{c['id']}</span></header>
      <p>{html.escape(c.get('description', ''))}</p>
      <p class="facts">{' · '.join(html.escape(f) for f in facts)}</p>
      {crops}
    </article>"""


def write_report(out_dir: Path) -> Path:
    described = _load(out_dir / "changes_described.json")
    geometric = _load(out_dir / "changes.json") or {}
    record = described or geometric
    quality = _load(out_dir / "quality.json") or {"verdict": "unknown", "warnings": []}
    nav = _load(out_dir / "navigation.json")
    score = _load(out_dir / "score.json")
    changes = record.get("changes", [])

    verdict = quality["verdict"]
    warnings = "".join(f"<li>{html.escape(w)}</li>" for w in quality.get("warnings", []))
    summary = html.escape(record.get("summary", "")) if described else (
        "Descriptions were skipped (no Anthropic API key), so changes are listed from geometry only.")

    score_html = ""
    if score:
        rows = "".join(
            f"<tr><td>{html.escape(p['answer'])}</td><td>{p['answer_type']}</td><td>{html.escape(p['detected'])}</td>"
            f"<td>{'✅' if p['type_correct'] else '⚠️ type'}</td></tr>" for p in score["pairs"])
        rows += "".join(f"<tr><td>{html.escape(m)}</td><td></td><td>missed</td><td>❌</td></tr>"
                        for m in score["missed_changes"])
        rows += "".join(f"<tr><td></td><td></td><td>{html.escape(f['detected'])} ({f['type']})</td><td>false alarm</td></tr>"
                        for f in score["false_alarm_changes"])
        score_html = f"""
    <section><h2>Score against the answer key</h2>
      <div class="stats">
        <div><b>{score['found']} / {score['answer_key_changes']}</b><span>changes found</span></div>
        <div><b>{score['false_alarms']}</b><span>false alarms</span></div>
        <div><b>{score['type_correct']} / {score['found']}</b><span>type correct</span></div>
      </div>
      <table><thead><tr><th>Answer key</th><th>Type</th><th>Detected as</th><th></th></tr></thead><tbody>{rows}</tbody></table>
    </section>"""

    nav_html = ""
    if nav:
        route = ""
        if "route_before" in nav and nav["route_before"].get("reachable") is not None:
            rb, ra = nav["route_before"], nav["route_after"]
            fmt = lambda r: f"{r['length_m']} m" if r["reachable"] else "no route"
            route = f"<div><b>{fmt(rb)} → {fmt(ra)}</b><span>route to {html.escape(nav.get('goal_name', 'goal'))}</span></div>"
        nav_html = f"""
    <section><h2>What it means for a robot</h2>
      <div class="stats">
        <div><b>{nav['reachable_floor_before_m2']} → {nav['reachable_floor_after_m2']} m²</b><span>floor reachable from the start</span></div>
        <div><b>{nav['newly_blocked_floor_m2']} m²</b><span>newly blocked</span></div>
        <div><b>{nav['free_floor_cut_off_m2']} m²</b><span>still free but cut off</span></div>
        {route}
      </div>
      <img class="figure" src="{_img(out_dir / 'navmap.png', 1600)}" alt="Robot map before and after">
    </section>"""

    rejected = (described or {}).get("rejected", [])
    rejected_html = ""
    if rejected:
        items = "".join(f"<li><b>{html.escape(r.get('object_label', r['id']))}</b>: {html.escape(r['real_change_reason'])}</li>"
                        for r in rejected)
        rejected_html = f"""
    <section><details><summary>{len(rejected)} detection(s) rejected by the visual check</summary><ul>{items}</ul></details></section>"""

    title = f"{record.get('before', 'before')} vs {record.get('after', 'after')}"
    page = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>RoomDelta report</title>
<style>
:root {{ --bg:#f7f7f5; --panel:#ffffff; --text:#1d1d1f; --muted:#6b6b70; --line:#e3e3e0;
  --moved:#7b3fd1; --added:#1e9e4a; --removed:#d33b30; --good:#1e9e4a; --warn:#c77700; --bad:#d33b30; }}
@media (prefers-color-scheme: dark) {{ :root:not([data-theme="light"]) {{ --bg:#141416; --panel:#1e1e21;
  --text:#ececee; --muted:#9a9aa2; --line:#2e2e33; }} }}
:root[data-theme="dark"] {{ --bg:#141416; --panel:#1e1e21; --text:#ececee; --muted:#9a9aa2; --line:#2e2e33; }}
* {{ box-sizing:border-box; }}
body {{ margin:0; background:var(--bg); color:var(--text); font:15px/1.5 -apple-system, system-ui, Segoe UI, sans-serif; }}
main {{ max-width:1080px; margin:0 auto; padding:32px 16px 64px; }}
h1 {{ font-size:28px; margin:0 0 4px; }} h2 {{ font-size:19px; margin:0 0 12px; }} h3 {{ font-size:16px; margin:0; }}
.sub {{ color:var(--muted); margin:0 0 20px; }}
section {{ background:var(--panel); border:1px solid var(--line); border-radius:12px; padding:20px; margin:16px 0; }}
.verdict {{ display:inline-block; padding:3px 10px; border-radius:999px; color:#fff; font-weight:600; font-size:13px; }}
.cards {{ display:grid; grid-template-columns:repeat(auto-fill, minmax(320px, 1fr)); gap:16px; }}
.card {{ border:1px solid var(--line); border-radius:10px; padding:14px; }}
.card header {{ display:flex; align-items:center; gap:8px; margin-bottom:6px; }}
.card .id {{ margin-left:auto; color:var(--muted); font-size:12px; }}
.badge {{ color:#fff; font-size:11px; font-weight:700; padding:2px 8px; border-radius:6px; }}
.card p {{ margin:6px 0; }} .facts {{ color:var(--muted); font-size:13px; }}
.crops {{ display:grid; grid-template-columns:1fr 1fr; gap:8px; margin-top:8px; }}
.crop {{ margin:0; }} .crop img {{ width:100%; height:180px; object-fit:contain; background:var(--bg); border-radius:6px; }}
.crop.empty div {{ height:180px; display:grid; place-items:center; color:var(--muted); background:var(--bg); border-radius:6px; }}
figcaption {{ font-size:12px; color:var(--muted); text-align:center; }}
.stats {{ display:flex; flex-wrap:wrap; gap:12px; margin-bottom:14px; }}
.stats div {{ flex:1 1 160px; background:var(--bg); border-radius:8px; padding:10px 12px; }}
.stats b {{ display:block; font-size:20px; }} .stats span {{ color:var(--muted); font-size:13px; }}
.figure {{ width:100%; border-radius:8px; background:#fff; }}
table {{ width:100%; border-collapse:collapse; font-size:14px; }}
th, td {{ text-align:left; padding:6px 8px; border-bottom:1px solid var(--line); }}
ul {{ margin:8px 0 0; padding-left:20px; }} summary {{ cursor:pointer; font-weight:600; }}
</style></head>
<body><main>
  <h1>What changed in the room?</h1>
  <p class="sub">{html.escape(title)} · RoomDelta</p>

  <section>
    <h2>Summary <span class="verdict" style="background:var(--{'good' if verdict == 'good' else 'warn' if verdict == 'questionable' else 'bad'})">reconstruction {html.escape(verdict)}</span></h2>
    <p>{summary}</p>
    {f'<ul>{warnings}</ul>' if warnings else ''}
  </section>
  {score_html}
  <section><h2>{len(changes)} change{'s' if len(changes) != 1 else ''}</h2><div class="cards">{''.join(_card(c) for c in changes)}</div></section>
  {rejected_html}
  <section><h2>Top view</h2><img class="figure" src="{_img(out_dir / 'changes_topdown.png', 1600)}" alt="Changes from above"></section>
  {nav_html}
</main></body></html>
"""
    out = out_dir / "report.html"
    out.write_text(page)
    return out
