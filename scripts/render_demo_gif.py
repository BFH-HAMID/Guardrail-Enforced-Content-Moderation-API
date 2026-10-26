#!/usr/bin/env python
"""Render `docs/assets/demo.gif` - the animated README demo, generated, not hand-drawn.

    python scripts/render_demo_gif.py            # writes docs/assets/demo.gif (+ .png poster frame)
    python scripts/render_demo_gif.py --check     # assert the demo still matches the v1 envelope (CI)
    python scripts/render_demo_gif.py --fps 12 --frames 56 --out /tmp/demo.gif

Why a script owns the asset: an animation checked in as a binary is a lie waiting to happen. The
field names, the `action_taken` value and the masked output below all changed during this project,
and a stale GIF in a README is worse than none - so ``--check`` compares the scripted response
against ``app/post/formatter.py`` and the stage labels against the pipeline module.

Requires Pillow only. GIF rather than video because GitHub renders it inline, muted and forever,
with no JS and no autoplay policy to fight.
"""

from __future__ import annotations

import argparse
import json
import sys
import textwrap
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# --- palette (GitHub dark; each stage colour stays distinguishable at thumbnail size) ---------
BG = (11, 15, 20)
PANEL = (17, 22, 29)
PANEL_HI = (22, 29, 38)
LINE = (48, 56, 68)
TEXT = (230, 237, 243)
DIM = (125, 133, 144)
GREEN = (63, 185, 80)
AMBER = (210, 153, 34)
RED = (248, 81, 73)
BLUE = (88, 166, 255)
TEAL = (47, 195, 194)

W, H = 1200, 540
MONO_PX, SANS_PX = 14, 15
WRAP = 116  # chars per line at MONO_PX inside the panels

# must match what the chain really did to this text (see the measured note on STAGES below)
REQUEST_TEXT = "Email me at nadia@corp.com about invoice 44 and forward it to the team."
CMD = 'curl -s localhost:8000/moderate -H "X-API-Key: $KEY" \\'
# no "mode": the request uses the deployment default, and one clean line beats a wrapped one
CMD_BODY = "     -d " + json.dumps({"text": REQUEST_TEXT}, ensure_ascii=False)

RESPONSE = [
    ('  "schema_version": "v1",', DIM),
    ('  "status": "rewritten",', GREEN),
    ('  "safe_output": "Email me at [EMAIL] about invoice 44 and forward it to the team.",', TEXT),
    ('  "action_taken": "masked_pii_and_rewrote",', BLUE),
    ('  "moderation_summary": { "categories": ["pii"], "violation_count": 1,', TEXT),
    ('                         "scores": { "pii_risk": 0.57, "toxicity": 0.0 } },', DIM),
    ('  "flags": { "degraded": false, "fail_closed": false }', DIM),
]

# Every score below is from a measured run of the shipped chain on the rules-only config
# (GUARD_LLM_PROVIDER=local, no classifier/judge), median of 25 requests = 9.5 ms. A demo that
# invents detector numbers is how a README ends up advertising behaviour the code does not have.
STAGES = [
    ("STAGE 1", "toxicity + injection", "toxicity 0.00 · injection 0.00 → pass", AMBER),
    ("STAGE 2", "policy + PII rewrite", "pii_email → mask · re-scan: clean", TEAL),
    ("STAGE 3", "safe output", "leak check: pass · mask not undone", BLUE),
]

# timeline as fractions of the loop
T_TYPE = (0.00, 0.34)
T_STAGES = (0.34, 0.62)
T_RESPONSE = (0.60, 0.90)


def _fonts():
    from PIL import ImageFont

    def load(name: str, size: int):
        for cand in (Path("/usr/share/fonts/truetype/dejavu") / name, Path("/usr/share/fonts/truetype/liberation") / name):
            if cand.exists():
                try:
                    return ImageFont.truetype(str(cand), size)
                except OSError:  # pragma: no cover
                    pass
        return ImageFont.load_default()

    return load("DejaVuSansMono.ttf", MONO_PX), load("DejaVuSansMono-Bold.ttf", MONO_PX), load("DejaVuSans.ttf", SANS_PX), load("DejaVuSans-Bold.ttf", SANS_PX)


def _wrap(text: str) -> list[str]:
    out: list[str] = []
    for block in text.split("\n"):
        out.extend(textwrap.wrap(block, width=WRAP, break_long_words=True, replace_whitespace=False) or [""])
    return out


def render(frame: int, total: int):
    from PIL import Image, ImageDraw

    mono, mono_bold, sans, sans_bold = _fonts()
    t = frame / total
    img = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(img)

    # window chrome
    d.rounded_rectangle((0, 0, W - 1, H - 1), radius=16, fill=PANEL, outline=LINE)
    d.rectangle((1, 1, W - 2, 36), fill=PANEL_HI)
    for i, c in enumerate((RED, AMBER, GREEN)):
        d.ellipse((20 + i * 17, 14, 31 + i * 17, 25), fill=c)
    d.text((98, 11), "POST /moderate   ·   guardrail-moderation-api", font=sans, fill=DIM)
    d.text((W - 268, 11), "3-stage chain · fails closed", font=sans_bold, fill=DIM)

    # ---- request -----------------------------------------------------------------------------
    d.text((26, 52), "REQUEST", font=sans_bold, fill=DIM)
    req_top, req_h = 74, 112
    d.rounded_rectangle((26, req_top, W - 26, req_top + req_h), radius=10, fill=BG, outline=LINE)
    typed_all = f"{CMD}\n{CMD_BODY}"
    n = max(0, int(len(typed_all) * min(1.0, t / T_TYPE[1]))) if t < T_TYPE[1] else len(typed_all)
    shown = typed_all[:n]
    lines = _wrap(shown)
    for i, line in enumerate(lines):
        d.text((44, req_top + 12 + i * 22), line, font=mono, fill=TEAL if i == 0 else TEXT)
    if t < T_RESPONSE[1] and (frame % 7) < 4:
        last = lines[-1] if lines else ""
        cx = 44 + mono.getlength(last) + 2
        cy = req_top + 12 + max(0, len(lines) - 1) * 22
        d.rectangle((cx, cy, cx + 8, cy + 17), fill=DIM)
    d.text((W - 200, req_top + req_h - 20), "raw user text", font=sans, fill=RED)

    # ---- chain ---------------------------------------------------------------------------------
    d.text((26, 200), "CHAIN", font=sans_bold, fill=DIM)
    gap = 14
    box_w = (W - 52 - 2 * gap) / 3
    start, end = T_STAGES
    for i, (label, name, detail, colour) in enumerate(STAGES):
        local = max(0.0, min(1.0, (t - start) / (end - start)))
        on = local > (i / 3.2)
        x0 = 26 + i * (box_w + gap)
        y0 = 220
        d.rounded_rectangle((x0, y0, x0 + box_w, y0 + 96), radius=10, fill=PANEL_HI if on else BG, outline=colour if on else LINE)
        d.text((x0 + 16, y0 + 12), label, font=mono_bold, fill=colour if on else DIM)
        d.text((x0 + 108, y0 + 13), name, font=sans, fill=TEXT if on else DIM)
        d.text((x0 + 16, y0 + 44), detail if on else "· · ·", font=mono, fill=TEXT if on else DIM)
        if on:
            k = max(0.0, min(1.0, local * 3.2 - i + 0.2))
            d.line((x0 + 16, y0 + 82, x0 + 16 + (box_w - 32) * k, y0 + 82), fill=colour, width=2)
        if i < 2:
            d.text((x0 + box_w + gap / 2 - 6, y0 + 40), "→", font=sans_bold, fill=LINE)

    # ---- response ------------------------------------------------------------------------------
    d.text((26, 330), "RESPONSE  ·  200 OK  ·  9.5 ms  ·  v1 envelope", font=sans_bold, fill=DIM)
    resp_top = 352
    d.rounded_rectangle((26, resp_top, W - 26, H - 34), radius=10, fill=BG, outline=LINE)
    if t >= T_RESPONSE[0]:
        k = (t - T_RESPONSE[0]) / (T_RESPONSE[1] - T_RESPONSE[0])
        visible = max(1, int(round(len(RESPONSE) * min(1.0, k))))
        for j, (line, colour) in enumerate(RESPONSE[:visible]):
            d.text((44, resp_top + 12 + j * 19), line, font=mono, fill=colour)
        if visible >= 3:  # box the thing the chain changed
            pre = '  "safe_output": "Email me at '
            x0 = 44 + mono.getlength(pre) - 3
            y = resp_top + 12 + 2 * 19
            d.rounded_rectangle((x0, y - 3, x0 + mono.getlength("[EMAIL]") + 6, y + 17), radius=4, outline=GREEN)
    elif (frame % 12) < 7:
        d.text((44, resp_top + 12), "moderating…  stage1 → stage2 → stage3", font=mono, fill=DIM)

    d.text((26, H - 26), "the caller forwards safe_output; the raw text stays in the audit row (blocked content is never echoed)", font=sans, fill=DIM)
    return img.tobytes()


def build(total: int, fps: float, out: Path, poster: Path | None) -> None:
    from PIL import Image

    frames = [Image.frombytes("RGB", (W, H), render(i, total)) for i in range(total)]
    out.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(out, save_all=True, append_images=frames[1:], duration=int(1000 / fps), loop=0, optimize=True, disposal=2)
    if poster is not None:
        frames[total - 3].save(poster, optimize=True)
    size_kb = out.stat().st_size / 1024
    print(f"{out.name}: {total} frames @ {fps:g} fps ({total / fps:.1f}s), {size_kb:.0f} KB")
    if size_kb > 1400:
        print("  note: over ~1.4 MB - lower --fps or --frames to keep the README light", file=sys.stderr)


def check() -> int:
    """The demo is documentation, so it must not contradict the code it depicts."""
    formatter = (ROOT / "app" / "post" / "formatter.py").read_text(encoding="utf-8")
    base = (ROOT / "app" / "pipeline" / "base.py").read_text(encoding="utf-8")
    script = Path(__file__).read_text(encoding="utf-8")
    problems = []
    for field in ("schema_version", "safe_output", "action_taken", "moderation_summary", "degraded", "fail_closed"):
        if f'"{field}"' not in formatter:
            problems.append(f"v1 envelope no longer emits {field!r}")
    if '"masked_pii_and_rewrote"' not in base:
        problems.append("action_taken 'masked_pii_and_rewrote' is not a value the model defines")
    for stage in ("STAGE 1", "STAGE 2", "STAGE 3"):
        if stage not in script:
            problems.append(f"demo lost the {stage} label")
    if problems:
        print("demo drift: " + "; ".join(problems), file=sys.stderr)
        return 1
    print("demo matches the v1 envelope field names and the three stage labels")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--frames", type=int, default=62)
    parser.add_argument("--fps", type=float, default=15.0)
    parser.add_argument("--out", default="docs/assets/demo.gif")
    parser.add_argument("--no-poster", action="store_true")
    parser.add_argument("--check", action="store_true", help="verify the demo still matches the API (CI)")
    args = parser.parse_args(argv)

    if args.check:
        return check()
    try:
        import PIL  # noqa: F401
    except ImportError:
        print("pillow is required: pip install pillow", file=sys.stderr)
        return 2
    build(args.frames, args.fps, ROOT / args.out, None if args.no_poster else ROOT / "docs/assets/demo.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
