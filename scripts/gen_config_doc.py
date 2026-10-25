#!/usr/bin/env python
"""Regenerate the settings table in docs/CONFIGURATION.md from the live Settings model.

    python scripts/gen_config_doc.py            # rewrite the generated block in place
    python scripts/gen_config_doc.py --check    # exit 1 if docs are stale (CI)

The table is generated because a hand-copied config table is how documentation quietly becomes a
lie: the defaults here are read from pydantic *after* validation, so aliases and computed fields
match what the process actually uses.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

BEGIN = "<!-- BEGIN GENERATED: settings table (scripts/gen_config_doc.py) -->"
END = "<!-- END GENERATED -->"
DOC = ROOT / "docs" / "CONFIGURATION.md"

# Fields that need a human note beside the default. Keys are Settings attribute names.
NOTES = {
    "environment": "prod turns on the strict validator set (timeouts, secret hygiene, fail-fast policy load).",
    "database_url": "empty string disables persistence; moderation still runs, but nothing is auditable.",
    "bootstrap_api_keys": "comma separated; digested + salted with `api_key_pepper`, never stored raw.",
    "api_key_pepper": "different pepper = different digests, so a leaked DB cannot be rainbow-tabled.",
    "auth_enabled": "dev only. With it off, callers are `anonymous` and no FK is written.",
    "default_mode": "per-request `mode` overrides it; the preset only scales policy thresholds.",
    "policies_path": "hot-reloaded on mtime change; `POST /admin/reload-policies` forces it.",
    "max_input_chars": "422 beyond this - truncating an input to *allow* it would silently drop evidence.",
    "stage1_consensus": "`both` requires model agreement before a model-only signal blocks; `any` trusts either layer.",
    "stage1_fail_closed_on_llm_error": "if the judge is configured and errors, the row is blocked with `moderation_error`.",
    "stage2_allow_llm_rewrite": "the rewrite is re-scanned; a rewrite that still hits is redacted instead.",
    "stage3_mode": "`passthrough` moderates and forwards; `assistant` also generates the answer.",
    "store_raw_input": "false keeps hash + preview only. Trace and violations still reference excerpts.",
    "encrypt_stored_text": "at-rest encryption for stored text; the key comes from `api_key_pepper`.",
    "retention_days": "0 disables the sweep. `POST /admin/retention` runs it on demand.",
    "store_trace": "persist the per-stage trace that `/explain/{request_id}` renders.",
    "pipeline_request_timeout_s": "the ceiling: past it, the response is fail-closed blocked, never a pass.",
    "rate_limit_requests_per_minute": "per key, in-process. Scale with replicas, not `--workers`.",
    "hash_user_ids": "keeps `metadata.user_id` non-reversible in the audit row.",
    "log_request_bodies": "raw user text into application logs. Leave off.",
}


def _fmt(value: Any) -> str:
    if isinstance(value, str):
        return f"`{value}`" if len(value) < 60 else f"`{value[:57]}…`"
    if value is None:
        return "_unset_"
    if isinstance(value, (list, tuple, set)):
        return "`" + ", ".join(str(item) for item in value) + "`" if value else "_empty_"
    if isinstance(value, dict):
        return "`" + ", ".join(f"{k}={v}" for k, v in value.items()) + "`" if value else "_empty_"
    return f"`{value}`"


def render() -> str:
    from app.core.config import Settings

    fields = Settings.model_fields
    lines = [
        "| variable | default | meaning |",
        "| --- | --- | --- |",
    ]
    for name, field in fields.items():
        env = f"GUARD_{name.upper()}"
        default = field.default
        if callable(default) or str(default) == "PydanticUndefined":
            factory = getattr(field, "default_factory", None)
            default = factory() if factory else ""
        if "secret" in type(default).__name__.lower():  # SecretStr: show the masked form, never the value
            default = "••••"
        note = NOTES.get(name, "")
        if not note and field.description:
            note = field.description.rstrip(".")
        lines.append(f"| `{env}` | {_fmt(default)} | {note or '—'} |")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="fail if the generated block is stale")
    args = parser.parse_args(argv)

    table = render()
    if not DOC.exists():
        print(f"{DOC} does not exist - create it with the two marker lines", file=sys.stderr)
        return 1
    text = DOC.read_text(encoding="utf-8")
    if BEGIN not in text or END not in text:
        print("markers missing in CONFIGURATION.md", file=sys.stderr)
        return 1
    head, rest = text.split(BEGIN, 1)
    _old, tail = rest.split(END, 1)
    updated = f"{head}{BEGIN}\n{table}\n{END}{tail}"
    if args.check:
        if updated != text:
            print("docs/CONFIGURATION.md is stale - run: python scripts/gen_config_doc.py", file=sys.stderr)
            return 1
        print("config doc is current")
        return 0
    DOC.write_text(updated, encoding="utf-8")
    print(f"rewrote {len(table.splitlines()) - 2} rows in {DOC.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
