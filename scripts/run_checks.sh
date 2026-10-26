#!/usr/bin/env bash
# Everything CI runs, runnable locally in one command.
#
#   scripts/run_checks.sh            # static checks + tests + eval gate
#   QUICK=1 scripts/run_checks.sh    # skip the full-corpus eval (still runs the 40-row gate)
#
# Exits non-zero on the first real problem, but keeps going through the report so you see all of
# them in one pass - iterating on a moderation policy is slow enough as it is.
set -uo pipefail

cd "$(dirname "$0")/.."
QUICK="${QUICK:-0}"
status=0

# Pick an interpreter that can actually import the project. Silently running python3 and watching
# pydantic explode is the number-one way a check script gets ignored.
if [ -z "${PYTHON:-}" ]; then
  for candidate in "${VIRTUAL_ENV:+$VIRTUAL_ENV/bin/python}" "$(dirname "$0")/../.venv/bin/python" .venv/bin/python ../.venv/bin/python python3; do
    [ -x "$candidate" ] || candidate="$(command -v "$candidate" 2>/dev/null)" || continue
    [ -n "$candidate" ] && "$candidate" -c "import fastapi, pydantic, yaml, pytest" >/dev/null 2>&1 && { PYTHON="$candidate"; break; }
  done
fi
if [ -z "${PYTHON:-}" ]; then
  echo "no interpreter with the project dependencies found - activate your venv, or run:"
  echo "  PYTHON=/path/to/venv/bin/python scripts/run_checks.sh"
  exit 2
fi

step() { printf '\n\033[1m== %s ==\033[0m\n' "$1"; }
ok()   { printf '  \033[32mok\033[0m %s\n' "$1"; }
bad()  { printf '  \033[31mFAIL\033[0m %s\n' "$1"; status=1; }

step "python"
echo "  using $PYTHON"
"$PYTHON" --version || bad "python missing"

step "policies.yaml: schema, ids, regexes"
"$PYTHON" - <<'PY' || bad "policy file is not loadable"
import re, sys, yaml, pathlib
sys.path.insert(0, ".")
doc = yaml.safe_load(pathlib.Path("config/policies.yaml").read_text(encoding="utf-8"))
policies = doc["policies"]
ids = [p["id"] for p in policies]
assert len(ids) == len(set(ids)), "duplicate policy ids"
for policy in policies:
    assert policy.get("name") and policy.get("category"), f"{policy.get('id')} needs name+category"
    assert (policy.get("description") or "").strip(), f"{policy.get('id')} needs a description for /policies and the audit trail"
    for pat in ((policy.get("regex") or {}).get("patterns") or []) + list(policy.get("exceptions") or []):
        re.compile(pat, re.IGNORECASE)
    for lex in ((policy.get("lexicon") or {}).get("terms") or []):
        assert isinstance(lex, str) and lex.strip()
print(f"  {len(policies)} policies, version {doc['version']}, all patterns compile")
PY

step "imports"
"$PYTHON" -c "import app.main; app.main.create_app(); print('  app builds')" || bad "app.main does not import"

step "ruff (if installed)"
if "$PYTHON" -c "import ruff" 2>/dev/null || command -v ruff >/dev/null; then
  ruff check . || bad "ruff"
else
  echo "  skipped (pip install ruff to enable)"
fi

step "pytest"
"$PYTHON" -m pytest -q ${PYTEST_ARGS:-} || bad "tests"

step "eval gate"
if [ "$QUICK" = "1" ]; then
  "$PYTHON" scripts/evaluate.py --limit 40 --modes balanced --gate 0.97 || bad "quick eval gate"
else
  "$PYTHON" scripts/evaluate.py --limit 0 --modes balanced strict lenient --failures --json -o tests/data/eval_latest.json | tail -n 20
  "$PYTHON" scripts/evaluate.py --limit 0 --modes balanced --gate 0.99 || bad "full-corpus gate"
  "$PYTHON" scripts/evaluate.py --dataset tests/data/heldout_adversarial.jsonl --limit 0 --json -o tests/data/eval_heldout.json | tail -n 6
fi

step "result"
if [ "$status" -eq 0 ]; then echo "  all checks passed"; else echo "  see FAIL lines above"; fi
exit "$status"
