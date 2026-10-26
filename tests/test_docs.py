"""Documentation is part of the product surface - so it gets tested.

Three bug classes these kill, all of which actually happened while writing these docs:

1. an env var named in a runbook that no code reads (`GUARD_API_KEY` vs the real
   `GUARD_BOOTSTRAP_API_KEYS`) - the reader's service silently boots with *no* auth keys or with
   defaults they did not choose;
2. a README metric that drifted from what the evaluator actually prints (the classic 100%-without-
   the-held-out-number mistake);
3. a relative link or an image asset that points at a file which no longer exists after a rename.

Also checks that the generated settings table in docs/CONFIGURATION.md is current.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "docs" / "assets"
GUARD_ENV = re.compile(r"\bGUARD_[A-Z][A-Z0-9_]*\b")

# Deliberately not Settings fields: compose plumbing and the UI's own target.
NON_SETTINGS = {
    "GUARD_PORT",
    "GUARD_UI_PORT",
    "GUARD_IMAGE_TAG",
    "GUARD_API_BASE",
    "GUARD_ENV",  # docker compose's own ${VAR:-} usage
}


def _settings_fields() -> set[str]:
    from app.core.config import Settings

    return {f"GUARD_{name.upper()}" for name in Settings.model_fields}


def _documented_env() -> list[tuple[Path, str]]:
    found: list[tuple[Path, str]] = []
    for path in [ROOT / "README.md", ROOT / "EVALUATION_REPORT.md", *sorted((ROOT / "docs").glob("*.md")), *sorted((ROOT / "deploy").glob("*.md"))]:
        for line in path.read_text(encoding="utf-8").splitlines():
            for name in GUARD_ENV.findall(line):
                found.append((path, name))
    for path in [ROOT / "deploy" / "docker-compose.yml", ROOT / "deploy" / ".env.example", ROOT / "deploy" / "Dockerfile", ROOT / ".github" / "workflows" / "ci.yml"]:
        for line in path.read_text(encoding="utf-8").splitlines():
            for name in GUARD_ENV.findall(line):
                found.append((path, name))
    return found


# --------------------------------------------------------------------------- env-var names


def test_every_guard_env_var_in_the_docs_is_real() -> None:
    """A typo'd setting is invisible: the loader ignores unknown names and the default wins."""
    valid = _settings_fields() | NON_SETTINGS
    bad = {(path.name, name) for path, name in _documented_env() if name not in valid}
    assert not bad, f"docs name settings that do not exist: {sorted(bad)}"


def test_compose_service_environment_matches_the_schema() -> None:
    """Same check for the env block of the file people actually copy."""
    compose = yaml.safe_load((ROOT / "deploy" / "docker-compose.yml").read_text(encoding="utf-8"))
    env = compose["services"]["api"]["environment"]
    fields = _settings_fields()
    unknown = []
    for key, value in env.items():
        if key not in fields:
            unknown.append(key)
        # a literal "${GUARD_ENV:-staging}" would set environment="staging" via a *different* name
        if isinstance(value, str) and "${" in value and f"{key[:-1].lower()}" in value and key.lower() not in value:
            unknown.append(f"{key}: placeholder references a different variable ({value})")
    assert not unknown, f"compose passes unknown settings: {unknown}"
    assert "GUARD_ENVIRONMENT" in env
    assert env["GUARD_ENVIRONMENT"] == "${GUARD_ENVIRONMENT:-staging}"


# --------------------------------------------------------------------------- claims in the docs


def test_readme_claims_about_the_policy_bundle_are_true() -> None:
    """The README quotes policy counts and stage names; a stale number there is a documentation bug."""
    from app.pipeline.policy import get_policy_bundle

    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    bundle = get_policy_bundle(ROOT / "config" / "policies.yaml")
    total = len(bundle.policies)
    summary = bundle.summary()
    counts = summary["by_stage"]
    assert summary["policies"] == total

    mentions = re.findall(r"(\d+)[^\n]{0,45}?\b(?:declarative\s+)?(?:policies|rules)\b", readme)
    assert str(total) in mentions, f"README never states the real policy count ({total}); found {mentions}"
    allowed = {str(total)} | {str(v) for v in counts.values()}
    strays = [m for m in mentions if m not in allowed]
    assert not strays, f"README claims policy counts that are not the file's: {strays}"

    for i in (1, 2, 3):
        assert re.search(rf"stage\s*{i}\b", readme, re.I), f"README no longer walks through stage {i}"
    assert "checksum" in readme.lower(), "policy hot-reload section should mention the checksum"


def test_docs_claim_the_real_measured_accuracy_pair() -> None:
    """A README that says "100% accuracy" without the held-out number next to it is misleading."""
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    report = (ROOT / "EVALUATION_REPORT.md").read_text(encoding="utf-8")
    for doc, text in (("README.md", readme), ("EVALUATION_REPORT.md", report)):
        numbers = re.findall(r"(\d{1,3}\.\d)\s?%", text)
        assert numbers, f"{doc} lost its measured numbers"
        if any(n.startswith("100") for n in numbers):
            assert any(n in {"72.7", "71.0", "55.0"} for n in numbers), f"{doc} quotes 100% without the held-out pair"


def test_readme_assets_exist_and_stay_light() -> None:
    """Images in a README are a download on every page view; keep them real and small."""
    required = {"hero.jpg": 900_000, "demo.gif": 2_000_000, "pipeline.svg": 60_000}
    for name, limit in required.items():
        path = ASSETS / name
        assert path.exists(), f"missing {path.relative_to(ROOT)} (regenerate the demo with scripts/render_demo_gif.py)"
        size = path.stat().st_size
        assert size <= limit, f"{name} is {size / 1024:.0f} KB, budget is {limit / 1024:.0f} KB"
        assert size > 4_000, f"{name} is suspiciously small ({size} B) - truncated or placeholder?"
    total = sum((ASSETS / n).stat().st_size for n in required)
    assert total < 3_500_000, f"README assets weigh {total / 1_000_000:.1f} MB in git"


def test_demo_gif_still_matches_the_api() -> None:
    """docs/assets/demo.gif is generated; the generator checks it against the envelope fields."""
    import subprocess
    import sys

    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "render_demo_gif.py"), "--check"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr


# --------------------------------------------------------------------------- links and tables


def test_relative_links_resolve() -> None:
    """Every markdown link to a repo file must point at something that exists."""
    checked = 0
    for path in [ROOT / "README.md", *sorted((ROOT / "docs").glob("*.md")), ROOT / "deploy" / "README.md", ROOT / "EVALUATION_REPORT.md"]:
        for line in path.read_text(encoding="utf-8").splitlines():
            for target in re.findall(r"\]\((?!https?:|#|mailto:)([^)\s#]+)", line):
                if "`" in line:  # inline code inside the link text, e.g. [`x.py`](app/x.py) - still check it
                    pass
                resolved = (path.parent / target).resolve()
                assert resolved.exists(), f"{path.relative_to(ROOT)} links to missing {target}"
                checked += 1
    assert checked > 25, f"only {checked} links found - did the doc layout change?"


def test_configuration_table_is_current() -> None:
    """docs/CONFIGURATION.md's generated block must match the live Settings class.

    Run with a deliberately polluted environment: the table documents *defaults*, so it must not move
    because CI or an operator exported GUARD_* vars. The first run of this repo's CI failed exactly
    that way - one generated default is an absolute path under the checkout.
    """
    import os
    import subprocess
    import sys

    env = {**os.environ, "GUARD_ENVIRONMENT": "dev", "GUARD_POLICIES_PATH": "/elsewhere/policies.yaml", "GUARD_LOG_LEVEL": "DEBUG"}
    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "gen_config_doc.py"), "--check"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=120,
        env=env,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_docs_carry_no_build_machine_paths() -> None:
    """A generated doc that embeds /home/<someone>/ is wrong for every other checkout."""
    prefixes = ("/home/", "/Users/", "/workspace/", "/actions-runner/", "/tmp/")
    offenders = []
    for path in [ROOT / "README.md", ROOT / "EVALUATION_REPORT.md", *sorted((ROOT / "docs").glob("*.md")), ROOT / "deploy" / "README.md"]:
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            hit = next((prefix for prefix in prefixes if prefix in line), None)
            if hit:
                offenders.append(f"{path.relative_to(ROOT)}:{number}: {line.strip()[:90]}")
    assert not offenders, "absolute build-machine paths in docs:\n" + "\n".join(offenders[:6])


def test_dockerfile_copies_only_files_that_exist() -> None:
    """`COPY tests/` broke a build after tests were added to .dockerignore - catch that statically."""
    df = (ROOT / "deploy" / "Dockerfile").read_text(encoding="utf-8")
    context = (ROOT / ".dockerignore").read_text(encoding="utf-8") if (ROOT / ".dockerignore").exists() else ""
    ignored = {line.strip() for line in context.splitlines() if line.strip() and not line.startswith("!")}
    for source in re.findall(r"^COPY\s+(?:--from=\S+\s+)?([^\n]+)$", df, re.M):
        for token in source.split():
            if token.startswith("--") or "=" in token:
                continue
            assert (ROOT / token).exists() or (ROOT / "deploy" / token).exists(), f"Dockerfile COPYs missing path {token!r}"
            if token in {"tests", "tests/"}:
                assert "tests" not in ignored, "Dockerfile wants tests/ but .dockerignore excludes it"


@pytest.mark.parametrize("name", ["README.md", "EVALUATION_REPORT.md"])
def test_no_dangling_todo_in_shipped_docs(name: str) -> None:
    text = (ROOT / name).read_text(encoding="utf-8")
    assert "coming in a later" not in text
    assert not re.search(r"\bTODO\b", text), "shipped docs should not contain TODO"
