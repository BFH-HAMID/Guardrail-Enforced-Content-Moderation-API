"""Documentation is part of the product surface - so it gets tested.

Two bug classes this kills, both of which actually happened while writing these docs:

1. an env var named in a runbook that no code reads (`GUARD_API_KEY` vs the real
   `GUARD_BOOTSTRAP_API_KEYS`) - the reader's service silently boots with *no* auth keys or with
   defaults they did not choose;
2. a policy count / claim in README drifting away from ``policies.yaml``.

Also checks that every relative link in the top-level docs resolves to a real file, and that the
generated settings table in docs/CONFIGURATION.md is current.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
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


def _doc_files() -> list[Path]:
    files = [ROOT / "README.md", ROOT / "EVALUATION_REPORT.md"]
    files += sorted((ROOT / "docs").glob("*.md"))
    files += sorted((ROOT / "deploy").glob("*"))
    files += [ROOT / ".github" / "workflows" / "ci.yml", ROOT / ".dockerignore", ROOT / "deploy" / ".env.example"]
    return [f for f in files if f.is_file()]


def test_every_guard_env_var_in_the_docs_is_real() -> None:
    known = _settings_fields() | NON_SETTINGS
    unknown: dict[str, list[str]] = {}
    for path in _doc_files():
        for name in set(GUARD_ENV.findall(path.read_text(encoding="utf-8"))):
            if name not in known:
                unknown.setdefault(name, []).append(str(path.relative_to(ROOT)))
    assert not unknown, f"docs reference settings that do not exist: {unknown}"


def test_compose_service_environment_matches_the_schema() -> None:
    compose = yaml.safe_load((ROOT / "deploy" / "docker-compose.yml").read_text(encoding="utf-8"))
    api = compose["services"]["api"]
    assert api["build"]["context"] == ".." and api["build"]["dockerfile"] == "deploy/Dockerfile"
    assert api["depends_on"]["db"]["condition"] == "service_healthy"
    for key, value in api["environment"].items():
        assert key in _settings_fields(), f"compose sets {key}, which Settings does not read"
        if isinstance(value, str):
            for match in re.finditer(r"\$\{([A-Z_]+)(?::\?[^}]*)?\}", value):
                assert match.group(1) not in {"GUARD_API_KEY"}, "the bootstrap key var is GUARD_BOOTSTRAP_API_KEYS"
    # the UI profile must not need its own image build
    assert compose["services"]["ui"]["image"] == api["image"]


def test_readme_claims_about_the_policy_bundle_are_true() -> None:
    from app.core.config import get_settings
    from app.pipeline.policy import get_policy_bundle

    bundle = get_policy_bundle(get_settings().policies_file)
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert f"{len(bundle.policies)} declarative policies" in readme or f"{len(bundle.policies)}\n" in readme
    assert f"{len(bundle.policies)} in [`config/policies.yaml`]" in readme, "README's policy count drifted from the file"
    assert f"`stage1` / `stage2` / `stage3_output`" in readme
    # the corpus sizes quoted in README must match the committed data
    tuned = (ROOT / "tests" / "data" / "moderation_testset.jsonl").read_text(encoding="utf-8").splitlines()
    held = (ROOT / "tests" / "data" / "heldout_adversarial.jsonl").read_text(encoding="utf-8").splitlines()
    assert str(len([line for line in tuned if line.strip()])) in readme
    assert str(len([line for line in held if line.strip()])) in readme


def test_relative_links_resolve() -> None:
    broken = []
    for path in _doc_files():
        if path.suffix != ".md":
            continue
        for target in re.findall(r"\]\((?!https?:|#|mailto:)([^)\s#]+)", path.read_text(encoding="utf-8")):
            resolved = (path.parent / target).resolve()
            if not resolved.exists():
                broken.append(f"{path.relative_to(ROOT)} -> {target}")
    assert not broken, f"broken doc links: {broken}"


def test_configuration_table_is_current() -> None:
    """`docs/CONFIGURATION.md` embeds a generated block; a stale one is worse than none."""
    import subprocess
    import sys

    result = subprocess.run(  # noqa: S603
        [sys.executable, str(ROOT / "scripts" / "gen_config_doc.py"), "--check"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_dockerfile_copies_only_files_that_exist() -> None:
    """A COPY of a path that was renamed breaks the image build, which CI catches late and loudly."""
    dockerfile = (ROOT / "deploy" / "Dockerfile").read_text(encoding="utf-8")
    missing = []
    for line in dockerfile.splitlines():
        match = re.match(r"^COPY\s+(?:--from=\S+\s+)?(\S+)\s+(\S+)$", line.strip())
        if not match:
            continue
        src = match.group(1)
        if "*" in src or src.startswith("$"):
            continue
        if not (ROOT / src).exists():
            missing.append(src)
    assert not missing, f"Dockerfile COPYs paths that do not exist: {missing}"


def test_no_dangling_todo_in_shipped_docs() -> None:
    offenders = []
    for path in _doc_files():
        if path.suffix != ".md":
            continue
        text = path.read_text(encoding="utf-8")
        for line_no, line in enumerate(text.splitlines(), 1):
            if re.search(r"\b(TODO|FIXME|XXX|coming soon|TBD)\b", line) and "TBD" not in path.name:
                offenders.append(f"{path.relative_to(ROOT)}:{line_no}: {line.strip()[:70]}")
    assert not offenders, "unresolved placeholders in user-facing docs:\n" + "\n".join(offenders)
