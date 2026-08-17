"""Release-document integrity checks and v0.20 onboarding regressions."""

from __future__ import annotations

import ast
import json
import re
from pathlib import Path

ROOT = Path(__file__).parent.parent
NOTEBOOK = ROOT / "notebooks" / "async_batch_llm_quickstart.ipynb"
TERMINAL_ASSET = ROOT / "docs" / "assets" / "v0.20-quickstart.gif"


def test_notebook_is_valid_safe_json_with_parseable_python() -> None:
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    assert notebook["nbformat"] == 4
    assert notebook["cells"]

    source = "\n".join(
        "".join(cell.get("source", []))
        for cell in notebook["cells"]
        if cell.get("cell_type") == "code"
    )
    assert "sk-" not in source
    assert "AIza" not in source
    assert not re.search(r"(?i)(?:api_key|token|secret)\s*=\s*['\"][^.'\"]{8,}", source)

    for cell in notebook["cells"]:
        if cell.get("cell_type") != "code":
            continue
        python_source = "\n".join(
            line
            for line in "".join(cell.get("source", [])).splitlines()
            if not line.lstrip().startswith(("%", "!"))
        )
        compile(
            python_source,
            str(NOTEBOOK),
            "exec",
            flags=ast.PyCF_ALLOW_TOP_LEVEL_AWAIT,
        )


def test_terminal_asset_is_portable_and_small() -> None:
    assert TERMINAL_ASSET.read_bytes().startswith((b"GIF87a", b"GIF89a"))
    assert TERMINAL_ASSET.stat().st_size < 2 * 1024 * 1024

    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert (
        "https://raw.githubusercontent.com/geoff-davis/async-batch-llm/main/"
        "docs/assets/v0.20-quickstart.gif"
    ) in readme
    assert (
        "https://colab.research.google.com/github/geoff-davis/async-batch-llm/blob/main/"
        "notebooks/async_batch_llm_quickstart.ipynb"
    ) in readme
    assert not re.search(r"!\[[^]]*]\((?!https://)", readme)


def test_primary_onboarding_uses_high_level_api() -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    docs_home = (ROOT / "docs" / "index.md").read_text(encoding="utf-8")
    getting_started = (ROOT / "docs" / "getting-started.md").read_text(encoding="utf-8")
    for content in (readme, docs_home, getting_started):
        assert 'llm("openai:gpt-4o-mini")' in content
        assert "process_prompts" in content
        assert "concurrency=" in content
        assert "summary()" in content
    assert "progress=True" in readme
    assert "ParallelBatchProcessor(" not in docs_home
    assert "PydanticAIStrategy(" not in docs_home


def test_comparison_and_troubleshooting_are_linked_and_navigable() -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    docs_home = (ROOT / "docs" / "index.md").read_text(encoding="utf-8")
    navigation = (ROOT / "mkdocs.yml").read_text(encoding="utf-8")
    for path in ("comparison.md", "troubleshooting.md"):
        assert (ROOT / "docs" / path).is_file()
        assert path in navigation
        assert path in docs_home
        assert path.removesuffix(".md") in readme


def test_release_history_and_migration_are_coherent() -> None:
    changelog = (ROOT / "CHANGELOG.md").read_text(encoding="utf-8")
    navigation = (ROOT / "mkdocs.yml").read_text(encoding="utf-8")
    migration = (ROOT / "docs" / "MIGRATION_V0_20.md").read_text(encoding="utf-8")
    historical = (ROOT / "docs" / "MIGRATION_V0_19.md").read_text(encoding="utf-8")
    migration_v022 = (ROOT / "docs" / "MIGRATION_V0_22.md").read_text(encoding="utf-8")

    assert "v0.19.0 was not published" in changelog
    assert "## [0.19" not in changelog
    assert "## [0.21.0] - 2026-07-26" in changelog
    assert "## [0.22.0] - 2026-08-17" in changelog
    assert "## [0.20.0] - 2026-07-20" in changelog
    assert "v0.18.x to v0.20.0" in migration
    assert "v0.19.0 was not published" in migration
    assert "MIGRATION_V0_20.md" in historical
    assert "v0.20 Migration: MIGRATION_V0_20.md" in navigation
    assert "Historical v0.19 Link: MIGRATION_V0_19.md" in navigation
    assert "v0.22 Migration: MIGRATION_V0_22.md" in navigation
    assert "v0.21.x to v0.22" in migration_v022
    assert "No checkpoint migration or invalidation is required" in migration_v022


def test_release_version_and_tag_workflow_agree() -> None:
    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    workflow = (ROOT / ".github" / "workflows" / "publish.yml").read_text(encoding="utf-8")
    assert re.search(r'^version = "0\.22\.0"$', pyproject, re.MULTILINE)
    assert '"v${PKG_VERSION}" != "${GITHUB_REF_NAME}"' in workflow


def test_source_does_not_claim_features_shipped_in_v019() -> None:
    stale = []
    for path in (ROOT / "src").rglob("*.py"):
        if "v0.19" in path.read_text(encoding="utf-8"):
            stale.append(path.relative_to(ROOT))
    assert not stale
