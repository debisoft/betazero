"""Tests for the corpus sanitiser.

These use synthetic rows rather than the real corpus: the point is to pin the
sanitiser's behaviour, and a test that reads the published corpus would pass
just as happily if the corpus were empty.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

# The sanitiser is a standalone script rather than a package module, so it is
# loaded by path instead of imported.
_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "sanitize_corpus.py"
_spec = importlib.util.spec_from_file_location("sanitize_corpus", _SCRIPT)
assert _spec and _spec.loader
sanitize_corpus = importlib.util.module_from_spec(_spec)
sys.modules["sanitize_corpus"] = sanitize_corpus
_spec.loader.exec_module(sanitize_corpus)


def make_row(
    diff: str, target: str = "fixed the thing", repo: str = "/home/ubuntu/cobweb2"
) -> dict:
    """A corpus row in the shape the real builder emits."""
    return {
        "row_id": "test-row",
        "dev_id": "D2",
        "repo": repo,
        "commit_sha": "0" * 40,
        "messages": [
            {"role": "system", "content": "You are an AI Twin of the user."},
            {"role": "tool", "content": diff},
            {"role": "twin", "content": "<think>\n</think>"},
            {"role": "user", "content": target},
        ],
        "feedback": {"free_text": target},
    }


def sanitize(row: dict) -> tuple[dict, dict]:
    counts: dict[str, int] = {}
    return sanitize_corpus.sanitize_row(row, counts), counts


@pytest.mark.parametrize(
    ("secret", "kind"),
    [
        ("brad.bass@ec.gc.ca", "email"),
        ("TEL: (416) 978-6285", "phone"),
        ("FAX: (416) 978-3884", "phone"),
        ("M5S 3E8", "postal"),
    ],
)
def test_contact_details_are_redacted(secret: str, kind: str) -> None:
    cleaned, counts = sanitize(make_row(f"credits line {secret} end"))
    blob = json.dumps(cleaned)
    assert secret not in blob
    assert counts.get(kind, 0) >= 1


def test_contributor_names_are_kept() -> None:
    """Attribution is the purpose of a credits screen; only contact details go."""
    cleaned, _ = sanitize(make_row('"Brad Bass", "brad.bass@ec.gc.ca"'))
    assert "Brad Bass" in json.dumps(cleaned)


def test_service_addresses_are_allowlisted() -> None:
    cleaned, counts = sanitize(make_row("remote: git@github.com:COBWEB-ca/cobweb2"))
    assert "git@github.com" in json.dumps(cleaned)
    assert counts.get("email", 0) == 0


def test_repo_path_is_canonicalised() -> None:
    cleaned, counts = sanitize(make_row("no secrets", repo="/home/ubuntu/cobweb2"))
    assert cleaned["repo"] == "COBWEB-ca/cobweb2"
    assert counts["repo_path"] == 1


def test_unrelated_repo_value_is_left_alone() -> None:
    cleaned, counts = sanitize(make_row("no secrets", repo="COBWEB-ca/cobweb2"))
    assert cleaned["repo"] == "COBWEB-ca/cobweb2"
    assert "repo_path" not in counts


def test_training_target_is_never_rewritten() -> None:
    """The guarantee that matters: redaction must not change what the model learns.

    A target containing an email would otherwise be silently rewritten, so the
    sanitiser refuses to produce output at all.
    """
    row = make_row("no secrets", target="fixed contact form for brad.bass@ec.gc.ca")
    with pytest.raises(SystemExit, match="altered the training target"):
        sanitize(row)


def test_clean_row_survives_untouched() -> None:
    row = make_row("harmless diff", repo="COBWEB-ca/cobweb2")
    cleaned, counts = sanitize(row)
    assert cleaned == row
    assert counts == {}
