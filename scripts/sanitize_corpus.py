"""Produce a publishable copy of the D2 corpus.

The corpus is derived entirely from the public COBWEB-ca/cobweb2 git history:
every row's input is a commit diff and every row's target is the commit message
the developer wrote. Two things in it are still not fit to publish:

1.  Third-party contact details. cobweb2's credits/About screen is source code,
    so a handful of diffs carry real people's email, phone, fax and postal
    address. Those details are already public in the upstream repo, but
    republishing them inside a training corpus -- and baking them into
    published model weights -- is a different act, so they are redacted here.
    Contributor *names* are deliberately kept: attribution in an open-source
    credits screen is the point of that screen.

2.  The `repo` field records the training VM's filesystem path
    (`/home/ubuntu/cobweb2`) rather than the project identity. That is
    infrastructure detail leaking into a published artifact.

Targets (the commit messages BetaZero is trained to produce) are verified
untouched: redaction must not silently rewrite training labels.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from typing import Any

REPO_PATH_RE = re.compile(r"^/home/[^/]+/cobweb2/?$")
CANONICAL_REPO = "COBWEB-ca/cobweb2"

# Service/bot addresses that identify no individual and carry no contact risk.
EMAIL_ALLOWLIST = frozenset({"git@github.com"})

REDACTIONS: tuple[tuple[str, re.Pattern[str], str], ...] = (
    # `.this` variants appear in the upstream source verbatim; match them too so
    # a mangled address is redacted rather than passed through as "clean".
    (
        "email",
        re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}"),
        "<redacted-email>",
    ),
    (
        "phone",
        re.compile(
            r"(?i)\b(?:TEL|FAX|PHONE)\b[:\s]*\(?\d{3}\)?[\s.-]?\d{3}[\s.-]?\d{4}"
        ),
        "<redacted-phone>",
    ),
    (
        "postal",
        re.compile(r"\b[A-Z]\d[A-Z]\s?\d[A-Z]\d\b"),
        "<redacted-postal>",
    ),
)


def redact_text(text: str, counts: dict[str, int]) -> str:
    """Apply every redaction rule to one string, tallying what fired."""
    for name, pattern, placeholder in REDACTIONS:
        # Bind the loop variables as defaults: a bare closure over `name` and
        # `placeholder` happens to work only because `sub` runs inside this
        # iteration, which is exactly the kind of accident that breaks later.
        def _sub(
            match: re.Match[str],
            name: str = name,
            placeholder: str = placeholder,
        ) -> str:
            if name == "email" and match.group(0) in EMAIL_ALLOWLIST:
                return match.group(0)
            counts[name] = counts.get(name, 0) + 1
            return placeholder

        text = pattern.sub(_sub, text)
    return text


def walk(node: Any, counts: dict[str, int]) -> Any:
    """Recursively redact every string in a decoded JSON value."""
    if isinstance(node, str):
        return redact_text(node, counts)
    if isinstance(node, list):
        return [walk(item, counts) for item in node]
    if isinstance(node, dict):
        return {key: walk(value, counts) for key, value in node.items()}
    return node


def sanitize_row(row: dict, counts: dict[str, int]) -> dict:
    """Redact one row and canonicalise its `repo` field."""
    target_before = row["feedback"]["free_text"]

    cleaned = walk(row, counts)

    repo = cleaned.get("repo")
    if isinstance(repo, str) and REPO_PATH_RE.match(repo):
        cleaned["repo"] = CANONICAL_REPO
        counts["repo_path"] = counts.get("repo_path", 0) + 1

    # A redaction that rewrites a training label would silently change what the
    # model learns, so treat it as a hard failure rather than a warning.
    if cleaned["feedback"]["free_text"] != target_before:
        raise SystemExit(
            f"refusing to write: redaction altered the training target of row "
            f"{row.get('row_id')!r}"
        )
    return cleaned


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", help="raw D2_corpus.jsonl")
    parser.add_argument("dest", help="sanitized output path")
    args = parser.parse_args()

    counts: dict[str, int] = {}
    rows_changed = 0
    total = 0

    with (
        open(args.source, encoding="utf-8") as src,
        open(args.dest, "w", encoding="utf-8") as out,
    ):
        for line in src:
            line = line.strip()
            if not line:
                continue
            total += 1
            row = json.loads(line)
            before = json.dumps(row, sort_keys=True)
            cleaned = sanitize_row(row, counts)
            if json.dumps(cleaned, sort_keys=True) != before:
                rows_changed += 1
            out.write(json.dumps(cleaned, ensure_ascii=False) + "\n")

    print(f"rows read:     {total}")
    print(f"rows modified: {rows_changed}")
    for name in sorted(counts):
        print(f"  {name:10s} {counts[name]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
