"""Turn corpus rows into SFT training examples.

There is one trap here that is worth stating plainly, because it silently
destroyed an earlier run of this pipeline.

SmolLM3's chat template has branches for `user`, `assistant` and `tool` only.
Corpus rows carry a fourth role, `twin`, holding the twin's stored hypothesis.
That role matches no branch, so the template **drops it without error**. A
builder that passes corpus rows through verbatim therefore renders a prompt
with *no assistant turn in it at all*, and SFT optimises a string containing
none of the text it is supposed to be learning. Loss goes down, nothing is
learnt, and the resulting adapter is indistinguishable from an untrained one.

So this module does not pass rows through. It builds an explicit two-part
example:

  prompt  = messages[0:2]        -- the system turn and the `code committed`
                                   tool turn, the same slice used at eval and
                                   serving time
  target  = one `assistant` turn -- the developer's own commit message

Renaming the `twin` role to `assistant` would also produce an assistant turn,
but the wrong one: that turn holds the twin's stored hypothesis, not the
developer's text. The target has to come from the developer.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator
from pathlib import Path

#: Roles SmolLM3's chat template actually renders. Anything else is dropped.
RENDERABLE_ROLES = frozenset({"system", "user", "assistant", "tool"})

#: Number of leading turns forming the prompt: the system turn and the tool
#: turn carrying the commit.
PROMPT_TURNS = 2

#: Empty reasoning block. The DPO/GRPO stages wrap their preferred completion
#: this way, so SFT teaches the same target shape rather than a second one.
THINK_PREFIX = "<think>\n</think>\n"


class CorpusError(ValueError):
    """A corpus row cannot be turned into a training example."""


def load_corpus(path: str | Path) -> Iterator[dict]:
    """Yield rows from a JSONL corpus file."""
    with Path(path).open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise CorpusError(f"{path}:{number}: not valid JSON") from exc


def extract_target(row: dict) -> str:
    """Return the developer's own text for this row.

    The corpus stores it twice -- as the final user turn and as
    `feedback.free_text`. They are required to agree: if they have drifted,
    it is not obvious which one the model should learn, and guessing would
    train on the wrong text silently.
    """
    feedback = (row.get("feedback") or {}).get("free_text", "")
    user_turns = [m for m in row.get("messages") or [] if m.get("role") == "user"]
    message_copy = user_turns[-1].get("content", "") if user_turns else ""

    if feedback.strip() and message_copy.strip() and feedback != message_copy:
        raise CorpusError(
            f"row {row.get('row_id')!r}: target copies disagree; "
            f"feedback.free_text and the final user turn must match"
        )

    target = feedback or message_copy
    if not target.strip():
        raise CorpusError(f"row {row.get('row_id')!r}: empty target")
    return target


def build_sft_example(row: dict, *, think_symmetry: bool = True) -> dict:
    """Build one SFT example: prompt turns plus a real assistant turn."""
    messages = row.get("messages") or []
    if len(messages) < PROMPT_TURNS:
        raise CorpusError(
            f"row {row.get('row_id')!r}: need at least {PROMPT_TURNS} prompt turns"
        )

    prompt = [dict(turn) for turn in messages[:PROMPT_TURNS]]
    for turn in prompt:
        if turn.get("role") not in RENDERABLE_ROLES:
            raise CorpusError(
                f"row {row.get('row_id')!r}: prompt turn has role "
                f"{turn.get('role')!r}, which the chat template drops"
            )

    target = extract_target(row)
    content = f"{THINK_PREFIX}{target}" if think_symmetry else target

    return {
        "id": row.get("row_id"),
        "messages": [*prompt, {"role": "assistant", "content": content}],
    }


def build_sft_dataset(
    rows: Iterable[dict], *, think_symmetry: bool = True
) -> Iterator[dict]:
    """Build SFT examples for every row."""
    for row in rows:
        yield build_sft_example(row, think_symmetry=think_symmetry)
