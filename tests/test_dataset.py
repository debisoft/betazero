"""Tests for the SFT dataset builder.

The load-bearing test here is `test_target_survives_a_smollm3_style_template`.
Everything else checks shape; that one checks the property whose absence
silently trained an earlier adapter on nothing at all.
"""

from __future__ import annotations

import json

import pytest

from betazero.dataset import (
    THINK_PREFIX,
    CorpusError,
    build_sft_dataset,
    build_sft_example,
    extract_target,
    load_corpus,
)


def make_row(target: str = "Class Path in Manifest", row_id: str = "r1") -> dict:
    return {
        "row_id": row_id,
        "repo": "COBWEB-ca/cobweb2",
        "messages": [
            {"role": "system", "content": "You are an AI Twin of the user."},
            {"role": "tool", "content": "code committed\n<code>diff...</code>"},
            {"role": "twin", "content": "<think>\n</think>"},
            {"role": "user", "content": target},
        ],
        "feedback": {"free_text": target},
    }


def render_like_smollm3(messages: list[dict]) -> str:
    """A stand-in for SmolLM3's chat template.

    The point is its *omission*: it has branches for system, user, assistant
    and tool, and silently drops anything else -- exactly the behaviour that
    made the previous builder render zero assistant turns. Using a stub keeps
    the test hermetic; using the real tokeniser would test the network.
    """
    parts = []
    for message in messages:
        role = message.get("role")
        if role in {"system", "user", "assistant", "tool"}:
            parts.append(f"<|im_start|>{role}\n{message['content']}<|im_end|>")
    return "\n".join(parts)


def test_target_survives_a_smollm3_style_template() -> None:
    """The regression that matters: the rendered example must contain the
    developer's text in an assistant turn."""
    example = build_sft_example(make_row("Class Path in Manifest"))
    rendered = render_like_smollm3(example["messages"])

    assert "<|im_start|>assistant" in rendered
    assert "Class Path in Manifest" in rendered.split("<|im_start|>assistant")[1]


def test_raw_corpus_row_would_have_rendered_no_assistant_turn() -> None:
    """Documents the bug this builder exists to avoid: passing rows through
    verbatim drops the `twin` turn and leaves nothing to learn from."""
    rendered = render_like_smollm3(make_row()["messages"])
    assert "<|im_start|>assistant" not in rendered


def test_prompt_is_the_first_two_turns() -> None:
    example = build_sft_example(make_row())
    assert [m["role"] for m in example["messages"]] == ["system", "tool", "assistant"]


def test_target_is_the_developer_text_not_the_twin_hypothesis() -> None:
    row = make_row("adjusted settings menu")
    row["messages"][2]["content"] = "<think>\n</think>\nsome twin guess"
    example = build_sft_example(row)
    assistant = example["messages"][-1]["content"]
    assert "adjusted settings menu" in assistant
    assert "some twin guess" not in assistant


def test_think_symmetry_wraps_the_target() -> None:
    example = build_sft_example(make_row("fixed it"))
    assert example["messages"][-1]["content"] == f"{THINK_PREFIX}fixed it"


def test_think_symmetry_can_be_disabled() -> None:
    example = build_sft_example(make_row("fixed it"), think_symmetry=False)
    assert example["messages"][-1]["content"] == "fixed it"


def test_disagreeing_target_copies_are_rejected() -> None:
    """Guessing which copy to train on would silently learn the wrong text."""
    row = make_row("subject a")
    row["feedback"]["free_text"] = "subject b"
    with pytest.raises(CorpusError, match="disagree"):
        extract_target(row)


def test_empty_target_is_rejected() -> None:
    row = make_row("")
    with pytest.raises(CorpusError, match="empty target"):
        build_sft_example(row)


def test_unrenderable_prompt_role_is_rejected() -> None:
    row = make_row()
    row["messages"][0]["role"] = "twin"
    with pytest.raises(CorpusError, match="drops"):
        build_sft_example(row)


def test_too_few_turns_is_rejected() -> None:
    row = make_row()
    row["messages"] = row["messages"][:1]
    with pytest.raises(CorpusError, match="prompt turns"):
        build_sft_example(row)


def test_build_dataset_over_many_rows() -> None:
    rows = [make_row(f"subject {n}", row_id=f"r{n}") for n in range(5)]
    built = list(build_sft_dataset(rows))
    assert len(built) == 5
    assert [e["id"] for e in built] == [f"r{n}" for n in range(5)]


def test_load_corpus_skips_blank_lines(tmp_path) -> None:
    path = tmp_path / "corpus.jsonl"
    path.write_text(
        json.dumps(make_row()) + "\n\n" + json.dumps(make_row(row_id="r2")) + "\n",
        encoding="utf-8",
    )
    assert len(list(load_corpus(path))) == 2


def test_load_corpus_reports_bad_json(tmp_path) -> None:
    path = tmp_path / "corpus.jsonl"
    path.write_text("{not json\n", encoding="utf-8")
    with pytest.raises(CorpusError, match="not valid JSON"):
        list(load_corpus(path))
