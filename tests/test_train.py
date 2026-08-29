"""Tests for the SFT training logic.

These exercise the pure parts -- splitting, feature building, masking -- with a
stub tokeniser. `betazero.train` keeps torch/transformers/peft imports inside
`run()` precisely so this file can run without a CUDA stack.
"""

from __future__ import annotations

import json

import pytest

from betazero.train import (
    IGNORE_INDEX,
    TrainConfig,
    assert_target_is_learnable,
    build_features,
    collate,
    split_corpus,
    write_run_config,
)


class StubTokenizer:
    """Renders chat messages the way SmolLM3's template does, and tokenises on
    whitespace so token spans are easy to reason about in assertions."""

    def apply_chat_template(
        self, messages, tokenize=False, add_generation_prompt=False
    ):
        parts = [f"<{m['role']}> {m['content']} </{m['role']}>" for m in messages]
        if add_generation_prompt:
            parts.append("<assistant>")
        return " ".join(parts)

    def __call__(
        self, text, add_special_tokens=False, truncation=False, max_length=None
    ):
        ids = [abs(hash(tok)) % 1000 for tok in text.split()]
        if truncation and max_length is not None:
            ids = ids[:max_length]
        return {"input_ids": ids}


def make_example(target: str = "Class Path in Manifest", eid: str = "r1") -> dict:
    return {
        "id": eid,
        "messages": [
            {"role": "system", "content": "You are an AI Twin of the user."},
            {"role": "tool", "content": "code committed"},
            {"role": "assistant", "content": target},
        ],
    }


def test_prompt_is_masked_out_of_the_loss() -> None:
    features = build_features(StubTokenizer(), make_example(), max_length=128)
    labels = features["labels"]
    assert IGNORE_INDEX in labels, "prompt tokens must be masked"
    supervised = [t for t in labels if t != IGNORE_INDEX]
    assert supervised, "assistant tokens must be supervised"
    # Masked span must be a prefix: no interleaving.
    first_supervised = next(i for i, t in enumerate(labels) if t != IGNORE_INDEX)
    assert all(t == IGNORE_INDEX for t in labels[:first_supervised])


def test_labels_align_with_input_ids() -> None:
    features = build_features(StubTokenizer(), make_example(), max_length=128)
    assert len(features["labels"]) == len(features["input_ids"])


def test_supervised_tokens_come_from_the_assistant_turn() -> None:
    short = build_features(StubTokenizer(), make_example("one"), max_length=128)
    long = build_features(
        StubTokenizer(),
        make_example("a much longer commit message here"),
        max_length=128,
    )
    n_short = sum(1 for t in short["labels"] if t != IGNORE_INDEX)
    n_long = sum(1 for t in long["labels"] if t != IGNORE_INDEX)
    assert n_long > n_short


def test_missing_assistant_turn_is_caught() -> None:
    """The guard against the failure that wasted an earlier run."""
    features = {"labels": [IGNORE_INDEX, IGNORE_INDEX], "input_ids": [1, 2]}
    with pytest.raises(ValueError, match="train on nothing"):
        assert_target_is_learnable(features, make_example())


def test_learnable_example_passes_the_guard() -> None:
    features = build_features(StubTokenizer(), make_example(), max_length=128)
    assert_target_is_learnable(features, make_example())


def test_truncation_does_not_desync_labels() -> None:
    features = build_features(StubTokenizer(), make_example(), max_length=4)
    assert len(features["input_ids"]) == 4
    assert len(features["labels"]) == 4


def test_split_is_proportional_and_seeded() -> None:
    examples = [make_example(eid=f"r{n}") for n in range(100)]
    train, heldout = split_corpus(examples, heldout_fraction=0.1, seed=7)
    assert len(heldout) == 10
    assert len(train) == 90
    again = split_corpus(examples, heldout_fraction=0.1, seed=7)
    assert [e["id"] for e in again[1]] == [e["id"] for e in heldout]


def test_split_is_disjoint_and_lossless() -> None:
    examples = [make_example(eid=f"r{n}") for n in range(43)]
    train, heldout = split_corpus(examples, heldout_fraction=0.1, seed=1)
    ids_train = {e["id"] for e in train}
    ids_heldout = {e["id"] for e in heldout}
    assert not ids_train & ids_heldout
    assert ids_train | ids_heldout == {e["id"] for e in examples}


def test_split_never_consumes_the_whole_corpus() -> None:
    train, heldout = split_corpus([make_example()], heldout_fraction=0.9, seed=1)
    assert len(train) >= 1


def test_split_rejects_a_nonsense_fraction() -> None:
    with pytest.raises(ValueError, match="heldout_fraction"):
        split_corpus([make_example()], heldout_fraction=1.0, seed=1)


def test_collate_pads_and_ignores_padding_in_labels() -> None:
    batch = [
        {"input_ids": [1, 2, 3], "labels": [IGNORE_INDEX, 2, 3]},
        {"input_ids": [4, 5], "labels": [IGNORE_INDEX, 5]},
    ]
    out = collate(batch, pad_token_id=0)
    assert [len(r) for r in out["input_ids"]] == [3, 3]
    assert out["input_ids"][1] == [4, 5, 0]
    assert out["labels"][1][-1] == IGNORE_INDEX, "padding must not be supervised"
    assert out["attention_mask"][1] == [1, 1, 0]


def test_run_config_records_the_run(tmp_path) -> None:
    write_run_config(TrainConfig(), tmp_path, {"train_loss": 1.23, "n_train": 401})
    payload = json.loads((tmp_path / "run_config.json").read_text())
    assert payload["lora_r"] == 4
    assert payload["lora_alpha"] == 8
    assert payload["save_dtype"] == "bfloat16"
    assert payload["train_loss"] == 1.23
    assert payload["n_train"] == 401
