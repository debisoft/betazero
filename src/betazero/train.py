"""SFT training for BetaZero: supervised fine-tuning only, no RL stage.

Two decisions here are deliberate and worth keeping.

**Heavy imports are lazy.** `transformers`, `torch` and `peft` are imported
inside the functions that need them, so the pure logic below -- feature
building, masking, splitting -- can be imported and tested without a CUDA
stack present.

**The prompt is masked out of the loss.** Only the assistant turn contributes.
Training on the prompt as well would spend most of the gradient on the commit
diff, which the model is not being asked to reproduce.

The preflight check in `assert_target_is_learnable` exists because of a
specific past failure: an earlier pipeline rendered examples containing no
assistant turn at all, and trained happily on nothing. Loss fell, and the
resulting adapter was indistinguishable from an untrained one. Checking that
the loss actually covers the developer's text is cheap; discovering it did not,
after a training run and an eval, is not.
"""

from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

#: Ignore index the loss uses to skip a token.
IGNORE_INDEX = -100


@dataclass
class TrainConfig:
    """Everything that defines a run. Serialised beside the adapter."""

    base_model: str = "HuggingFaceTB/SmolLM3-3B"
    corpus: str = "data/corpus/D2_corpus.public.jsonl"
    output_dir: str = "artifacts/betazero-sft"

    # The v5-shape LoRA: small rank over all seven projections. r=4 keeps the
    # adapter committable to git -- ~14 MB saved in bf16.
    lora_r: int = 4
    lora_alpha: int = 8
    lora_dropout: float = 0.05
    target_modules: tuple[str, ...] = (
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    )

    learning_rate: float = 2e-4
    epochs: float = 3.0
    batch_size: int = 1
    grad_accum: int = 8
    max_length: int = 1024
    warmup_ratio: float = 0.03
    seed: int = 20260829

    heldout_fraction: float = 0.1
    #: Adapters are saved bf16, not fp32: same weights for training purposes,
    #: half the bytes, and the difference between ~29 MB and ~14 MB in git.
    save_dtype: str = "bfloat16"

    versions: dict[str, str] = field(default_factory=dict)

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, sort_keys=True)


def split_corpus(
    examples: list[dict], *, heldout_fraction: float, seed: int
) -> tuple[list[dict], list[dict]]:
    """Split into train and held-out sets with a seeded shuffle.

    The held-out set is proportional rather than a fixed count, so the split
    keeps its meaning if the corpus grows or shrinks.
    """
    if not 0.0 <= heldout_fraction < 1.0:
        raise ValueError(f"heldout_fraction must be in [0, 1), got {heldout_fraction}")

    shuffled = list(examples)
    random.Random(seed).shuffle(shuffled)
    n_heldout = int(round(len(shuffled) * heldout_fraction))
    # Never consume the whole corpus as held-out, and never claim a held-out
    # split that is actually empty.
    n_heldout = min(n_heldout, max(len(shuffled) - 1, 0))
    return shuffled[n_heldout:], shuffled[:n_heldout]


def build_features(tokenizer: Any, example: dict, *, max_length: int) -> dict:
    """Tokenise one example, masking the prompt out of the labels.

    `example["messages"]` is `[*prompt_turns, assistant_turn]`. The prompt is
    rendered with a generation prompt appended -- the same string serving
    sends -- and its token span is masked, so loss is computed only over the
    developer's own text.
    """
    messages = example["messages"]
    prompt_text = tokenizer.apply_chat_template(
        messages[:-1], tokenize=False, add_generation_prompt=True
    )
    full_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False
    )

    if not full_text.startswith(prompt_text):
        raise ValueError(
            f"example {example.get('id')!r}: rendered prompt is not a prefix of "
            f"the full rendering; cannot locate the assistant span to train on"
        )

    prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
    input_ids = tokenizer(
        full_text, add_special_tokens=False, truncation=True, max_length=max_length
    )["input_ids"]

    n_prompt = min(len(prompt_ids), len(input_ids))
    labels = [IGNORE_INDEX] * n_prompt + list(input_ids[n_prompt:])

    return {"input_ids": input_ids, "labels": labels, "id": example.get("id")}


def assert_target_is_learnable(features: dict, example: dict) -> None:
    """Fail loudly if a feature row carries no supervised tokens.

    This is the guard against the failure that wasted an earlier run: an
    example whose assistant turn vanished in templating trains on nothing while
    reporting a perfectly healthy loss curve.
    """
    supervised = [t for t in features["labels"] if t != IGNORE_INDEX]
    if not supervised:
        raise ValueError(
            f"example {example.get('id')!r}: no supervised tokens -- the "
            f"assistant turn did not survive templating, so this row would "
            f"train on nothing"
        )


def collate(batch: list[dict], *, pad_token_id: int) -> dict:
    """Pad a batch to equal length; padded label positions are ignored."""
    width = max(len(row["input_ids"]) for row in batch)
    return {
        "input_ids": [
            row["input_ids"] + [pad_token_id] * (width - len(row["input_ids"]))
            for row in batch
        ],
        "labels": [
            row["labels"] + [IGNORE_INDEX] * (width - len(row["labels"]))
            for row in batch
        ],
        "attention_mask": [
            [1] * len(row["input_ids"]) + [0] * (width - len(row["input_ids"]))
            for row in batch
        ],
    }


def write_run_config(config: TrainConfig, output_dir: Path, extra: dict) -> None:
    """Record the run beside the adapter, so a published adapter is reproducible."""
    payload = json.loads(config.to_json())
    payload.update(extra)
    (output_dir / "run_config.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )


def run(config: TrainConfig) -> Path:
    """Train the adapter. Heavy imports happen here, not at module import."""
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        Trainer,
        TrainingArguments,
        set_seed,
    )

    from betazero.dataset import build_sft_dataset, load_corpus

    set_seed(config.seed)
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(config.base_model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    examples = list(build_sft_dataset(load_corpus(config.corpus)))
    train_examples, heldout = split_corpus(
        examples, heldout_fraction=config.heldout_fraction, seed=config.seed
    )
    print(
        f"corpus {len(examples)} -> train {len(train_examples)} heldout {len(heldout)}"
    )

    features = []
    for example in train_examples:
        row = build_features(tokenizer, example, max_length=config.max_length)
        assert_target_is_learnable(row, example)
        features.append(row)

    supervised = sum(sum(1 for t in r["labels"] if t != IGNORE_INDEX) for r in features)
    print(f"supervised tokens: {supervised} across {len(features)} examples")
    if not supervised:
        raise SystemExit("refusing to train: no supervised tokens in the dataset")

    # low_cpu_mem_usage + an explicit device map: the plain
    # `from_pretrained(...).to(device)` path materialises the whole model in
    # host RAM first, which has already OOM-killed this class of box once.
    model = AutoModelForCausalLM.from_pretrained(
        config.base_model,
        dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        device_map={"": 0} if torch.cuda.is_available() else None,
    )
    model.config.use_cache = False

    model = get_peft_model(
        model,
        LoraConfig(
            r=config.lora_r,
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
            target_modules=list(config.target_modules),
            bias="none",
            task_type="CAUSAL_LM",
        ),
    )
    model.print_trainable_parameters()

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    trainer = Trainer(
        model=model,
        args=TrainingArguments(
            output_dir=str(output_dir / "checkpoints"),
            per_device_train_batch_size=config.batch_size,
            gradient_accumulation_steps=config.grad_accum,
            num_train_epochs=config.epochs,
            learning_rate=config.learning_rate,
            warmup_ratio=config.warmup_ratio,
            bf16=torch.cuda.is_available(),
            logging_steps=10,
            save_strategy="no",
            report_to=[],
            seed=config.seed,
        ),
        train_dataset=features,
        data_collator=lambda batch: {
            key: torch.tensor(value)
            for key, value in collate(
                batch, pad_token_id=tokenizer.pad_token_id
            ).items()
        },
    )
    result = trainer.train()

    # Save bf16 so the published adapter stays small enough to commit.
    adapter_dir = output_dir / "adapter"
    model = model.to(dtype=getattr(torch, config.save_dtype))
    model.save_pretrained(str(adapter_dir), safe_serialization=True)
    tokenizer.save_pretrained(str(adapter_dir))

    weights = adapter_dir / "adapter_model.safetensors"
    peak = torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0
    write_run_config(
        config,
        output_dir,
        {
            "n_examples": len(examples),
            "n_train": len(train_examples),
            "n_heldout": len(heldout),
            "supervised_tokens": supervised,
            "train_loss": result.training_loss,
            "peak_vram_bytes": peak,
            "adapter_bytes": weights.stat().st_size if weights.exists() else None,
            "heldout_ids": [e["id"] for e in heldout],
        },
    )
    print(f"adapter: {weights} ({weights.stat().st_size / 1e6:.1f} MB)")
    return adapter_dir


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Train the BetaZero SFT adapter.")
    defaults = TrainConfig()
    parser.add_argument("--corpus", default=defaults.corpus)
    parser.add_argument("--output-dir", default=defaults.output_dir)
    parser.add_argument("--base-model", default=defaults.base_model)
    parser.add_argument("--epochs", type=float, default=defaults.epochs)
    parser.add_argument("--learning-rate", type=float, default=defaults.learning_rate)
    parser.add_argument("--lora-r", type=int, default=defaults.lora_r)
    parser.add_argument("--seed", type=int, default=defaults.seed)
    args = parser.parse_args()

    run(
        TrainConfig(
            corpus=args.corpus,
            output_dir=args.output_dir,
            base_model=args.base_model,
            epochs=args.epochs,
            learning_rate=args.learning_rate,
            lora_r=args.lora_r,
            lora_alpha=args.lora_r * 2,
            seed=args.seed,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
