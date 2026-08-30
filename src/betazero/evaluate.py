"""A/B evaluation of the adapter against the untrained base model.

An adapter shipped without evidence is a claim, not a result. This module
produces the evidence, and it is built around three lessons that cost real
time on this project's earlier measurements.

**Average both arms over N runs.** The eval path samples rather than decoding
greedily, so a single run per arm measures sampling noise as much as it
measures the adapter.

**Judge against the PAIRED spread, not the independent one.** Both arms see the
same prompts under the same seed each run, so their run means drift together.
The independent standard error treats that shared drift as noise and hides real
effects.

**Prefer the exact permutation test when the arms share a per-row zero floor.**
Targets here are short commit messages, so a great many rows score zero in
*both* arms. That floor makes the per-row paired standard error badly behaved,
because most pairs contribute an exact zero difference that is structural
rather than sampled. The permutation test over run means makes no distributional
assumption and is exact for the run counts used here.

A noise-floor arm -- base against base under different seeds -- is reported
alongside, so a delta can be read against how much the harness moves when
nothing has changed at all.
"""

from __future__ import annotations

import itertools
import json
import statistics as st
from dataclasses import dataclass, field
from pathlib import Path


def _lcs_length(a: list[str], b: list[str]) -> int:
    """Length of the longest common subsequence."""
    if not a or not b:
        return 0
    previous = [0] * (len(b) + 1)
    for token_a in a:
        current = [0]
        for index, token_b in enumerate(b):
            if token_a == token_b:
                current.append(previous[index] + 1)
            else:
                current.append(max(current[index], previous[index + 1]))
        previous = current
    return previous[-1]


def rouge_l(prediction: str, reference: str) -> float:
    """ROUGE-L F1 over whitespace tokens.

    Implemented here rather than pulled in as a dependency: it is twenty lines,
    it removes a version-drift risk from the number the project reports, and it
    means the metric can be read by anyone auditing the result.
    """
    pred_tokens = prediction.split()
    ref_tokens = reference.split()
    if not pred_tokens or not ref_tokens:
        return 0.0
    lcs = _lcs_length(pred_tokens, ref_tokens)
    if lcs == 0:
        return 0.0
    precision = lcs / len(pred_tokens)
    recall = lcs / len(ref_tokens)
    return 2 * precision * recall / (precision + recall)


@dataclass
class ArmResult:
    """One arm's scores, one entry per run."""

    name: str
    run_means: list[float] = field(default_factory=list)
    per_row: list[list[float]] = field(default_factory=list)

    @property
    def mean(self) -> float:
        return st.fmean(self.run_means) if self.run_means else 0.0


def paired_delta(a: ArmResult, b: ArmResult) -> list[float]:
    """Per-run differences a - b. Requires the arms to be run-aligned."""
    if len(a.run_means) != len(b.run_means):
        raise ValueError(
            f"arms are not run-aligned: {a.name} has {len(a.run_means)} runs, "
            f"{b.name} has {len(b.run_means)}; a paired comparison is meaningless"
        )
    return [x - y for x, y in zip(a.run_means, b.run_means, strict=True)]


def paired_standard_error(deltas: list[float]) -> float:
    """Standard error of the mean paired difference."""
    if len(deltas) < 2:
        return float("nan")
    return st.stdev(deltas) / len(deltas) ** 0.5


def permutation_p_value(deltas: list[float]) -> float:
    """Exact two-sided sign-flip permutation test on the run deltas.

    Under the null that the adapter changes nothing, the sign of each run's
    delta is arbitrary. Enumerating all 2^N sign assignments gives an exact
    p-value with no distributional assumption -- which matters because the
    per-row scores are floored at zero and are nowhere near normal.
    """
    n = len(deltas)
    if n == 0:
        return float("nan")
    if n > 20:
        raise ValueError(f"exact enumeration is 2^{n}; use fewer runs or sample")

    observed = abs(st.fmean(deltas))
    at_least_as_extreme = sum(
        1
        for signs in itertools.product((1, -1), repeat=n)
        if abs(st.fmean([s * d for s, d in zip(signs, deltas, strict=True)]))
        >= observed - 1e-12
    )
    return at_least_as_extreme / 2**n


def summarise(treatment: ArmResult, control: ArmResult, floor: list[float]) -> dict:
    """Build the verdict, stating the effect against both the paired spread and
    the harness's own noise floor."""
    deltas = paired_delta(treatment, control)
    delta = st.fmean(deltas)
    sem = paired_standard_error(deltas)
    floor_spread = st.fmean([abs(f) for f in floor]) if floor else float("nan")

    return {
        "treatment": treatment.name,
        "control": control.name,
        "n_runs": len(deltas),
        "treatment_mean": treatment.mean,
        "control_mean": control.mean,
        "delta": delta,
        "paired_sem": sem,
        "delta_over_sem": (delta / sem) if sem and sem == sem and sem > 0 else None,
        "permutation_p": permutation_p_value(deltas),
        "noise_floor_mean_abs_delta": floor_spread,
        "delta_exceeds_noise_floor": (
            abs(delta) > floor_spread if floor_spread == floor_spread else None
        ),
        "per_run_deltas": deltas,
    }


def verdict_line(summary: dict) -> str:
    """A one-line reading of the result that does not overstate it."""
    delta = summary["delta"]
    p = summary["permutation_p"]
    if p != p:
        return "INCONCLUSIVE: no runs"
    if p > 0.05:
        return (
            f"INCONCLUSIVE: delta {delta:+.4f}, permutation p={p:.3f} "
            f"(n={summary['n_runs']}); cannot distinguish from no effect"
        )
    if not summary.get("delta_exceeds_noise_floor", True):
        return (
            f"INCONCLUSIVE: delta {delta:+.4f} is within the harness noise "
            f"floor ({summary['noise_floor_mean_abs_delta']:.4f}) despite "
            f"p={p:.3f}"
        )
    direction = "BETTER" if delta > 0 else "WORSE"
    return (
        f"{direction}: delta {delta:+.4f}, permutation p={p:.3f}, "
        f"{summary['delta_over_sem']:.1f}x paired SE (n={summary['n_runs']})"
    )


def write_report(summary: dict, path: str | Path) -> None:
    Path(path).write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )


def generate_arm(
    *,
    base_model: str,
    adapter: str | None,
    examples: list[dict],
    name: str,
    n_runs: int,
    seeds: list[int],
    max_new_tokens: int = 64,
    temperature: float = 0.7,
    max_length: int = 3072,
) -> ArmResult:
    """Generate and score one arm over `n_runs` sampled passes.

    Heavy imports are local so the statistics above stay testable without a
    CUDA stack.
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

    tokenizer = AutoTokenizer.from_pretrained(base_model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        device_map={"": 0} if torch.cuda.is_available() else None,
    )
    if adapter:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, adapter)
    model.eval()

    result = ArmResult(name=name)
    for run_index in range(n_runs):
        set_seed(seeds[run_index])
        scores = []
        for example in examples:
            prompt = tokenizer.apply_chat_template(
                example["messages"][:-1], tokenize=False, add_generation_prompt=True
            )
            inputs = tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=max_length,
            ).to(model.device)
            with torch.no_grad():
                output = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=True,
                    temperature=temperature,
                    top_p=0.95,
                    pad_token_id=tokenizer.pad_token_id,
                )
            completion = tokenizer.decode(
                output[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True
            )
            # Strip the reasoning wrapper the target shape uses, so the metric
            # compares the developer's text with the model's text.
            completion = completion.split("</think>")[-1].strip()
            scores.append(rouge_l(completion, example["target"]))
        result.per_row.append(scores)
        result.run_means.append(st.fmean(scores))
        print(f"  {name} run {run_index + 1}/{n_runs}: {result.run_means[-1]:.4f}")

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result


def main() -> int:
    import argparse

    from betazero.dataset import build_sft_dataset, load_corpus
    from betazero.train import TrainConfig, split_corpus

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--corpus", default=TrainConfig().corpus)
    parser.add_argument("--base-model", default=TrainConfig().base_model)
    parser.add_argument("--n-runs", type=int, default=6)
    parser.add_argument("--out", default="artifacts/eval_report.json")
    parser.add_argument("--seed", type=int, default=TrainConfig().seed)
    args = parser.parse_args()

    # The held-out split must be reproduced exactly as training made it, or the
    # evaluation is quietly scoring rows the adapter was trained on.
    config = TrainConfig(seed=args.seed)
    examples = list(build_sft_dataset(load_corpus(args.corpus)))
    _, heldout = split_corpus(
        examples, heldout_fraction=config.heldout_fraction, seed=config.seed
    )
    print(f"held-out examples: {len(heldout)}")

    # Both arms share the seed sequence, so each run pairs like with like.
    seeds = [args.seed + i for i in range(args.n_runs)]
    floor_seeds = [args.seed + 1000 + i for i in range(args.n_runs)]

    shared = {
        "base_model": args.base_model,
        "examples": heldout,
        "n_runs": args.n_runs,
    }
    treatment = generate_arm(
        adapter=args.adapter, name="betazero", seeds=seeds, **shared
    )
    control = generate_arm(adapter=None, name="base", seeds=seeds, **shared)
    # Noise floor: the same untrained model against itself under other seeds.
    floor_arm = generate_arm(
        adapter=None, name="base-floor", seeds=floor_seeds, **shared
    )

    summary = summarise(treatment, control, paired_delta(floor_arm, control))
    summary["heldout_ids"] = [e["id"] for e in heldout]
    summary["adapter"] = args.adapter
    write_report(summary, args.out)
    print(
        json.dumps({k: v for k, v in summary.items() if k != "heldout_ids"}, indent=2)
    )
    print(verdict_line(summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
