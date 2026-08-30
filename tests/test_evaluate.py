"""Tests for the A/B statistics.

The generation path needs a GPU; the statistics do not, and the statistics are
what decide whether the project claims a result. These pin them.
"""

from __future__ import annotations

import json

import pytest

from betazero.evaluate import (
    ArmResult,
    empirical_p_value,
    paired_delta,
    paired_standard_error,
    permutation_p_value,
    rouge_l,
    summarise,
    verdict_line,
    write_report,
)


def test_rouge_l_is_one_for_an_exact_match() -> None:
    assert rouge_l("fixed the parser", "fixed the parser") == pytest.approx(1.0)


def test_rouge_l_is_zero_with_no_overlap() -> None:
    assert rouge_l("aaa bbb", "ccc ddd") == 0.0


def test_rouge_l_rewards_subsequence_order() -> None:
    ordered = rouge_l("fixed the parser bug", "fixed the parser")
    scrambled = rouge_l("parser the fixed bug", "fixed the parser")
    assert ordered > scrambled


def test_rouge_l_handles_empty_strings() -> None:
    assert rouge_l("", "something") == 0.0
    assert rouge_l("something", "") == 0.0


def test_rouge_l_penalises_padding() -> None:
    """A model that emits a wall of text should not win by covering the target."""
    tight = rouge_l("fixed the parser", "fixed the parser")
    padded = rouge_l("fixed the parser " + "noise " * 20, "fixed the parser")
    assert padded < tight


def test_paired_delta_requires_aligned_runs() -> None:
    a = ArmResult("a", run_means=[0.1, 0.2])
    b = ArmResult("b", run_means=[0.1])
    with pytest.raises(ValueError, match="not run-aligned"):
        paired_delta(a, b)


def test_paired_standard_error_shrinks_with_more_runs() -> None:
    few = paired_standard_error([0.02, 0.04, 0.03])
    many = paired_standard_error([0.02, 0.04, 0.03] * 4)
    assert many < few


def test_permutation_p_is_one_when_deltas_are_symmetric() -> None:
    """A delta of zero cannot be more extreme than its own sign flips."""
    assert permutation_p_value([0.01, -0.01, 0.01, -0.01]) == pytest.approx(1.0)


def test_permutation_p_is_small_when_every_run_agrees() -> None:
    p = permutation_p_value([0.05, 0.06, 0.04, 0.05, 0.07, 0.06])
    assert p == pytest.approx(2 / 2**6)


def test_permutation_p_refuses_an_intractable_run_count() -> None:
    with pytest.raises(ValueError, match=r"2\^21"):
        permutation_p_value([0.01] * 21)


def test_summary_reports_inconclusive_for_a_wobbling_effect() -> None:
    treatment = ArmResult("betazero", run_means=[0.10, 0.06, 0.09, 0.05])
    control = ArmResult("base", run_means=[0.09, 0.08, 0.06, 0.08])
    summary = summarise(treatment, control, floor=[0.01, -0.02, 0.015, -0.01])
    assert "INCONCLUSIVE" in verdict_line(summary)


def test_summary_reports_better_for_a_consistent_effect() -> None:
    treatment = ArmResult("betazero", run_means=[0.15, 0.16, 0.14, 0.15, 0.17, 0.16])
    control = ArmResult("base", run_means=[0.09, 0.08, 0.09, 0.08, 0.09, 0.08])
    summary = summarise(
        treatment, control, floor=[0.005, -0.004, 0.003, -0.005, 0.002, -0.003]
    )
    assert "BETTER" in verdict_line(summary)
    assert summary["permutation_p"] < 0.05


def test_effect_inside_the_noise_floor_is_not_claimed() -> None:
    """A consistent but tiny delta must not be reported as a win when the
    harness moves further than that with nothing changed."""
    treatment = ArmResult(
        "betazero", run_means=[0.101, 0.102, 0.101, 0.103, 0.102, 0.101]
    )
    control = ArmResult("base", run_means=[0.100, 0.101, 0.100, 0.102, 0.101, 0.100])
    summary = summarise(
        treatment, control, floor=[0.05, -0.04, 0.06, -0.05, 0.04, -0.06]
    )
    assert summary["permutation_p"] < 0.05, "precondition: consistent direction"
    assert summary["delta_exceeds_noise_floor"] is False
    assert "INCONCLUSIVE" in verdict_line(summary)


def test_report_round_trips(tmp_path) -> None:
    treatment = ArmResult("betazero", run_means=[0.2, 0.3])
    control = ArmResult("base", run_means=[0.1, 0.1])
    summary = summarise(treatment, control, floor=[0.01, -0.01])
    path = tmp_path / "report.json"
    write_report(summary, path)
    assert json.loads(path.read_text())["treatment"] == "betazero"


def test_zero_paired_sem_does_not_crash_the_verdict() -> None:
    """Six identical positive deltas give p = 2/64 (significant) and a paired
    SEM of exactly zero. The directional branch must still render."""
    treatment = ArmResult("betazero", run_means=[0.2] * 6)
    control = ArmResult("base", run_means=[0.1] * 6)
    summary = summarise(treatment, control, floor=[0.001, -0.001] * 3)
    assert summary["permutation_p"] < 0.05, "precondition: significant"
    assert summary["delta_over_sem"] is None, "precondition: zero spread"
    assert "paired SE is zero" in verdict_line(summary)


def test_report_creates_its_parent_directory(tmp_path) -> None:
    """The write happens after three generation arms; a missing directory here
    would discard an expensive result at the last step."""
    treatment = ArmResult("betazero", run_means=[0.2, 0.3])
    control = ArmResult("base", run_means=[0.1, 0.1])
    summary = summarise(treatment, control, floor=[0.01, -0.01])
    destination = tmp_path / "artifacts" / "nested" / "report.json"
    write_report(summary, destination)
    assert destination.exists()


def test_empirical_p_uses_the_noise_floor_as_the_null() -> None:
    """A delta the floor never reaches is rarer than one it exceeds often."""
    floor = [0.01, -0.02, 0.015, -0.01, 0.02, -0.015]
    big = empirical_p_value(0.5, floor)
    small = empirical_p_value(0.001, floor)
    assert big < small
    assert big == pytest.approx(1 / (len(floor) + 1))


def test_empirical_p_is_nan_without_a_floor() -> None:
    assert empirical_p_value(0.1, []) != empirical_p_value(0.1, [])


def test_empirical_p_resolution_is_reported() -> None:
    """Its floor of 1/(N+1) is why it qualifies a reading rather than deciding
    one; the summary must say so rather than leave it to be inferred."""
    treatment = ArmResult("betazero", run_means=[0.2] * 6)
    control = ArmResult("base", run_means=[0.1] * 6)
    summary = summarise(treatment, control, floor=[0.001, -0.001] * 3)
    assert summary["empirical_p_resolution_limit"] == pytest.approx(1 / 7)
    assert (
        summary["empirical_p_vs_noise_floor"] >= summary["empirical_p_resolution_limit"]
    )
