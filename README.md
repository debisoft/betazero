# BetaZero

**A developer's AI twin, small enough to commit to the repo.**

BetaZero is an SFT-only twin of a single developer: a LoRA adapter over
[SmolLM3-3B](https://huggingface.co/HuggingFaceTB/SmolLM3-3B), trained on that
developer's own git history, and served to coding agents as MCP tools through
the [TrueForge](https://github.com/truefoundry/trueforge) agent harness.

No RL, no reward model, no reinforcement stage. Supervised fine-tuning only —
hence *Beta*Zero. The whole twin is one ~29 MB file in this repository, so you
can clone it and run it.

Built for the [Agent Harness Hackathon](https://www.wemakedevs.org/hackathons/trueforge).

---

## The problem

When a developer leaves, the code stays and the reasoning goes with them. Git
records *what* changed and *when*; it rarely records *why*.

Coding agents inherit that blind spot. They read code, see something that looks
wrong, and "fix" it — reverting a deliberate decision nobody wrote down.

## What BetaZero does

BetaZero exposes a developer's recorded voice to any agent as MCP tools:

| Tool | Question it answers |
| --- | --- |
| `why_is_this_code_like_this(path, line_range)` | What did the developer say when this changed? |
| `ask_developer(question)` | How would this developer describe this work? |
| `who_knew_about(topic)` | Who made this call, and in which commits? |

A TrueForge agent about to refactor a file asks first, and revises its plan
instead of undoing a decision it never knew about.

## What it is trained on, and what that means

The twin in this repo is trained on **D2**, a developer with 443 commits to the
public [COBWEB-ca/cobweb2](https://github.com/COBWEB-ca/cobweb2) repository
(2008–2019). Each training row pairs a commit diff with the commit message that
developer wrote.

Be clear about what that teaches. The targets are commit messages — a median of
49 characters, 85% under 80. So BetaZero learns the developer's **voice and
intent at commit time**. It does not learn unwritten reasoning, because git
history does not contain unwritten reasoning. Recovering *that* requires
elicitation, not observation, and is out of scope here.

The corpus is derived entirely from public git history and is sanitised before
publication — see [`scripts/sanitize_corpus.py`](scripts/sanitize_corpus.py),
which redacts third-party contact details that appear in the upstream credits
screen and refuses to run if a redaction would alter a training label.

## Does it actually work?

An adapter that ships without evidence is a claim, not a result. This repo
publishes the A/B: BetaZero against the untrained base model, averaged over
repeated runs and judged against the **paired standard error**, with the
sampling noise floor stated alongside the effect.

See [`docs/evaluation.md`](docs/evaluation.md) — including the honest answer if
the effect turns out to be small.

## Status

Early. See the [issue tracker](../../issues) for planned work and
[CONTRIBUTING.md](CONTRIBUTING.md) for how changes get made.

## Qodo Code Review Evidence

> This project is entered in the hackathon's **Q Branch (code quality)** track.
> Every substantive change reaches `main` through a pull request reviewed by
> [Qodo](https://www.qodo.ai/). `main` is branch-protected; there are no direct
> pushes.
>
> _This section will link representative merged PRs, what Qodo surfaced, which
> findings were addressed or dismissed and why, and the follow-up review._

## License

Apache License 2.0 — see [LICENSE](LICENSE) and [NOTICE](NOTICE). The cobweb2
corpus derives from a public repository; see the sanitiser for how it is
prepared.
