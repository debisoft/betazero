# Contributing

## Branching

`main` is protected. Every change — including documentation — arrives via a pull
request. There are no direct pushes to `main`.

Branch names follow `<type>/<short-slug>`:

```
feat/mcp-server-skeleton
fix/empty-line-range
docs/qodo-evidence
chore/ci-lint
```

## Commits

[Conventional Commits](https://www.conventionalcommits.org/): `type(scope): subject`.

```
feat(tools): add why_is_this_code_like_this
fix(kg): handle a path with no recorded reasoning
docs(readme): add Qodo review evidence
```

Keep the subject imperative and under ~72 characters. One logical change per
commit; one coherent concern per PR.

## Pull requests

1. Open (or claim) an issue describing the work.
2. Branch, commit, push, open a PR that closes that issue.
3. **Wait for the Qodo review.** Address the findings, or reply on the thread
   explaining why a finding is being dismissed.
4. Push the fixes and request a **follow-up Qodo review** on the same PR, so the
   thread shows the finding resolved.
5. Merge once the review is clean.

A PR that skips step 3 does not get merged, even if the code is right. The review
trail is part of the deliverable.

## Review expectations

Qodo reviews every PR automatically. A dismissed finding needs a one-line reason
in the thread — "false positive: `path` is validated by the caller" is enough.
Silence is not a dismissal.
