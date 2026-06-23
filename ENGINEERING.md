# Compound engineering in FeatherSim

This project is built so that **every unit of engineering work makes the next one cheaper** — and so that
none of that leverage is lost when the working context is compacted, summarized, or handed to a fresh
session/agent. The codebase, the docs, the subagents, and CI form one self-reinforcing loop. This file is
the map of that loop: read it when you want to *extend the system that does the work*, not just do the work.

## Why "compound"

A one-off fix pays once. A fix that also (a) lands a regression test, (b) records the gotcha where the next
person will look, and (c) sharpens the gate that would have caught it — pays every time the area is touched
again. We optimize for the second kind. The bar for "done" is not "it works"; it's "it works, it can't
silently regress, and the lesson is captured where it compounds."

## The knowledge that survives `/compact`

LLM context windows get compacted and conversations get summarized — detail is lost. The defense is to push
every durable lesson **out of the conversation and into the repo**, where it's reloaded every session:

| Artifact | What it captures | How it compounds |
|---|---|---|
| [`LEARNINGS.md`](LEARNINGS.md) | Every time a green suite certified a property that didn't hold, and the fix. | The next change in that area starts forewarned — *never get bitten by the same class of bug twice.* |
| [`DECISIONS.md`](DECISIONS.md) | The *why* + tradeoff behind each architectural choice (append-only). | Future work doesn't relitigate settled calls or accidentally undo their rationale. |
| [`PLAN.md`](PLAN.md) | The phased roadmap + per-phase acceptance criteria. | Review judges the diff against *concrete* criteria, not a vibe; scope stays honest. |
| [`CLAUDE.md`](CLAUDE.md) | Conventions + how to run everything (the source of truth). | A fresh session is productive immediately, with the same conventions. |
| [`.claude/agents/`](.claude/agents) | Encoded reviewer / test-runner / docs-researcher / world-artist / … expertise. | Specialized judgment is reusable and improves over time (see the reviewer). |

These files are the project's long-term memory. When in doubt, **write the lesson down** — a sentence in
`LEARNINGS.md` outlives any chat.

## The loop (every phase)

1. **Refine acceptance criteria** in `PLAN.md` — what would prove this slice correct?
2. **Implement the smallest vertical slice** that satisfies them (use `docs-researcher` for unfamiliar APIs).
3. **`test-runner`** — nothing advances on red. Tests assert the *property*, across seeds / legs / configs,
   not one favorable sample (this is the #1 lesson in `LEARNINGS.md`).
4. **`reviewer`** — the adversarial last gate. It assumes the green suite lies until proven otherwise and
   *verifies with commands*. Address every CRITICAL/HIGH before commit; log deferred MEDIUM/LOW in `PLAN.md`.
5. **Capture the lesson** — update `DECISIONS.md` (why) and `LEARNINGS.md` (any gotcha), and sharpen an agent
   if review found a recurring blind spot.
6. **Commit** a small, focused change; tick `PLAN.md`. CI re-runs lint + the suite on push/PR.

The loop is the same whether a human or an agent drives it — that's what makes it survive a handoff.

## The automated gates

- **CI** ([`.github/workflows/ci.yml`](.github/workflows/ci.yml)): `ruff check` + the full pytest suite on
  every push and PR. Rendering tests skip gracefully without a GL backend, so the signal is reliable.
- **Lint** (`make lint`): import hygiene, dead code, and bug patterns (pyflakes/bugbear/pyupgrade) — the
  cheap checks that stop small rot from compounding.
- **Benchmarks** (`make bench`, `make bench-perception`): the headline results (collision-free + deadlock-
  free coordination; robust-vs-clean perception across difficulty) as **re-runnable numbers**, not claims —
  so a regression in a measured property is visible, not silent.
- **The reviewer agent**: the judgment gate that a test suite can't be — grounded in this repo's own
  `LEARNINGS.md` history of green-suite lies.

## Extending the compounding system

When you find yourself doing something the loop didn't catch or make easy:

- A bug a green suite hid → add the regression test **and** the `LEARNINGS.md` entry **and**, if it's a
  recurring class, a line in the `reviewer` agent so it hunts that pattern next time.
- A manual check you repeated → make it a script under `scripts/` and a `make` target (and wire it into CI
  if it's fast and deterministic).
- An API you had to re-learn → that's what `docs-researcher` is for; capture the snippet in `DECISIONS.md`
  if it shaped a choice.

Each of these turns a one-time cost into a permanent capability. That's the whole game.
