# ADR-P6-01 — One YAML File per Scenario, Authored in `configs/scenarios/`

**Status:** Accepted (Phase 6, 2026-08)
**Deciders:** Phase-6 engineering; three-lens cold-read council 2026-08-07

## Context

The scenario dataset is hand-authored, reviewed in pull requests, and expected to grow from ~20 to a few
hundred rows. It must be diff-reviewable, must not generate merge conflicts on every parallel edit, must
support nested predicates and multi-locale prompts, and must give scenario IDs a durable reservation so a
deprecated ID is never reused.

`configs/risk_rules.yaml` — a single flat file of six rules — is the incumbent.

## Decision

One YAML file per scenario at `configs/scenarios/SC-<CAT>-<NNN>.yaml`, globbed and compiled.

Authored knowledge lives in **`configs/`, not `data/`**. In this repository `data/` is the DVC plane: 21
stages write into it and `.gitignore` excludes most of it with narrow `!` re-allowances. Hand-authored,
PR-reviewed knowledge is *source*. `data/scenario_engine/` holds only what the compiler produces, plus
the clips.

A deprecated scenario's file **stays on disk** with `status: deprecated`. The file's continued existence
*is* the ID reservation — no separate registry is needed — and a validator hard-fails if any ID present
in the previously committed compiled artifact is absent from the new one.

Because rule-set correctness is a property of the *set* rather than of any row, the committed
`scenarios.compiled.json` is the semantic review surface in a pull request, exactly as `dvc.lock` is
elsewhere in this repo. A one-file diff must never be mistaken for the whole change.

## Alternatives considered

1. **A single `scenarios.yaml`.** The honest runner-up, and strictly better below ~40 scenarios — less
   tooling, one file to read. Rejected because two parallel pull requests appending to a list tail
   conflict every time, and blame becomes unreadable at scale. The migration between the two forms is
   mechanical in both directions, so this is reversible if the set stays small.
2. **CSV as master.** Rejected — see [ADR-P6-02](ADR-P6-02-csv-is-a-view.md).
3. **SQLite or a JSON database.** Rejected outright. A binary blob in git destroys diff review, which is
   the entire justification for a stakeholder-facing knowledge base; and this repo's configs are 40-50%
   comments, so dropping comment support drops the reasoning. There is no SQLite anywhere in the repo.
4. **OWL/RDF with a description-logic reasoner** — the literature-standard approach for ADL recognition
   in smart homes. Rejected for four concrete reasons: there is no class hierarchy to reason over (23
   flat classes; the semantic groupings are comments); a JVM reasoner with unbounded tableau expansion
   cannot live inside a 5 ms rule budget on a Pi; the composition layer it would serve is SmolVLM2, which
   ships disabled; and caregivers cannot review Turtle. The one thing OWL genuinely offers — automatic
   consistency detection — is stolen as three deterministic validators with better error messages.

## Consequences

- Positive: near-zero merge conflicts; `git log <file>` is a scenario's full history; per-file validation
  errors; ID immutability enforced structurally rather than by policy.
- Positive: no new dependency. Frozen dataclasses plus a hand-rolled `_validate()` match the house
  pattern; pydantic would be a new dependency and would need its own ADR.
- Constraint: the file count grows linearly, and the compiled artifact must be committed and reviewed or
  set-level semantics become invisible.
- Constraint: `dvc.yaml` must declare `configs/scenarios` as a directory **dep**, not a param — DVC has
  no "params from N files" concept. Adding a scenario correctly re-triggers the compile stage.

Related: [ADR-P6-02](ADR-P6-02-csv-is-a-view.md), [ADR-P6-06](ADR-P6-06-rule-hash-semantic-versioning.md)
