# ADR-P6-02 — CSV Is an Export-Only View; Lossless Round-Trip Is Not Attempted

**Status:** Accepted (Phase 6, 2026-08)
**Deciders:** Phase-6 engineering; three-lens cold-read council 2026-08-07

## Context

The Phase-6 brief specified a CSV schema so non-engineer stakeholders — clinicians, caregivers, domain
reviewers — can read and contribute to the scenario set. But scenario rows carry list-valued fields
(`required_objects`), nested predicate trees (`spatial_predicates`), and multi-locale prompts including
Devanagari that may contain commas.

Three concrete hazards were verified in this repository:

- `src/utils/report_utils.py:122` opens with `newline=""` and omits `lineterminator="\n"`, so
  `csv.DictWriter` emits CRLF — while its JSON sibling at `:88` documents at length why LF is mandatory
  (DVC hashes on-disk bytes; `.gitattributes` normalises to LF). A committed CSV would be permanently
  dirty in `git status`, failing release gate RG5 and diverging between the `ubuntu-latest` and
  `windows-latest` CI legs.
- The repository has a recorded cp1252 console incident, fixed by removing non-ASCII from output rather
  than by hardening the stream. There is no `PYTHONIOENCODING` or `sys.stdout.reconfigure` anywhere.
- Excel on Windows opens UTF-8-without-BOM as cp1252. A stakeholder who opens the file, changes one risk
  level, and saves has turned every Devanagari prompt into mojibake — and the mojibake is now the data.

Field-level, `optional_objects: []` and `optional_objects: null` are **both the empty string** in CSV.
That loss is irrecoverable.

## Decision

**CSV is a view, not a format.** The compiler emits `data/scenario_engine/build/scenarios.view.csv` and
**refuses to read any `.csv` under `build/`**. The `.view.` infix is deliberate, because CSV has no
comment syntax in which to say so in-band.

Where stakeholders must contribute, they edit a deliberately restricted
`scenarios.editable.csv` containing only flat scalars — `scenario_id`, `scenario_name`, `category`,
`risk_level`, `next_best_action`, `caregiver_channel`, `status`. **Prompt fields are excluded**, which is
what keeps Devanagari out of Excel's reach. The importer matches rows by `scenario_id` only, cannot
create or delete scenarios, writes back only allowlisted columns, and hard-fails on any change outside
the allowlist.

Byte specification, mandatory for both files: `utf-8-sig` for the editable file and `utf-8` for the
CI-checked view; explicit `lineterminator="\n"`; `csv.QUOTE_ALL` so one cell gaining a comma cannot
reshape neighbouring diffs; deterministic row order sorted by `scenario_id`; explicit column order;
embedded newlines **rejected at compile time** rather than escaped.

Three CI tests, not one — "round-trip stability" would imply losslessness we will not have:

1. Golden-file: `export(load(dir))` is byte-identical to the committed `.view.csv`. This is what catches
   the CRLF defect, and it must run on both matrix legs.
2. `import_editable(export_editable(x)) == x` for the restricted subset only.
3. A negative test pinning the exclusion: a scenario with nested predicates and a Devanagari prompt
   containing a comma survives the editable round-trip *because those fields are absent from it*.

`save_csv_report` is fixed in M1 as part of the same milestone.

## Alternatives considered

1. **CSV as the authoring master.** Rejected: nested predicates would require an undeclared in-cell
   mini-language, and a stakeholder typing `, ` instead of `;` breaks it silently.
2. **Full lossless round-trip via embedded JSON in cells.** Rejected: RFC 4180 quote-doubling makes the
   cells unreadable to the very audience CSV exists for, and Excel corrupts them invisibly.
3. **No CSV at all.** Rejected: it forfeits the stakeholder-review capability the brief asked for, which
   is genuinely valuable for a clinically-reviewed knowledge base.

## Consequences

- Positive: the corruption path from Excel back into the repository is closed by construction, not by
  discipline.
- Positive: the diff-reviewability that justified a human-readable format survives, because embedded
  newlines are rejected rather than escaped into broken-looking rows.
- Constraint: stakeholders cannot propose new scenarios via CSV — they propose them in review, and an
  engineer creates the file. This is the correct place for the ID-assignment decision anyway.
- Constraint: two CSV artifacts exist with different encodings and different rules. Both are documented
  in the header of the compiler CLI.

Related: [ADR-P6-01](ADR-P6-01-per-scenario-yaml-master.md)
